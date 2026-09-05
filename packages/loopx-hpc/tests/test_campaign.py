import json
import sqlite3
import subprocess
import sys
import time

import pytest

from loopx_hpc.campaign import CampaignStore
from loopx_hpc.cli import demo, demo_spec
from loopx_hpc.local import LocalExecutor


@pytest.fixture
def store(tmp_path):
    result = CampaignStore(tmp_path / "campaign")
    result.initialize(demo_spec())
    return result


def wait(store, experiment_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = LocalExecutor(store).reconcile(experiment_id)
        if record["status"] in ("failed", "succeeded", "unknown"):
            return record
        time.sleep(0.05)
    pytest.fail("local worker did not finish within bounded test window")


def test_freeze_and_bounds(store):
    spec = demo_spec()
    store.initialize(spec)
    spec["objective"] = "changed"
    with pytest.raises(ValueError, match="immutable"):
        store.initialize(spec)
    for config in ({"x": 100}, {"x": True}, {"x": 0, "shell": "bad"}):
        with pytest.raises(ValueError):
            store.add_experiment(config, rationale="candidate")
    for x in (0, 1, 2):
        store.add_experiment({"x": x}, rationale="candidate")
    with pytest.raises(ValueError, match="budget"):
        store.add_experiment({"x": 3}, rationale="candidate")


def test_preview_and_duplicate_submission(store):
    experiment = store.add_experiment({"x": 0}, rationale="baseline")
    assert LocalExecutor(store).submit(experiment["id"])["preview"]
    assert not (store.root / "runs").exists()
    first = LocalExecutor(store).submit(experiment["id"], execute=True)
    second = LocalExecutor(CampaignStore(store.root)).submit(
        experiment["id"], execute=True
    )
    assert first["token"] == second["token"]
    completed = wait(store, experiment["id"])
    assert completed["status"] == "succeeded"
    assert completed["metrics"] == {"loss": 4}
    assert store.verify_record(completed)
    with sqlite3.connect(store.database) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM events WHERE kind='attempt_reserved'"
            ).fetchone()[0]
            == 1
        )


def test_uncertain_launch_never_retries_and_reserves_capacity(store):
    a = store.add_experiment({"x": 0}, rationale="first")
    b = store.add_experiment({"x": 1}, rationale="second")
    c = store.add_experiment({"x": 2}, rationale="third")
    with store.transaction() as db:
        db.execute(
            "UPDATE experiments SET status='submitting',token='lost-ack',updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (a["id"],),
        )
        db.execute(
            "UPDATE experiments SET status='unknown',token='other' WHERE id=?",
            (b["id"],),
        )
    assert LocalExecutor(store).reconcile(a["id"])["status"] == "unknown"
    assert LocalExecutor(store).submit(a["id"], execute=True)["token"] == "lost-ack"
    with pytest.raises(ValueError, match="concurrency"):
        LocalExecutor(store).submit(c["id"], execute=True)
    assert not (store.root / "runs").exists()


def test_proposal_staleness_atomic_budget_and_lineage(store):
    context = store.context()
    proposal = {
        "id": "first",
        "context_digest": context["context_digest"],
        "runtime": "codex",
        "rationale": "first candidates",
        "evidence_ids": [],
        "configs": [{"x": 0}, {"x": 1}],
    }
    accepted = store.apply_proposal(proposal)
    assert store.apply_proposal(proposal) == accepted
    changed = {**proposal, "runtime": "claude-code"}
    with pytest.raises(ValueError, match="reused"):
        store.apply_proposal(changed)
    with pytest.raises(ValueError, match="stale"):
        store.apply_proposal({**proposal, "id": "stale"})
    current = store.context()
    overbudget = {
        **proposal,
        "id": "too-many",
        "context_digest": current["context_digest"],
        "configs": [{"x": 2}, {"x": 3}],
    }
    with pytest.raises(ValueError, match="budget"):
        store.apply_proposal(overbudget)
    assert len(store.experiments()) == 2  # Whole proposal rolled back.
    assert store.context()["context_digest"] == current["context_digest"]


def test_result_tampering_excluded_from_context(store):
    experiment = store.add_experiment({"x": 0}, rationale="baseline")
    LocalExecutor(store).submit(experiment["id"], execute=True)
    completed = wait(store, experiment["id"])
    artifact = store.root / completed["artifacts"][-1]["path"]
    artifact.write_text('{"metrics":{"loss":-99}}')
    assert store.context()["eligible_evidence_ids"] == []
    with pytest.raises(ValueError, match="verified"):
        store.add_experiment(
            {"x": 1}, rationale="bad evidence", parent_ids=[experiment["id"]]
        )


def test_type_changing_config_tampering_rejected(tmp_path):
    spec = demo_spec()
    body = 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(\'{"x":true}\'); pathlib.Path(sys.argv[2]).write_text(\'{"metrics":{"loss":1}}\')'
    spec["command"] = [sys.executable, "-c", body, "{config}", "{result}"]
    store = CampaignStore(tmp_path)
    store.initialize(spec)
    record = store.add_experiment({"x": 1}, rationale="type-sensitive configuration")
    LocalExecutor(store).submit(record["id"], execute=True)
    assert wait(store, record["id"])["status"] == "failed"


@pytest.mark.parametrize(
    "body",
    [
        "pass",
        "import sys; sys.exit(3)",
        'import pathlib,sys; pathlib.Path(sys.argv[2]).write_text(\'{"metrics":{"loss":NaN}}\')',
    ],
)
def test_exit_zero_is_not_a_valid_result(tmp_path, body):
    spec = demo_spec()
    spec["command"] = [sys.executable, "-c", body, "{config}", "{result}"]
    store = CampaignStore(tmp_path)
    store.initialize(spec)
    record = store.add_experiment({"x": 0}, rationale="invalid result test")
    LocalExecutor(store).submit(record["id"], execute=True)
    assert wait(store, record["id"])["status"] == "failed"
    assert store.context()["eligible_evidence_ids"] == []


def test_cli_controller_process_can_exit(store):
    record = store.add_experiment({"x": 2}, rationale="separate controller")
    command = [sys.executable, "-m", "loopx_hpc.cli", "--root", str(store.root)]
    launched = subprocess.run(
        command + ["submit", "--experiment", record["id"], "--execute"],
        capture_output=True,
        text=True,
        check=True,
    )
    token = json.loads(launched.stdout)["token"]
    assert wait(CampaignStore(store.root), record["id"])["metrics"]["loss"] == 0
    replay = subprocess.run(
        command + ["submit", "--experiment", record["id"], "--execute"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout)["token"] == token


def test_bounded_adaptive_demo_resume(tmp_path):
    assert demo(tmp_path)["preview"]
    assert not tmp_path.joinpath("campaign.sqlite3").exists()
    result = demo(tmp_path, execute=True)
    context = result["context"]
    assert len(context["experiments"]) == 3
    assert context["remaining_experiments"] == 0
    winner = next(
        r for r in context["experiments"] if r["id"] == context["incumbent_id"]
    )
    assert winner["config"] == {"x": 2}
    assert len(winner["parents"]) == 2
    assert result["model_calls"] == result["remote_jobs"] == 0
    repeat = demo(tmp_path, execute=True)
    assert repeat["context"] == context
