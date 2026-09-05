"""Adversarial scheduler lifecycle checks; scheduler CLIs are always fixtures."""

from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from loopx_hpc.campaign import CampaignStore
from loopx_hpc.cli import demo_spec
from loopx_hpc.local import LocalExecutor
from loopx_hpc.scheduler_execution import SchedulerExecutor, validate_profile
import loopx_hpc.scheduler_execution as execution


def study(tmp_path):
    spec = demo_spec()
    spec["limits"]["max_concurrent"] = 1
    store = CampaignStore(tmp_path / "campaign")
    store.initialize(spec)
    first = store.add_experiment({"x": 0}, rationale="Preregistered first trial")
    second = store.add_experiment({"x": 1}, rationale="Independent bounded trial")
    return store, first, second


def profile(backend="slurm"):
    return {
        "scheduler": backend,
        "resources": {"job_name": "safety", "nodes": 1, "walltime": "00:01:00"},
        "environment": {"name": "test", "configured": True},
        "python": sys.executable,
        "launcher": [],
    }


def install_cli(monkeypatch, *, ack="417", observation="417|RUNNING|0:0"):
    calls = []
    outputs = {"ack": ack, "observation": observation, "cancel": "", "identity": ""}

    def run(argv, cwd):
        calls.append(list(argv))
        key = (
            "ack"
            if argv[0] in {"sbatch", "qsub"}
            else "cancel"
            if argv[0] in {"scancel", "qdel"}
            else "identity"
            if argv[0] == "squeue"
            else "observation"
        )
        response = outputs[key]
        if isinstance(response, BaseException):
            raise response
        return subprocess.CompletedProcess(argv, 0, response, "")

    monkeypatch.setattr(execution, "_run_cli", run)
    return calls, outputs


@pytest.mark.parametrize("ack", ["", "417\n418", "417;other-cluster", "not-a-job"])
def test_ambiguous_ack_never_resubmits_or_releases_capacity(tmp_path, monkeypatch, ack):
    store, first, second = study(tmp_path)
    calls, _ = install_cli(monkeypatch, ack=ack)
    executor = SchedulerExecutor(store)
    original = executor.submit(first["id"], profile(), execute=True)
    assert original["status"] == "unknown"
    assert len(calls) == 1
    for _ in range(2):
        replay = executor.submit(first["id"], profile(), execute=True)
        assert replay["token"] == original["token"]
        assert executor.reconcile(first["id"])["status"] == "unknown"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="concurrency"):
        executor.submit(second["id"], profile(), execute=True)
    assert (
        LocalExecutor(store).submit(first["id"], execute=True)["token"]
        == original["token"]
    )
    assert store.context()["eligible_evidence_ids"] == []


def test_submission_timeout_is_uncertain_not_a_safe_retry(tmp_path, monkeypatch):
    store, first, _ = study(tmp_path)
    calls, _ = install_cli(monkeypatch, ack=subprocess.TimeoutExpired(["sbatch"], 30))
    executor = SchedulerExecutor(store)
    submitted = executor.submit(first["id"], profile(), execute=True)
    assert submitted["status"] == "unknown"
    assert (
        executor.submit(first["id"], profile(), execute=True)["token"]
        == submitted["token"]
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "observation",
    ["418|COMPLETED|0:0", "417|COMPLETED+|0:0", "417|COMPLETED|0:0\n418|COMPLETED|0:0"],
)
def test_other_or_ambiguous_accounting_cannot_become_evidence(
    tmp_path, monkeypatch, observation
):
    store, first, _ = study(tmp_path)
    install_cli(monkeypatch, observation=observation)
    executor = SchedulerExecutor(store)
    executor.submit(first["id"], profile(), execute=True)
    result = executor.reconcile(first["id"])
    assert result["status"] == "unknown"
    assert result["metrics"] == {}
    assert result["artifacts"] == []
    assert store.context()["eligible_evidence_ids"] == []


def test_native_success_without_worker_receipt_stays_unknown(tmp_path, monkeypatch):
    store, first, _ = study(tmp_path)
    calls, _ = install_cli(monkeypatch, observation="417|COMPLETED|0:0")
    executor = SchedulerExecutor(store)
    executor.submit(first["id"], profile(), execute=True)
    result = executor.reconcile(first["id"])
    assert result["status"] == "unknown"
    assert result["failure"] == "terminal_receipt_unverified"
    assert result["metrics"] == {}
    assert store.context()["eligible_evidence_ids"] == []
    with store.transaction() as connection:
        before = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    executor.reconcile(first["id"])
    with store.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == before
    assert [c[0] for c in calls] == ["sbatch", "sacct", "sacct"]


def test_cancel_ack_does_not_prove_exit_and_replay_has_no_effect(tmp_path, monkeypatch):
    store, first, second = study(tmp_path)
    calls, outputs = install_cli(monkeypatch)
    executor = SchedulerExecutor(store)
    executor.submit(first["id"], profile(), execute=True)
    executor.reconcile(first["id"])
    token = store.get_experiment(first["id"])["token"]
    outputs["identity"] = f"417|lx-{token[:12]}\n"
    preview = executor.cancel(first["id"])
    assert preview["preview"] is True
    assert len(calls) == 2
    cancelled = executor.cancel(first["id"], execute=True)
    assert cancelled["acknowledged"] is True
    assert cancelled["terminal_confirmed"] is False
    assert store.get_experiment(first["id"])["status"] == "running"
    assert executor.cancel(first["id"], execute=True)["replayed"] is True
    assert len(calls) == 4
    with pytest.raises(ValueError, match="concurrency"):
        executor.submit(second["id"], profile(), execute=True)
    outputs["observation"] = "417|CANCELLED by 1001|0:15"
    result = executor.reconcile(first["id"])
    assert result["status"] == "failed"
    assert result["metrics"] == {}
    assert store.context()["eligible_evidence_ids"] == []
    assert len([c for c in calls if c[0] == "scancel"]) == 1


def test_changed_profile_cannot_rebind_existing_attempt(tmp_path, monkeypatch):
    store, first, _ = study(tmp_path)
    calls, _ = install_cli(monkeypatch)
    executor = SchedulerExecutor(store)
    original = executor.submit(first["id"], profile(), execute=True)
    changed = deepcopy(profile())
    changed["resources"]["nodes"] = 2
    with pytest.raises(ValueError, match="another backend or profile"):
        executor.submit(first["id"], changed, execute=True)
    assert store.get_experiment(first["id"])["token"] == original["token"]
    assert len(calls) == 1


@pytest.mark.parametrize("execute", ["false", "true", 0, 1, None])
def test_submit_requires_an_actual_boolean_execute_flag(tmp_path, monkeypatch, execute):
    store, first, _ = study(tmp_path)
    calls, _ = install_cli(monkeypatch)
    with pytest.raises(ValueError, match="boolean"):
        SchedulerExecutor(store).submit(first["id"], profile(), execute=execute)
    assert calls == []
    assert store.get_experiment(first["id"])["status"] == "planned"


@pytest.mark.parametrize("name", ["PBS_JOBID", "SLURM_JOB_ID", "SLURM_JOBID"])
def test_profiles_cannot_replace_scheduler_owned_job_identity(name):
    selected = profile()
    selected["environment"]["variables"] = {name: "417"}
    with pytest.raises(ValueError, match="scheduler|identity|reserved"):
        validate_profile(selected)


def run_worker(store, record, *, job_id="417"):
    directory = store.root / "runs" / record["id"] / record["token"]
    env = {**os.environ, "SLURM_JOB_ID": job_id}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loopx_hpc.batch_worker",
            str(directory / "packet.json"),
        ],
        env=env,
        cwd=store.root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return directory, result


def test_lost_ack_recovers_only_after_real_receipt_and_exact_accounting(
    tmp_path, monkeypatch
):
    store, first, _ = study(tmp_path)
    calls, outputs = install_cli(monkeypatch, ack="", observation="417|RUNNING|0:0")
    executor = SchedulerExecutor(store)
    submitted = executor.submit(first["id"], profile(), execute=True)
    assert submitted["status"] == "unknown"
    directory, process = run_worker(store, submitted)
    assert process.returncode == 0, process.stderr
    # Worker success alone cannot release the scheduler allocation or establish evidence.
    assert executor.reconcile(first["id"])["status"] == "running"
    assert store.context()["eligible_evidence_ids"] == []
    outputs["observation"] = "417|COMPLETED|0:0"
    finished = executor.reconcile(first["id"])
    assert finished["status"] == "succeeded"
    assert store.verify_record(finished)
    assert (directory / "receipt.json").is_file()
    assert [c[0] for c in calls].count("sbatch") == 1


@pytest.mark.parametrize(
    "mutation",
    ["token", "job_id", "packet_digest", "result", "config_type", "artifact_digest"],
)
def test_real_worker_receipt_or_artifact_drift_is_not_evidence(
    tmp_path, monkeypatch, mutation
):
    store, first, _ = study(tmp_path)
    install_cli(monkeypatch, observation="417|COMPLETED|0:0")
    executor = SchedulerExecutor(store)
    submitted = executor.submit(first["id"], profile(), execute=True)
    directory, process = run_worker(store, submitted)
    assert process.returncode == 0, process.stderr
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if mutation == "result":
        (directory / "result.json").write_text('{"metrics":{"loss":-1000}}')
    elif mutation == "config_type":
        (directory / "config.json").write_text('{"x":0.0}')
    elif mutation == "artifact_digest":
        receipt["artifacts"][1]["sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt))
    else:
        receipt[mutation] = "418" if mutation == "job_id" else "0" * 64
        receipt_path.write_text(json.dumps(receipt))
    result = executor.reconcile(first["id"])
    assert result["status"] == "unknown"
    assert result["metrics"] == {}
    assert result["artifacts"] == []
    assert store.context()["eligible_evidence_ids"] == []


def test_compute_redelivery_does_not_reexecute_or_replace_receipt(
    tmp_path, monkeypatch
):
    store, first, _ = study(tmp_path)
    install_cli(monkeypatch)
    submitted = SchedulerExecutor(store).submit(first["id"], profile(), execute=True)
    directory, process = run_worker(store, submitted)
    assert process.returncode == 0, process.stderr
    original_receipt = (directory / "receipt.json").read_bytes()
    original_result = (directory / "result.json").read_bytes()
    _, repeated = run_worker(store, submitted)
    assert repeated.returncode == 75
    assert (directory / "receipt.json").read_bytes() == original_receipt
    assert (directory / "result.json").read_bytes() == original_result


def test_batch_worker_does_not_write_controller_database(tmp_path, monkeypatch):
    store, first, _ = study(tmp_path)
    install_cli(monkeypatch)
    submitted = SchedulerExecutor(store).submit(first["id"], profile(), execute=True)
    before = store.database.read_bytes()
    _, process = run_worker(store, submitted)
    assert process.returncode == 0, process.stderr
    assert store.database.read_bytes() == before
    assert store.get_experiment(first["id"])["status"] == "submitting"


def test_worker_success_cannot_override_native_scheduler_failure(tmp_path, monkeypatch):
    store, first, _ = study(tmp_path)
    install_cli(monkeypatch, observation="417|FAILED|1:0")
    executor = SchedulerExecutor(store)
    submitted = executor.submit(first["id"], profile(), execute=True)
    _, process = run_worker(store, submitted)
    assert process.returncode == 0, process.stderr
    result = executor.reconcile(first["id"])
    assert result["status"] == "failed"
    assert result["metrics"] == {}
    assert result["artifacts"] == []
    assert store.context()["eligible_evidence_ids"] == []


@pytest.mark.parametrize("execute", ["false", "true", 0, 1, None])
def test_cancel_requires_an_actual_boolean_execute_flag(tmp_path, monkeypatch, execute):
    store, first, _ = study(tmp_path)
    calls, _ = install_cli(monkeypatch)
    executor = SchedulerExecutor(store)
    executor.submit(first["id"], profile(), execute=True)
    with pytest.raises(ValueError, match="boolean"):
        executor.cancel(first["id"], execute=execute)
    assert [c[0] for c in calls] == ["sbatch"]
