"""Real local MLflow SDK: attempt identity, source times, replay and privacy."""

from datetime import datetime
import hashlib
import json
import os
import shutil
import sys

import pytest

from loopx_hpc.campaign import CampaignStore, canonical
from loopx_hpc.cli import demo_spec
from loopx_hpc.mlflow_tracking import sync_mlflow


T0 = "2026-01-02T01:00:00+00:00"
T1 = "2026-01-02T01:01:00+00:00"
T2 = "2026-01-02T01:02:00+00:00"


@pytest.fixture(autouse=True)
def sdk_privacy(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "true")


def store_at(path, objective="Bounded tracker test"):
    store = CampaignStore(path)
    spec = demo_spec()
    spec["objective"] = objective
    store.initialize(spec)
    experiment = store.add_experiment({"x": 0}, rationale="Measure baseline")
    return store, experiment["id"]


def transition(store, identifier, status, kind, when, *, payload=None):
    with store.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET token='test-attempt',status=?,updated_at=? WHERE id=?",
            (status, when, identifier),
        )
        store.event(
            connection, kind, identifier, {"token": "test-attempt", **(payload or {})}
        )
        connection.execute(
            "UPDATE events SET created_at=? WHERE sequence=(SELECT MAX(sequence) FROM events)",
            (when,),
        )


def finish(store, identifier):
    record = store.get_experiment(identifier)
    artifacts = []
    directory = store.root / "results"
    directory.mkdir()
    for name, content in (
        ("config.json", record["config"]),
        ("result.json", {"metrics": {"loss": 0.25}}),
    ):
        data = canonical(content).encode()
        (directory / name).write_bytes(data)
        artifacts.append(
            {"path": "results/" + name, "sha256": hashlib.sha256(data).hexdigest()}
        )
    transition(
        store,
        identifier,
        "succeeded",
        "attempt_finished",
        T2,
        payload={
            "status": "succeeded",
            "metrics": {"loss": 0.25},
            "artifacts": artifacts,
            "started_at": T1,
            "finished_at": T2,
        },
    )
    with store.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET metrics=?,artifacts=? WHERE id=?",
            (canonical({"loss": 0.25}), canonical(artifacts), identifier),
        )


def client_for(receipt):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    return MlflowClient(
        tracking_uri=receipt["tracking_uri"], registry_uri=receipt["tracking_uri"]
    )


def test_preview_and_remote_gates_do_not_create_tracking_state(tmp_path):
    store, identifier = store_at(tmp_path / "source")
    target = tmp_path / "tracker"
    result = sync_mlflow(store, target)
    assert result["preview"] is True
    assert result["pending_experiment_ids"] == [identifier]
    assert result["runs"] == []
    assert not target.exists()
    with pytest.raises(ValueError, match="allow_external"):
        sync_mlflow(
            store, target, tracking_uri="https://not-contacted.invalid", execute=True
        )
    with pytest.raises(ValueError, match="without credentials"):
        sync_mlflow(
            store,
            target,
            tracking_uri="https://secret:not-public@not-contacted.invalid",
            allow_external=True,
        )
    assert not target.exists()


def test_real_lifecycle_one_run_original_times_params_artifacts_and_copy_replay(
    tmp_path, monkeypatch
):
    store, identifier = store_at(tmp_path / "source")
    target = tmp_path / "tracker"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://must-not-contact.invalid")
    transition(
        store,
        identifier,
        "submitting",
        "attempt_reserved",
        T0,
        payload={"backend": "slurm"},
    )
    first = sync_mlflow(store, target, execute=True)
    client = client_for(first)
    run_id = first["runs"][0]["run_id"]
    run = client.get_run(run_id)
    assert run.info.status == "RUNNING"
    assert run.info.start_time == int(datetime.fromisoformat(T0).timestamp() * 1000)
    transition(
        store,
        identifier,
        "running",
        "scheduler_observed",
        T1,
        payload={
            "backend": "slurm",
            "job_id": "123",
            "observation": {
                "status": "running",
                "native_state": "RUNNING",
                "terminal": False,
                "raw_output": "PRIVATE-SCHEDULER-OUTPUT",
                "env": {"SECRET": "PRIVATE-ENV"},
            },
        },
    )
    from loopx_hpc.scheduler_execution import _schema

    with store.transaction() as connection:
        _schema(connection)
        connection.execute(
            "INSERT INTO scheduler_attempts(experiment_id,token,backend,profile,profile_digest,packet_digest,job_id,observation,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                identifier,
                "test-attempt",
                "slurm",
                '{"env":{"SECRET":"PRIVATE-ENV"}}',
                "a" * 64,
                "b" * 64,
                "123",
                "{}",
                T0,
                T1,
            ),
        )
    running = sync_mlflow(store, target, execute=True)
    assert running["runs"][0]["run_id"] == run_id
    finish(store, identifier)
    before = store.database.read_bytes()
    complete = sync_mlflow(store, target, execute=True)
    assert store.database.read_bytes() == before
    assert complete["runs"][0]["run_id"] == run_id
    assert not complete["runs"][0]["reused"]
    assert sync_mlflow(store, target, execute=True)["runs"][0]["reused"]
    run = client.get_run(run_id)
    assert len(client.search_runs([run.info.experiment_id])) == 1
    assert run.info.status == "FINISHED"
    assert run.info.end_time == int(datetime.fromisoformat(T2).timestamp() * 1000)
    assert run.data.tags["loopx.started_at"] == T1
    assert run.data.tags["loopx.finished_at"] == T2
    assert run.data.tags["loopx.scheduler.job_id"] == "123"
    assert run.data.tags["loopx.scheduler.backend"] == "slurm"
    assert run.data.tags["loopx.scheduler.profile_digest"] == "a" * 64
    assert run.data.params["config.x"] == "0"
    assert "provenance.code_revision" in run.data.params
    history = client.get_metric_history(run_id, "loss")
    assert len(history) == 1 and history[0].value == 0.25
    assert history[0].timestamp == int(datetime.fromisoformat(T2).timestamp() * 1000)
    files = list((target / "mlflow-lifecycle" / "artifacts").rglob("*.json"))
    assert {file.name for file in files} >= {"config.json", "result.json"}
    assert all(
        "PRIVATE-SCHEDULER-OUTPUT" not in file.read_text()
        and "PRIVATE-ENV" not in file.read_text()
        for file in files
    )
    copied_root = tmp_path / "copied-source"
    shutil.copytree(store.root, copied_root)
    assert (
        sync_mlflow(CampaignStore(copied_root), target, execute=True)["runs"][0][
            "run_id"
        ]
        == run_id
    )
    # Local durable success does not turn later tampered artifact bytes valid.
    (copied_root / "results" / "result.json").write_text('{"metrics":{"loss":999}}')
    with pytest.raises(ValueError, match="verified successful"):
        sync_mlflow(CampaignStore(copied_root), target, execute=True)
    assert os.environ["MLFLOW_TRACKING_URI"] == "https://must-not-contact.invalid"


def test_unknown_and_failed_are_not_finished_and_study_namespace_isolated(tmp_path):
    target = tmp_path / "tracker"
    first, identifier = store_at(tmp_path / "first", "First study")
    transition(first, identifier, "unknown", "attempt_reserved", T0)
    unknown = sync_mlflow(first, target, execute=True)
    client = client_for(unknown)
    assert client.get_run(unknown["runs"][0]["run_id"]).info.status == "RUNNING"
    transition(
        first,
        identifier,
        "failed",
        "attempt_finished",
        T2,
        payload={"status": "failed", "failure": "PRIVATE-RAW-FAILURE"},
    )
    failed = sync_mlflow(first, target, execute=True)
    run = client.get_run(failed["runs"][0]["run_id"])
    assert run.info.status == "FAILED"
    assert run.data.tags["loopx.source_status"] == "failed"
    second, second_id = store_at(tmp_path / "second", "Second frozen study")
    assert second_id == identifier  # Identical old ID namespace is not enough.
    transition(second, second_id, "running", "attempt_reserved", T0)
    other = sync_mlflow(second, target, execute=True)
    assert other["runs"][0]["run_id"] != failed["runs"][0]["run_id"]
    assert other["runs"][0]["attempt_key"] != failed["runs"][0]["attempt_key"]
    assert all(
        "PRIVATE-RAW-FAILURE" not in path.read_text()
        for path in (target / "mlflow-lifecycle" / "artifacts").rglob("*.json")
    )


def test_real_sdk_crash_after_create_or_metric_is_recovered_without_duplicate(
    tmp_path, monkeypatch
):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    store, identifier = store_at(tmp_path / "source")
    transition(store, identifier, "submitting", "attempt_reserved", T0)
    finish(store, identifier)
    target = tmp_path / "tracker"
    original_create = MlflowClient.create_run
    calls = []

    def create_then_crash(self, *args, **kwargs):
        result = original_create(self, *args, **kwargs)
        calls.append(result.info.run_id)
        raise RuntimeError("simulated lost create response")

    monkeypatch.setattr(MlflowClient, "create_run", create_then_crash)
    with pytest.raises(RuntimeError, match="lost create response"):
        sync_mlflow(store, target, execute=True)
    monkeypatch.setattr(MlflowClient, "create_run", original_create)
    original_log = MlflowClient.log_metric

    def log_then_crash(self, *args, **kwargs):
        original_log(self, *args, **kwargs)
        raise RuntimeError("simulated lost metric response")

    monkeypatch.setattr(MlflowClient, "log_metric", log_then_crash)
    with pytest.raises(RuntimeError, match="lost metric response"):
        sync_mlflow(store, target, execute=True)
    monkeypatch.setattr(MlflowClient, "log_metric", original_log)
    receipt = sync_mlflow(store, target, execute=True)
    client = client_for(receipt)
    assert receipt["runs"][0]["run_id"] == calls[0]
    assert len(client.get_metric_history(calls[0], "loss")) == 1
    assert len(client.search_runs([client.get_run(calls[0]).info.experiment_id])) == 1
    with store.transaction() as connection:
        connection.execute(
            "UPDATE events SET created_at=? WHERE kind='attempt_reserved'", (T1,)
        )
    with pytest.raises(ValueError, match="history changed"):
        sync_mlflow(store, target, execute=True)


def test_creation_intent_without_discoverable_run_fails_closed(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    store, identifier = store_at(tmp_path / "source")
    transition(store, identifier, "submitting", "attempt_reserved", T0)
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("simulated ambiguous outage")

    monkeypatch.setattr(MlflowClient, "create_run", unavailable)
    with pytest.raises(RuntimeError, match="outage"):
        sync_mlflow(store, tmp_path / "tracker", execute=True)
    with pytest.raises(RuntimeError, match="creation unresolved"):
        sync_mlflow(store, tmp_path / "tracker", execute=True)
    assert len(calls) == 1


@pytest.mark.parametrize("option", ["execute", "allow_external", "include_artifacts"])
@pytest.mark.parametrize("value", ["false", 1, None])
def test_truthy_permission_values_fail_before_reads_or_sdk_effects(
    tmp_path, monkeypatch, option, value
):
    store, _ = store_at(tmp_path / "source")
    target = tmp_path / "must-not-create"

    def unexpected_read(*args):
        pytest.fail("invalid permission option reached source reads or SDK setup")

    monkeypatch.setattr("loopx_hpc.mlflow_tracking._read", unexpected_read)
    with pytest.raises(ValueError, match=option + " must be a boolean"):
        sync_mlflow(store, target, **{option: value})
    assert not target.exists()


def test_cli_tracking_sync_preview_and_real_sdk_execution(tmp_path, capsys):
    from loopx_hpc.cli import main

    store, identifier = store_at(tmp_path / "source")
    transition(store, identifier, "submitting", "attempt_reserved", T0)
    finish(store, identifier)
    destination = tmp_path / "tracker"
    argv = [
        "--root",
        str(store.root),
        "tracking-sync",
        "--destination",
        str(destination),
    ]
    before = store.database.read_bytes()
    assert main(argv) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview"] and not destination.exists()
    assert main(argv + ["--execute"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    run_id = receipt["runs"][0]["run_id"]
    client = client_for(receipt)
    assert client.get_run(run_id).info.status == "FINISHED"
    assert client.get_run(run_id).data.metrics == {"loss": 0.25}
    assert main(argv + ["--execute"]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["runs"][0]["run_id"] == run_id
    assert again["runs"][0]["reused"]
    assert store.database.read_bytes() == before


@pytest.mark.parametrize("backend", ["pbs", "slurm"])
@pytest.mark.parametrize("fail_worker", [False, True])
def test_actual_scheduler_batch_worker_to_local_mlflow(
    tmp_path, monkeypatch, backend, fail_worker
):
    from test_scheduler_execution import SchedulerDouble, profile
    from loopx_hpc.scheduler_execution import SchedulerExecutor

    spec = demo_spec()
    if fail_worker:
        spec["command"] = [
            sys.executable,
            "-c",
            "raise SystemExit(3)",
            "{config}",
            "{result}",
        ]
    store = CampaignStore(tmp_path / "source")
    store.initialize(spec)
    experiment_id = store.add_experiment(
        {"x": 2}, rationale="Actual scheduler tracking acceptance"
    )["id"]
    scheduler = SchedulerDouble(backend)
    monkeypatch.setattr("loopx_hpc.scheduler_execution._run_cli", scheduler)
    executor = SchedulerExecutor(store)
    executor.submit(experiment_id, profile(backend), execute=True)
    target = tmp_path / "tracker"
    first = sync_mlflow(store, target, execute=True)
    run_id = first["runs"][0]["run_id"]
    client = client_for(first)
    assert client.get_run(run_id).info.status == "RUNNING"
    assert (
        client.get_run(run_id).data.tags["loopx.scheduler.job_id"] == scheduler.job_id
    )
    scheduler.state = "running"
    assert executor.reconcile(experiment_id)["status"] == "running"
    assert sync_mlflow(store, target, execute=True)["runs"][0]["run_id"] == run_id
    worker = scheduler.work()
    assert worker.returncode == (1 if fail_worker else 0), worker.stderr.decode()
    completed = executor.reconcile(experiment_id)
    assert completed["status"] == ("failed" if fail_worker else "succeeded")
    worker_receipt = json.loads((scheduler.directory / "receipt.json").read_text())
    before = store.database.read_bytes()
    complete = sync_mlflow(store, target, execute=True)
    assert store.database.read_bytes() == before
    assert complete["runs"][0]["run_id"] == run_id
    run = client.get_run(run_id)
    assert run.info.status == ("FAILED" if fail_worker else "FINISHED")
    assert run.data.tags["loopx.source_status"] == completed["status"]
    assert run.data.tags["loopx.started_at"] == worker_receipt["started_at"]
    assert run.data.tags["loopx.finished_at"] == worker_receipt["finished_at"]
    assert run.data.tags["loopx.finished_at_basis"] == "worker_receipt"
    assert run.info.end_time == int(
        datetime.fromisoformat(worker_receipt["finished_at"]).timestamp() * 1000
    )
    if fail_worker:
        assert run.data.metrics == {}
        assert client.list_artifacts(run_id, "evidence") == []
    else:
        assert run.data.metrics == {"loss": 0}
        history = client.get_metric_history(run_id, "loss")
        assert len(history) == 1 and history[0].timestamp == run.info.end_time
        assert len(client.list_artifacts(run_id, "evidence")) == 2
    assert sync_mlflow(store, target, execute=True)["runs"][0]["reused"]
    assert len(client.search_runs([run.info.experiment_id])) == 1
    assert scheduler.submissions == 1
