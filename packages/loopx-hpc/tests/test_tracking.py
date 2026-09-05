"""Local projection identity, optional-backend and no-upload contracts."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from loopx_hpc.tracking import _check_local_artifact_uri, export_tracking


def snapshot():
    config = {"learning_rate": 0.01, "model": {"width": 16}, "seed": 7}
    return {
        "campaign": {
            "id": "synthetic-study",
            "objective": "Test reproducible exports",
            "metric": {"name": "loss", "direction": "minimize"},
        },
        "experiments": [
            {
                "id": "baseline",
                "config": config,
                "config_hash": hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest(),
                "status": "completed",
                "metrics": {"loss": 0.25},
                "artifacts": [{"path": "outputs/metrics.json", "sha256": "a" * 64}],
            }
        ],
    }


def test_json_receipt_replay_is_noop_and_source_stays_authority(tmp_path):
    source = snapshot()
    result = export_tracking(source, tmp_path)
    files = {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob("*.json")}
    again = export_tracking(source, tmp_path)
    assert again == {**result, "reused": True}
    assert files == {
        str(path): path.stat().st_mtime_ns for path in tmp_path.rglob("*.json")
    }
    assert result["source_authority"] == "campaign_store"
    assert result["local_only"] is True
    assert result["online_delivery_validated"] is False
    assert json.loads(Path(result["projection_path"]).read_text()) == source


def test_run_keys_stable_across_updates_but_projection_is_revisioned(tmp_path):
    source = snapshot()
    first = export_tracking(source, tmp_path)
    source["experiments"][0]["metrics"]["loss"] = 0.2
    second = export_tracking(source, tmp_path)
    assert first["run_keys"] == second["run_keys"]
    assert first["snapshot_hash"] != second["snapshot_hash"]
    key = first["run_keys"][0]
    assert (
        first["backend_runs"][key]["projection_id"]
        != second["backend_runs"][key]["projection_id"]
    )
    assert (
        json.loads(Path(first["projection_path"]).read_text())["experiments"][0][
            "metrics"
        ]["loss"]
        == 0.25
    )


@pytest.mark.parametrize("change", ["config", "config_hash"])
def test_config_identity_is_immutable_across_backends(tmp_path, change):
    source = snapshot()
    export_tracking(source, tmp_path)
    changed = deepcopy(source)
    if change == "config":
        changed["experiments"][0]["config"]["seed"] = 8
    else:
        changed["experiments"][0]["config_hash"] = "different-hash"
    with pytest.raises(ValueError, match="conflicting immutable config"):
        export_tracking(changed, tmp_path, "mlflow")


def test_campaign_namespaces_do_not_collide(tmp_path):
    source = snapshot()
    first = export_tracking(source, tmp_path)
    source["campaign"]["id"] = "a-different-study"
    source["experiments"][0]["config"]["seed"] = 8
    second = export_tracking(source, tmp_path)
    assert first["run_keys"] != second["run_keys"]


def test_experiment_order_is_not_a_new_snapshot(tmp_path):
    source = snapshot()
    second = deepcopy(source["experiments"][0])
    second["id"] = "candidate"
    source["experiments"].append(second)
    first = export_tracking(source, tmp_path)
    source["experiments"].reverse()
    assert export_tracking(source, tmp_path) == {**first, "reused": True}


def test_artifact_inventory_does_not_read_or_copy_source(tmp_path):
    private = tmp_path / "private-input"
    private.write_text("do not copy raw source bodies")
    source = snapshot()
    source["experiments"][0]["artifacts"][0]["path"] = str(private)
    destination = tmp_path / "export"
    export_tracking(source, destination)
    assert all(
        "do not copy raw source bodies" not in file.read_text()
        for file in destination.rglob("*.json")
    )


def test_corrupt_projection_cannot_be_reused_as_verified(tmp_path):
    result = export_tracking(snapshot(), tmp_path)
    Path(result["projection_path"]).write_text("{}")
    with pytest.raises(ValueError, match="corrupted"):
        export_tracking(snapshot(), tmp_path)


def test_mlflow_artifact_uri_must_remain_local_and_contained(tmp_path):
    _check_local_artifact_uri((tmp_path / "run" / "artifacts").as_uri(), tmp_path)
    for uri in (
        "https://tracker.invalid/artifacts",
        "s3://bucket/artifacts",
        tmp_path.parent.as_uri(),
        "file://remote-host/share",
    ):
        with pytest.raises(ValueError, match="local"):
            _check_local_artifact_uri(uri, tmp_path)


@pytest.mark.parametrize(
    "bad_metrics",
    [{"loss": float("nan")}, {"loss": "0.2"}, {"loss": True}, {"loss": None}],
)
def test_invalid_metrics_fail_before_writes(tmp_path, bad_metrics):
    source = snapshot()
    source["experiments"][0]["metrics"] = bad_metrics
    destination = tmp_path / "export"
    with pytest.raises(ValueError):
        export_tracking(source, destination)
    assert not destination.exists()


def test_metric_value_records_are_supported(tmp_path):
    source = snapshot()
    source["experiments"][0]["metrics"] = {
        "loss": {"value": 0.25, "unit": "dimensionless"}
    }
    assert export_tracking(source, tmp_path)["local_only"]


def test_duplicate_identity_and_unknown_backends_are_rejected(tmp_path):
    source = snapshot()
    source["experiments"].append(deepcopy(source["experiments"][0]))
    with pytest.raises(ValueError, match="duplicate experiment"):
        export_tracking(source, tmp_path)
    with pytest.raises(ValueError, match="backend must"):
        export_tracking(snapshot(), tmp_path, "online")
    with pytest.raises(ValueError, match="local directory"):
        export_tracking(snapshot(), "https://tracker.invalid")


def test_missing_optional_dependency_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(RuntimeError, match="optional 'wandb'"):
        export_tracking(snapshot(), tmp_path, "wandb")


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


def test_wandb_forces_offline_without_upload_and_reuses_unchanged_records(
    tmp_path, monkeypatch
):
    calls = []
    runs = []

    def initialize(**kwargs):
        assert os.environ["WANDB_MODE"] == "offline"
        assert os.environ["WANDB_ERROR_REPORTING"] == "false"
        calls.append(kwargs)
        run = FakeRun()
        runs.append(run)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=initialize, Settings=lambda **kwargs: kwargs),
    )
    source = snapshot()
    first = export_tracking(source, tmp_path, "wandb")
    assert export_tracking(source, tmp_path, "wandb-offline")["reused"]
    candidate = deepcopy(source["experiments"][0])
    candidate["id"] = "candidate"
    source["experiments"].append(candidate)
    export_tracking(source, tmp_path, "wandb-offline")
    assert len(calls) == 2  # The original experiment is not spooled again.
    assert all(
        call["mode"] == "offline" and call["save_code"] is False for call in calls
    )
    assert all(
        call["settings"]["disable_git"] and call["settings"]["x_disable_meta"]
        for call in calls
    )
    assert all(run.finished for run in runs)
    assert runs[0].logged == [({"loss": 0.25}, 0)]
    assert runs[0].summary["loopx_run_key"] == first["run_keys"][0]


def test_wandb_does_not_reuse_an_active_online_run(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=object()))
    with pytest.raises(RuntimeError, match="existing W&B run"):
        export_tracking(snapshot(), tmp_path, "wandb-offline")


def test_interrupted_wandb_export_cannot_silently_duplicate_spool(
    tmp_path, monkeypatch
):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic SDK failure")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fail, Settings=lambda **kwargs: kwargs),
    )
    with pytest.raises(RuntimeError, match="synthetic SDK failure"):
        export_tracking(snapshot(), tmp_path, "wandb-offline")
    with pytest.raises(RuntimeError, match="interrupted"):
        export_tracking(snapshot(), tmp_path, "wandb-offline")
    assert len(calls) == 1


def test_real_mlflow_sqlite_is_local_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    # An inherited remote URI must not influence this explicit local adapter.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://must-not-contact.invalid")
    result = export_tracking(snapshot(), tmp_path, "mlflow")
    assert export_tracking(snapshot(), tmp_path, "mlflow")["reused"]
    key = result["run_keys"][0]
    details = result["backend_runs"][key]
    assert details["tracking_uri"].startswith("sqlite:///")
    client = MlflowClient(
        tracking_uri=details["tracking_uri"], registry_uri=details["tracking_uri"]
    )
    run = client.get_run(details["run_id"])
    assert run.data.metrics == {"loss": 0.25}
    assert run.data.tags["loopx.run_key"] == key
    assert run.info.status == "FINISHED"
    assert len(client.search_runs([run.info.experiment_id])) == 1
    artifacts = client.list_artifacts(details["run_id"])
    assert [entry.path for entry in artifacts] == [details["projection_id"] + ".json"]


@pytest.mark.skipif(
    os.environ.get("LOOPX_HPC_TEST_WANDB") != "1",
    reason="opt in to local W&B SDK IPC test",
)
def test_real_wandb_offline_spool_and_replay(tmp_path, monkeypatch):
    if importlib.util.find_spec("wandb") is None:
        pytest.skip("optional W&B SDK is not installed")
    # Deliberately hostile inherited defaults cannot turn this into online sync.
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "true")
    result = export_tracking(snapshot(), tmp_path, "wandb-offline")
    again = export_tracking(snapshot(), tmp_path, "wandb-offline")
    assert again == {**result, "reused": True}
    assert len(list(tmp_path.rglob("*.wandb"))) == 1
    assert not list(tmp_path.rglob("wandb-metadata.json"))
    assert not list(tmp_path.rglob("diff.patch"))
    assert not list(tmp_path.rglob("requirements.txt"))
    assert all(not list(directory.rglob("*")) for directory in tmp_path.rglob("code"))
    assert result["local_only"] is True
    assert result["online_delivery_validated"] is False
    assert os.environ["WANDB_MODE"] == "online"  # Caller environment is restored.
