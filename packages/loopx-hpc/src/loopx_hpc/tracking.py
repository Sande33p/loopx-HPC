"""Local-only tracker projections; the campaign store remains authoritative.

Each experiment has a stable run key and immutable projection revisions. This
is deliberate: W&B cannot resume offline runs. Tracker runs describe exported
facts, not new experiments, execution approval, or validated online delivery.
No optional SDK is imported until its backend is explicitly selected.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit


SCHEMA = "loopx_hpc_tracking_receipt_v1"
_SDK_ENVIRONMENT_LOCK = threading.RLock()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tracking-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _private_sdk_environment(extra: dict[str, str] | None = None) -> Iterator[None]:
    # Explicitly override SDK telemetry settings during import and local export.
    settings = {
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "DO_NOT_TRACK": "true",
        **(extra or {}),
    }
    with _SDK_ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    result = {}
    for name, item in metrics.items():
        value = item.get("value") if isinstance(item, dict) else item
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {name!r} must have a numeric value")
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        result[name] = float(value)
    return result


def _normalize(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a mapping")
    # A JSON round trip rejects non-JSON state and decouples caller mutation.
    try:
        data = json.loads(_canonical(snapshot))
    except (ValueError, TypeError) as error:
        raise ValueError("snapshot must contain finite JSON data") from error
    campaign = data.get("campaign")
    experiments = data.get("experiments")
    if (
        not isinstance(campaign, dict)
        or not isinstance(campaign.get("id"), str)
        or not campaign["id"]
    ):
        raise ValueError("snapshot requires campaign.id")
    if not isinstance(experiments, list):
        raise ValueError("snapshot requires an experiments list")
    seen = set()
    for record in experiments:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
        ):
            raise ValueError("each experiment requires an id")
        if record["id"] in seen:
            raise ValueError("duplicate experiment id")
        seen.add(record["id"])
        if not isinstance(record.get("config"), dict):
            raise ValueError("each experiment requires its resolved config")
        if not isinstance(record.get("config_hash"), str) or not record["config_hash"]:
            raise ValueError("each experiment requires config_hash")
        if not isinstance(record.get("metrics", {}), dict):
            raise ValueError("experiment metrics must be a mapping")
        _metric_values(record.get("metrics", {}))
        artifacts = record.get("artifacts", [])
        if not isinstance(artifacts, list) or any(
            not isinstance(item, dict) for item in artifacts
        ):
            raise ValueError("experiment artifacts must be an inventory list")
    data["experiments"] = sorted(experiments, key=lambda row: row["id"])
    return data


def _mlflow_export(
    destination: Path,
    campaign: dict[str, Any],
    record: dict[str, Any],
    run_key: str,
    projection_id: str,
    projection_path: Path,
) -> dict[str, Any]:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise RuntimeError(
            "MLflow export requires the optional 'mlflow' dependency"
        ) from error

    database = destination / "mlflow" / "tracking.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + str(database)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_name = "loopx-hpc-" + _digest(campaign["id"])[:24]
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            experiment_name,
            artifact_location=(database.parent / "artifacts").as_uri(),
            tags={"loopx.campaign_id": campaign["id"], "loopx.local_only": "true"},
        )
    )
    if experiment:
        _check_local_artifact_uri(
            experiment.artifact_location, database.parent / "artifacts"
        )
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`loopx.projection_id` = '{projection_id}'",
    )
    if len(runs) > 1:
        raise ValueError("ambiguous MLflow projection identity")
    run = (
        runs[0]
        if runs
        else client.create_run(
            experiment_id,
            tags={
                "loopx.run_key": run_key,
                "loopx.projection_id": projection_id,
                "loopx.experiment_id": record["id"],
                "loopx.config_hash": record["config_hash"],
                "loopx.local_only": "true",
                "mlflow.runName": record["id"],
            },
        )
    )
    run_id = run.info.run_id
    _check_local_artifact_uri(run.info.artifact_uri, database.parent / "artifacts")
    if run.data.tags.get("loopx.export_complete") != "true":
        # Config values are kept exactly in an artifact, not flattened/truncated.
        client.log_param(run_id, "config_hash", record["config_hash"])
        client.set_tag(
            run_id, "loopx.source_status", str(record.get("status", "unknown"))
        )
        for name, value in _metric_values(record.get("metrics", {})).items():
            # Fixed timestamps make a replay after an interrupted export stable.
            client.log_metric(run_id, name, value, timestamp=0, step=0)
        client.log_artifact(run_id, str(projection_path))
        client.set_terminated(run_id, status="FINISHED")
        client.set_tag(run_id, "loopx.export_complete", "true")
    return {"run_id": run_id, "tracking_uri": uri, "mode": "local-sqlite"}


def _check_local_artifact_uri(uri: str, root: Path) -> None:
    """A reused SQLite store must not smuggle in a remote artifact repository."""
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(
            "MLflow artifacts must stay in the destination-owned local directory"
        )
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(
            "MLflow artifact location escapes the local tracking destination"
        )


def _wandb_export(
    destination: Path,
    campaign: dict[str, Any],
    record: dict[str, Any],
    run_key: str,
    projection_id: str,
    projection_path: Path,
) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B offline export requires the optional 'wandb' dependency"
        ) from error
    if getattr(wandb, "run", None) is not None:
        raise RuntimeError(
            "finish the existing W&B run before exporting an isolated offline projection"
        )

    spool = destination / "wandb-offline" / projection_id
    spool.mkdir(parents=True, exist_ok=True)
    # A previous unreceipted spool may be partial. Offline resume is unsupported,
    # so fail closed instead of silently creating a second SDK run after a crash.
    marker = spool / "export-started.json"
    if marker.exists():
        raise RuntimeError(
            "W&B export interrupted; inspect the local spool before retrying in a new destination"
        )
    _write_json(marker, {"projection_id": projection_id, "run_key": run_key})
    run = wandb.init(
        project="loopx-hpc-local",
        id=projection_id,
        name=record["id"],
        group=_digest(campaign["id"])[:24],
        dir=str(spool),
        mode="offline",
        config={
            "resolved_config": record["config"],
            "config_hash": record["config_hash"],
        },
        save_code=False,
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            disable_job_creation=True,
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
            x_disable_viewer=True,
            x_save_requirements=False,
            console="off",
            silent=True,
        ),
    )
    if run is None:
        raise RuntimeError("W&B did not initialize an offline run")
    try:
        run.log(_metric_values(record.get("metrics", {})), step=0)
        run.summary.update(
            {
                "loopx_run_key": run_key,
                "loopx_projection_id": projection_id,
                "loopx_source_status": record.get("status", "unknown"),
                "loopx_projection_path": str(projection_path),
                "local_only": True,
                "artifact_inventory": record.get("artifacts", []),
            }
        )
    finally:
        run.finish()
    return {"run_id": projection_id, "spool_path": str(spool), "mode": "offline"}


def export_tracking(snapshot: dict, destination: Path, backend: str = "json") -> dict:
    """Export local fact projections, preserving identity and replay receipts.

    SDK runs are immutable projection revisions, linked by stable experiment
    run keys. Artifact inventories are exported as data; source artifact paths
    are never opened, uploaded, or trusted as tracker authority. Destinations
    are private local storage and must not be shared as public-safe exports.
    """
    backend = "wandb-offline" if backend == "wandb" else backend
    if backend not in {"json", "mlflow", "wandb-offline"}:
        raise ValueError("backend must be json, mlflow, or wandb-offline")
    if "://" in str(destination):
        raise ValueError("tracking destination must be a local directory, not a URI")
    data = _normalize(snapshot)
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".tracking.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaign = data["campaign"]
        campaign_key = _digest(campaign["id"])
        campaign_dir = root / "campaigns" / campaign_key
        registry_path = campaign_dir / "identities.json"
        registry = (
            json.loads(registry_path.read_text())
            if registry_path.exists()
            else {"configs": {}}
        )
        for record in data["experiments"]:
            identity = {
                "config": record["config"],
                "config_hash": record["config_hash"],
            }
            prior = registry["configs"].get(record["id"])
            if prior is not None and prior != identity:
                raise ValueError(
                    f"conflicting immutable config for experiment {record['id']!r}"
                )
            registry["configs"][record["id"]] = identity
        snapshot_hash = _digest(data)
        projection_path = campaign_dir / "snapshots" / f"{snapshot_hash}.json"
        receipt_path = campaign_dir / "receipts" / backend / f"{snapshot_hash}.json"
        if receipt_path.exists():
            if (
                not projection_path.exists()
                or _digest(json.loads(projection_path.read_text())) != snapshot_hash
            ):
                raise ValueError("tracking snapshot projection is missing or corrupted")
            return {**json.loads(receipt_path.read_text()), "reused": True}
        _write_json(registry_path, registry)
        _write_json(projection_path, data)
        exported = {}
        sdk_environment = {}
        if backend == "wandb-offline":
            sdk_environment = {
                "WANDB_MODE": "offline",
                "WANDB_ERROR_REPORTING": "false",
                "WANDB_SENTRY_DSN": "",
                "WANDB_CONFIG_DIR": str(root / "wandb-offline" / "config"),
                "WANDB_CACHE_DIR": str(root / "wandb-offline" / "cache"),
                "WANDB_DATA_DIR": str(root / "wandb-offline" / "data"),
            }
        with _private_sdk_environment(sdk_environment):
            for record in data["experiments"]:
                run_key = _digest([campaign["id"], record["id"]])
                projection_id = _digest([run_key, record])
                record_path = campaign_dir / "runs" / run_key / f"{projection_id}.json"
                backend_receipt = record_path.parent / backend / f"{projection_id}.json"
                if backend_receipt.exists():
                    exported[run_key] = json.loads(backend_receipt.read_text())
                    continue
                _write_json(
                    record_path,
                    {
                        "campaign_id": campaign["id"],
                        "run_key": run_key,
                        "experiment": record,
                    },
                )
                details = {"mode": "local-json", "run_id": projection_id}
                if backend == "mlflow":
                    details = _mlflow_export(
                        root, campaign, record, run_key, projection_id, record_path
                    )
                elif backend == "wandb-offline":
                    details = _wandb_export(
                        root, campaign, record, run_key, projection_id, record_path
                    )
                exported[run_key] = {
                    "experiment_id": record["id"],
                    "projection_id": projection_id,
                    **details,
                }
                _write_json(backend_receipt, exported[run_key])
        receipt = {
            "schema_version": SCHEMA,
            "backend": backend,
            "local_only": True,
            "online_delivery_validated": False,
            "source_authority": "campaign_store",
            "snapshot_hash": snapshot_hash,
            "campaign_id": campaign["id"],
            "run_keys": sorted(exported),
            "projection_path": str(projection_path),
            "backend_runs": exported,
            "reused": False,
        }
        _write_json(receipt_path, receipt)
        return receipt
