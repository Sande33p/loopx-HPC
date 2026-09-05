"""Optional MLflow attempt lifecycle mirror, never campaign execution authority.

Unlike immutable tracking.export_tracking revisions, this explicitly selected
mirror keeps one run per frozen study, experiment, and actual attempt. A local
destination serializes writers. Remote creation has no MLflow idempotency key:
an ambiguous create is searched on replay and otherwise held for inspection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

from .campaign import CampaignStore, canonical, digest
from .tracking import _check_local_artifact_uri, _private_sdk_environment, _write_json


SCHEMA = "loopx_hpc_mlflow_lifecycle_v1"


def _milliseconds(value: str) -> int:
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("timestamp must be UTC")
        result = int(timestamp.timestamp() * 1000)
        if result < 0:
            raise ValueError("timestamp predates epoch")
        return result
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("tracking requires valid original UTC timestamps") from error


def _destination(destination: Path, tracking_uri: str | None, allow_external: bool):
    if "://" in str(destination):
        raise ValueError("tracking destination must be a local directory")
    root = Path(destination).expanduser().resolve() / "mlflow-lifecycle"
    local_uri = "sqlite:///" + str(root / "tracking.sqlite")
    uri = tracking_uri or local_uri
    external = uri != local_uri
    if external:
        parsed = urlsplit(uri)
        if not allow_external:
            raise ValueError("configured tracking URI requires allow_external=True")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "external tracking URI must be HTTP(S), without credentials or query"
            )
    return root, uri, external


def _read(store: CampaignStore):
    with store.transaction() as connection:
        snapshot = store._snapshot(connection)
        events = [
            {**dict(row), "payload": json.loads(row["payload"])}
            for row in connection.execute("SELECT * FROM events ORDER BY sequence")
        ]
    return snapshot, events


def _plan(
    store: CampaignStore, snapshot: dict, events: list[dict]
) -> tuple[list[dict], list[str]]:
    campaign = snapshot["campaign"]
    source = {"campaign_id": campaign["id"], "manifest_digest": digest(campaign)}
    plans, pending = [], []
    for record in snapshot["experiments"]:
        if not record.get("token"):
            pending.append(record["id"])
            continue
        if digest(record["config"]) != record["config_hash"]:
            raise ValueError("tracking source configuration hash mismatch")
        if record["status"] == "succeeded" and not store.verify_record(record):
            raise ValueError("tracking requires verified successful source artifacts")
        selected = [
            event
            for event in events
            if event["experiment_id"] == record["id"]
            and (
                event["payload"].get("token") == record["token"]
                or (
                    "token" not in event["payload"]
                    and event["kind"] in {"worker_spawn_failed", "attempt_unresolved"}
                )
            )
        ]
        if not selected:
            raise ValueError("tracking attempt has no attributable execution events")
        lifecycle = []
        for event in selected:
            _milliseconds(event["created_at"])
            payload = event["payload"]
            item = {
                "sequence": event["sequence"],
                "kind": event["kind"],
                "recorded_at": event["created_at"],
                "event_digest": digest(event),
            }
            for key in (
                "backend",
                "job_id",
                "profile_digest",
                "status",
                "started_at",
                "finished_at",
            ):
                if key in payload:
                    if key.endswith("_at"):
                        _milliseconds(payload[key])
                    item[key] = payload[key]
            observation = payload.get("observation")
            if isinstance(observation, dict):
                item["observation"] = {
                    key: observation[key]
                    for key in (
                        "state",
                        "status",
                        "native_state",
                        "exit_code",
                        "signal",
                        "terminal",
                        "job_id",
                    )
                    if key in observation
                }
            lifecycle.append(item)
        last = selected[-1]
        terminal_events = [
            event
            for event in selected
            if event["kind"] in {"attempt_finished", "worker_spawn_failed"}
        ]
        terminal = terminal_events[-1] if terminal_events else None
        if record["status"] in {"succeeded", "failed"}:
            expected_status = (
                "failed"
                if terminal and terminal["kind"] == "worker_spawn_failed"
                else terminal["payload"].get("status")
                if terminal
                else None
            )
            if expected_status != record["status"] or canonical(
                terminal["payload"].get("metrics", {})
            ) != canonical(record["metrics"]):
                raise ValueError(
                    "terminal source status or metrics differ from execution receipt"
                )
        # A live attempt's observation bound comes from its journal, not the
        # experiment row updated just before the reservation event was inserted.
        # This is an observed timestamp, never a fabricated terminal finish.
        finished_at = (
            (terminal["payload"].get("finished_at") or terminal["created_at"])
            if terminal
            else last["created_at"]
        )
        started_at = next(
            (
                event["payload"]["started_at"]
                for event in reversed(selected)
                if event["payload"].get("started_at")
            ),
            None,
        )
        start_ms = _milliseconds(selected[0]["created_at"])
        end_ms = _milliseconds(finished_at)
        if end_ms < start_ms or (started_at and _milliseconds(started_at) > end_ms):
            raise ValueError("tracking source execution timestamps are out of order")
        scheduler = [
            {
                key: row[key]
                for key in (
                    "experiment_id",
                    "token",
                    "backend",
                    "profile_digest",
                    "job_id",
                    "created_at",
                    "updated_at",
                )
                if key in row
            }
            for row in snapshot.get("scheduler_attempts", [])
            if row.get("experiment_id") == record["id"]
            and row.get("token") == record["token"]
        ]
        projection = {
            "source": source,
            "experiment_id": record["id"],
            "attempt_token": record["token"],
            "config": record["config"],
            "config_hash": record["config_hash"],
            "metric": campaign["metric"],
            "provenance": {
                key: campaign.get("provenance", {}).get(key)
                for key in ("code_revision", "dataset_revision", "environment")
            },
            "status": record["status"],
            "metrics": record["metrics"],
            # Raw failure messages, scheduler responses, logs, commands, and
            # process environment values are deliberately not tracker payloads.
            "execution_failed": record["status"] == "failed",
            "lifecycle": lifecycle,
            "scheduler_attempts": scheduler,
            "started_at": started_at,
            "started_at_basis": "worker_receipt" if started_at else "not_recorded",
            "finished_at": finished_at if terminal else None,
            "finished_at_basis": (
                "worker_receipt"
                if terminal and terminal["payload"].get("finished_at")
                else "terminal_journal_observation"
                if terminal
                else "not_recorded"
            ),
            "artifact_digests": [a["sha256"] for a in record["artifacts"]],
        }
        plans.append(
            {
                "run_key": digest(
                    [source["manifest_digest"], record["id"], record["token"]]
                ),
                "record": record,
                "projection": projection,
                "projection_digest": digest(projection),
                "start_ms": start_ms,
                "metric_ms": end_ms,
                "metric_step": (terminal or last)["sequence"],
            }
        )
    return plans, pending


def _artifact_boundary(uri: str, root: Path, external: bool) -> None:
    # External opt-in includes the server-selected artifact repository and SDK
    # redirects; this scheme guard is not an outbound hostname allowlist.
    if not external:
        _check_local_artifact_uri(uri, root / "artifacts")
    elif urlsplit(uri).scheme not in {
        "http",
        "https",
        "mlflow-artifacts",
        "s3",
        "gs",
        "wasbs",
        "abfss",
        "dbfs",
    }:
        raise ValueError(
            "external tracker cannot select an arbitrary local artifact directory"
        )


def _run(
    client, experiment_id: str, plan: dict, state_path: Path, root: Path, external: bool
):
    from mlflow.entities import ViewType

    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`loopx.attempt_key` = '{plan['run_key']}'",
        run_view_type=ViewType.ALL,
        max_results=2,
    )
    if len(runs) > 1 or (runs and runs[0].info.lifecycle_stage != "active"):
        raise ValueError(
            "ambiguous or deleted MLflow attempt identity; inspect tracker"
        )
    marker = state_path.with_suffix(".creating.json")
    if runs:
        run = runs[0]
    else:
        if marker.exists() or state_path.exists():
            raise RuntimeError(
                "MLflow run creation unresolved; inspect tracker before another create"
            )
        _write_json(
            marker,
            {
                "attempt_key": plan["run_key"],
                "event_digests": {
                    str(event["sequence"]): event["event_digest"]
                    for event in plan["projection"]["lifecycle"]
                },
            },
        )
        projection = plan["projection"]
        run = client.create_run(
            experiment_id,
            start_time=plan["start_ms"],
            tags={
                "loopx.attempt_key": plan["run_key"],
                "loopx.manifest_digest": projection["source"]["manifest_digest"],
                "loopx.experiment_id": projection["experiment_id"],
                "loopx.attempt_token": projection["attempt_token"],
                "loopx.config_hash": projection["config_hash"],
                "loopx.source_authority": "campaign_store",
                "loopx.local_only": str(not external).lower(),
                "loopx.start_time_basis": "first_attributable_execution_event",
                "mlflow.runName": projection["experiment_id"]
                + "-"
                + projection["attempt_token"][:12],
            },
        )
    _artifact_boundary(run.info.artifact_uri, root, external)
    projection = plan["projection"]
    expected = {
        "loopx.config_hash": projection["config_hash"],
        "loopx.manifest_digest": projection["source"]["manifest_digest"],
        "loopx.experiment_id": projection["experiment_id"],
        "loopx.attempt_token": projection["attempt_token"],
    }
    if any(run.data.tags.get(key) != value for key, value in expected.items()):
        raise ValueError("MLflow attempt has conflicting immutable identity")
    return run


def _log_artifacts(
    client, run_id: str, store: CampaignStore, plan: dict, root: Path
) -> None:
    # Stage exact verified bytes before the SDK can reopen them. Only scientific
    # input/result artifacts are selected; arbitrary inventory entries/logs are
    # never uploaded by this default adapter.
    if plan["record"]["status"] != "succeeded":
        return
    for artifact in plan["record"]["artifacts"]:
        name = Path(artifact["path"]).name
        if name not in {"config.json", "result.json"}:
            continue
        path = (store.root / artifact["path"]).resolve()
        if not path.is_relative_to(store.root):
            raise ValueError("tracking artifact escapes source campaign")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError("tracking artifact changed during export")
        with tempfile.TemporaryDirectory(prefix=".artifact-", dir=root) as directory:
            staged = Path(directory) / name
            staged.write_bytes(data)
            client.log_artifact(
                run_id, str(staged), artifact_path="evidence/" + artifact["sha256"]
            )


def sync_mlflow(
    store: CampaignStore,
    destination: Path,
    *,
    execute: bool = False,
    tracking_uri: str | None = None,
    allow_external: bool = False,
    include_artifacts: bool = True,
) -> dict:
    """Preview or explicitly mirror source attempt lifecycles into MLflow.

    Default destination is isolated local SQLite and owned local artifacts,
    ignoring inherited MLFLOW_TRACKING_URI. An explicit HTTP(S) URI plus
    allow_external opts into exporting config, provenance, metrics and selected
    artifact bytes to that tracker and its artifact repository. No job is
    submitted/reconciled, no source state is changed, and no scientific verdict
    is inferred. Use one destination per writer; remote multi-writer uniqueness
    is not guaranteed by MLflow's create-run API.
    External opt-in is not a hostname/redirect egress restriction.
    """
    for name, value in (
        ("execute", execute),
        ("allow_external", allow_external),
        ("include_artifacts", include_artifacts),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    root, uri, external = _destination(destination, tracking_uri, allow_external)
    snapshot, events = _read(store)
    plans, pending = _plan(store, snapshot, events)
    receipt = {
        "schema_version": SCHEMA,
        "source_authority": "campaign_store",
        "source": {
            "campaign_id": snapshot["campaign"]["id"],
            "manifest_digest": digest(snapshot["campaign"]),
        },
        "tracking_uri": uri,
        "local_only": not external,
        "preview": not execute,
        "pending_experiment_ids": pending,
        "include_artifacts": include_artifacts,
        "delivery_semantics": "single_destination_replay_ambiguous_create_held",
        "artifact_scope": "verified_config_and_result_only",
        "scientific_verdicts": "not_inferred",
        "runs": [
            {
                "attempt_key": p["run_key"],
                "experiment_id": p["record"]["id"],
                "attempt_token": p["record"]["token"],
                "source_status": p["record"]["status"],
            }
            for p in plans
        ],
    }
    if not execute or not plans:
        return receipt
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".sync.lock").open("a") as lock, _private_sdk_environment():
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as error:
            raise RuntimeError(
                "MLflow lifecycle sync requires optional 'mlflow' dependency"
            ) from error
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment_name = "loopx-hpc-study-" + receipt["source"]["manifest_digest"]
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                experiment_name,
                artifact_location=None if external else (root / "artifacts").as_uri(),
                tags={
                    "loopx.campaign_id": receipt["source"]["campaign_id"],
                    "loopx.manifest_digest": receipt["source"]["manifest_digest"],
                },
            )
        )
        if experiment:
            _artifact_boundary(experiment.artifact_location, root, external)
        for summary, plan in zip(receipt["runs"], plans, strict=True):
            state_path = root / "targets" / digest(uri) / (plan["run_key"] + ".json")
            checkpoint = (
                state_path
                if state_path.exists()
                else state_path.with_suffix(".creating.json")
            )
            previous = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            event_digests = {
                str(event["sequence"]): event["event_digest"]
                for event in plan["projection"]["lifecycle"]
            }
            if any(
                event_digests.get(key) != value
                for key, value in previous.get("event_digests", {}).items()
            ):
                raise ValueError("previously tracked source execution history changed")
            run = _run(client, experiment_id, plan, state_path, root, external)
            run_id = run.info.run_id
            reused = (
                previous.get("projection_digest") == plan["projection_digest"]
                and previous.get("include_artifacts") == include_artifacts
            )
            if not reused:
                # Record source identity before effects, not only after success:
                # a copied/diverged source must not overwrite a partial delivery.
                _write_json(
                    state_path,
                    {
                        "run_id": run_id,
                        "event_digests": event_digests,
                        "projection_digest": None,
                    },
                )
                projection = plan["projection"]
                for key, value in {
                    "config_hash": projection["config_hash"],
                    **{
                        "config." + key: canonical(value)
                        for key, value in projection["config"].items()
                    },
                    **{
                        "provenance." + key: canonical(value)
                        for key, value in projection["provenance"].items()
                    },
                }.items():
                    safe_key = (
                        key
                        if re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", key)
                        else "field." + digest(key)
                    )
                    client.log_param(
                        run_id,
                        safe_key,
                        value if len(value) <= 6000 else "sha256:" + digest(value),
                        synchronous=True,
                    )
                for name, value in projection["metrics"].items():
                    existing = {
                        (item.value, item.timestamp, item.step)
                        for item in client.get_metric_history(run_id, name)
                    }
                    point = (float(value), plan["metric_ms"], plan["metric_step"])
                    if point not in existing:
                        client.log_metric(
                            run_id,
                            name,
                            point[0],
                            timestamp=point[1],
                            step=point[2],
                            synchronous=True,
                        )
                client.log_dict(
                    run_id,
                    projection,
                    "lifecycle/" + plan["projection_digest"] + ".json",
                )
                if include_artifacts:
                    _log_artifacts(client, run_id, store, plan, root)
                tags = {
                    "loopx.source_status": projection["status"],
                    "loopx.projection_digest": plan["projection_digest"],
                    "loopx.provenance_digest": digest(projection["provenance"]),
                    "loopx.execution_failed": str(
                        projection["execution_failed"]
                    ).lower(),
                }
                for key in (
                    "started_at",
                    "finished_at",
                    "started_at_basis",
                    "finished_at_basis",
                ):
                    if projection[key]:
                        tags["loopx." + key] = projection[key]
                if projection["scheduler_attempts"]:
                    tags.update(
                        {
                            "loopx.scheduler." + key: str(value)
                            for key, value in projection["scheduler_attempts"][
                                -1
                            ].items()
                            if key in {"backend", "job_id", "profile_digest"}
                            and value is not None
                        }
                    )
                for key, value in tags.items():
                    client.set_tag(run_id, key, value, synchronous=True)
                status = {"succeeded": "FINISHED", "failed": "FAILED"}.get(
                    projection["status"], "RUNNING"
                )
                if status == "RUNNING":
                    client.update_run(run_id, status=status)
                else:
                    client.set_terminated(
                        run_id, status=status, end_time=plan["metric_ms"]
                    )
                _write_json(
                    state_path,
                    {
                        "run_id": run_id,
                        "projection_digest": plan["projection_digest"],
                        "event_digests": event_digests,
                        "include_artifacts": include_artifacts,
                    },
                )
            summary.update(
                {
                    "run_id": run_id,
                    "reused": reused,
                    "projection_digest": plan["projection_digest"],
                }
            )
    return receipt
