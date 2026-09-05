"""Optional LoopX provider; exact scientific data stays in the campaign registry.

``succeeded`` in the provider protocol means that a terminal execution outcome
was recorded, NOT that the experiment or scientific hypothesis succeeded. Read
``observations[].experiment_status`` before deciding any subsequent task state.
The governed host owns admission and settlement; this local subprocess is not
an authentication or sandbox boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .campaign import CampaignStore
from .local import LocalExecutor

CAPABILITY_ID = "experiment-execution"
REQUEST_SCHEMA = "loopx_hpc_experiment_request_v0"
RESULT_SCHEMA = "loopx_hpc_experiment_result_v0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_INVOCATION = re.compile(r"capability-[a-f0-9]{24}\Z")
_STATUSES = {"planned", "submitting", "running", "succeeded", "failed", "unknown"}
_REQUEST_FIELDS = {
    "schema_version",
    "invocation_id",
    "capability_id",
    "operation",
    "goal",
    "provider",
    "context_refs",
    "input",
    "authority",
    "lifecycle",
}


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _experiment(store: CampaignStore, experiment_id: str) -> dict[str, Any]:
    matches = [
        row for row in store.status()["experiments"] if row["id"] == experiment_id
    ]
    if len(matches) != 1:
        raise ValueError("experiment must exist exactly once")
    return dict(matches[0])


def _observation(experiment: Mapping[str, Any]) -> dict[str, Any]:
    experiment_id = _token(experiment.get("id"), "experiment id")
    status = experiment.get("status")
    if status not in _STATUSES:
        raise ValueError("unsupported experiment status")
    config_hash = experiment.get("config_hash")
    if not isinstance(config_hash, str) or not re.fullmatch(
        r"[a-f0-9]{64}", config_hash
    ):
        raise ValueError("invalid configuration digest")
    # No config values, arbitrary metric names, paths, logs, or exception text
    # cross this public boundary. The opaque ref resolves in the private store.
    return {
        "kind": "experiment_outcome",
        "ref": f"experiment:{experiment_id}",
        "experiment_status": status,
        "config_digest": "sha256:" + config_hash,
        "result_digest": _digest(
            {
                "config_hash": config_hash,
                "status": status,
                "metrics": experiment.get("metrics") or {},
                "artifact_digests": [
                    item.get("sha256") for item in experiment.get("artifacts") or []
                ],
            }
        ),
        "metric_count": len(experiment.get("metrics") or {}),
        "artifact_count": len(experiment.get("artifacts") or []),
        "needs_attention": status in {"failed", "unknown", "submitting"},
    }


def handle_request(request: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Handle a validated upstream request without creating a campaign or goal."""
    if not isinstance(request, Mapping) or set(request) - _REQUEST_FIELDS:
        raise ValueError("unsupported request fields")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("unsupported request schema")
    if request.get("capability_id") != CAPABILITY_ID:
        raise ValueError("unsupported capability")
    invocation = request.get("invocation_id")
    if not isinstance(invocation, str) or not _INVOCATION.fullmatch(invocation):
        raise ValueError("invalid invocation id")
    payload = request.get("input")
    if not isinstance(payload, Mapping) or set(payload) != {"experiment_id"}:
        raise ValueError("input requires only experiment_id")
    experiment_id = _token(payload["experiment_id"], "experiment id")
    operation = request.get("operation")
    if operation not in {"observe", "run_local"}:
        raise ValueError("unsupported operation")
    store = CampaignStore(root)
    experiment = _experiment(store, experiment_id)
    effect_id = None
    if operation == "observe":
        if "authority" in request or "lifecycle" in request:
            raise ValueError("observe cannot carry effect authority")
        # Observation must not reconcile, launch, update, or initialize anything.
    else:
        lifecycle = request.get("lifecycle")
        authority = request.get("authority")
        if not isinstance(lifecycle, Mapping) or not isinstance(authority, Mapping):
            raise ValueError("run_local requires governed lifecycle authority")
        effect_id = _token(lifecycle.get("idempotency_key"), "idempotency key")
        if authority.get("effect_id") != effect_id:
            raise ValueError("lifecycle identity does not match effect authority")
        goal = request.get("goal")
        if not isinstance(goal, Mapping) or authority.get("goal_id") != goal.get(
            "goal_id"
        ):
            raise ValueError("effect authority does not match goal")
        phase = lifecycle.get("phase")
        if phase == "start":
            experiment = LocalExecutor(store).submit(experiment_id, execute=True)
        elif phase == "reconcile":
            experiment = LocalExecutor(store).reconcile(experiment_id)
        else:
            raise ValueError("unsupported lifecycle phase")

    observation = _observation(experiment)
    if experiment["status"] == "succeeded":
        observation["evidence_valid"] = store.verify_record(experiment)
        if operation == "run_local" and not observation["evidence_valid"]:
            raise ValueError("terminal experiment evidence did not validate")
    terminal = experiment["status"] in {"succeeded", "failed"}
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "invocation_id": invocation,
        "status": "succeeded" if operation == "observe" or terminal else "running",
        "observations": [observation],
        "domain_state_mutations": [],
        "domain_transition_receipts": [],
        "transition_proposals": [],
        "effect_receipt": None,
        "follow_up": {}
        if operation == "observe" or terminal
        else {
            "kind": "reconcile",
            "ref": f"experiment:{experiment_id}",
        },
    }
    if operation == "run_local" and terminal:
        result["effect_receipt"] = {
            "schema_version": "loopx_external_effect_receipt_v0",
            "invocation_id": invocation,
            "idempotency_key": effect_id,
            "status": "committed",
            "external_ref": f"experiment:{experiment_id}",
            "evidence_digest": _digest(observation),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args(argv)
    if args.doctor:
        # Installation health is independent of a particular campaign or grant.
        return 0
    try:
        configured_root = os.environ.get("LOOPX_HPC_ROOT")
        if not configured_root:
            raise ValueError("LOOPX_HPC_ROOT is required")
        root = Path(configured_root)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("LOOPX_HPC_ROOT must name an existing absolute directory")
        raw = sys.stdin.buffer.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("request exceeds size limit")
        result = handle_request(json.loads(raw), root=root)
    except (ValueError, TypeError, KeyError, OSError):
        # Provider error text can include private paths or commands; never echo it.
        json.dump(
            {"ok": False, "error": "invalid_request_or_local_execution_failure"},
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 2
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
