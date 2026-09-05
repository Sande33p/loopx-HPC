"""Optional host-side bridge to upstream LoopX; no duplicate lifecycle authority.

The host supplies a real quota admission packet and typed durable writeback and
spend callbacks. This module never synthesizes them or completes a Todo from an
experiment's process exit code. The independent host must interpret scientific
failure separately from successful recording of a terminal experiment outcome.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .provider import CAPABILITY_ID


def _environment(
    root: Path,
    scheduler_profile: Path | None = None,
    *,
    scheduler_runtime: bool = False,
) -> dict[str, str]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("campaign root must be a directory")
    # This v0 local provider needs interpreter lookup/locale/temp settings, not
    # ambient model/tracker/cloud credentials. Future remote providers need an
    # explicit credential-injection contract rather than widening this list.
    allowed = {
        "PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMROOT",
        "COMSPEC",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment["LOOPX_HPC_ROOT"] = str(resolved)
    if scheduler_profile is not None:
        profile = scheduler_profile.resolve(strict=True)
        if not profile.is_file():
            raise ValueError("scheduler profile must name a file")
        environment["LOOPX_HPC_SCHEDULER_PROFILE"] = str(profile)
    if scheduler_profile is not None or scheduler_runtime:
        # Native site CLI uses login identity and its selected scheduler config;
        # no model/cloud credentials or allocation environment is inherited.
        for key in ("HOME", "USER", "LOGNAME", "PBS_CONF_FILE", "SLURM_CONF"):
            if key in os.environ:
                environment[key] = os.environ[key]
    return environment


def observe_experiment(
    *,
    state_file: Path,
    registry_path: Path,
    goal_id: str,
    root: Path,
    experiment_id: str,
    context_refs: Sequence[Mapping[str, Any]],
    execute: bool = False,
) -> dict[str, Any]:
    from loopx.extensions.capability_admission import invoke_external_capability

    return invoke_external_capability(
        state_file=state_file,
        registry_path=registry_path,
        goal_id=goal_id,
        capability_id=CAPABILITY_ID,
        operation="observe",
        provider_input={
            "context_refs": list(context_refs),
            "input": {"experiment_id": experiment_id},
        },
        execute=execute,
        environment=_environment(root),
    )


def start_local_experiment(
    *,
    state_file: Path,
    run_dir: Path,
    registry_path: Path,
    goal_id: str,
    agent_id: str,
    todo_id: str,
    turn_instance_id: str,
    root: Path,
    experiment_id: str,
    context_refs: Sequence[Mapping[str, Any]],
    admission: Mapping[str, Any],
    execute: bool = False,
) -> dict[str, Any]:
    from loopx.extensions.governed_capability_execution import (
        start_governed_external_capability,
    )

    selected = admission.get("selected_todo")
    if (
        not isinstance(selected, Mapping)
        or selected.get("target_key") != f"experiment:{experiment_id}"
    ):
        raise ValueError("selected Todo must target this exact experiment")
    return start_governed_external_capability(
        state_file=state_file,
        run_dir=run_dir,
        registry_path=registry_path,
        goal_id=goal_id,
        agent_id=agent_id,
        todo_id=todo_id,
        turn_instance_id=turn_instance_id,
        capability_id=CAPABILITY_ID,
        operation="run_local",
        provider_input={
            "context_refs": list(context_refs),
            "input": {"experiment_id": experiment_id},
        },
        admission=admission,
        execute=execute,
        environment=_environment(root),
    )


def start_scheduler_experiment(
    *,
    state_file: Path,
    run_dir: Path,
    registry_path: Path,
    goal_id: str,
    agent_id: str,
    todo_id: str,
    turn_instance_id: str,
    root: Path,
    experiment_id: str,
    scheduler_profile: Path,
    expected_manifest_digest: str,
    context_refs: Sequence[Mapping[str, Any]],
    admission: Mapping[str, Any],
    execute: bool = False,
) -> dict[str, Any]:
    """Reuse upstream governed execution with a frozen, operator-owned site profile."""
    from loopx.extensions.governed_capability_execution import (
        start_governed_external_capability,
    )
    from .campaign import CampaignStore, digest
    from .scheduler_execution import validate_profile
    from .worker import _read_artifact

    selected = admission.get("selected_todo")
    if (
        not isinstance(selected, Mapping)
        or selected.get("target_key") != f"experiment:{experiment_id}"
    ):
        raise ValueError("selected Todo must target this exact experiment")
    if (
        not isinstance(expected_manifest_digest, str)
        or not re.fullmatch(r"[a-f0-9]{64}", expected_manifest_digest)
        or digest(CampaignStore(root).manifest()) != expected_manifest_digest
    ):
        raise ValueError("scheduler source manifest does not match expected study")
    _, profile = _read_artifact(scheduler_profile)
    return start_governed_external_capability(
        state_file=state_file,
        run_dir=run_dir,
        registry_path=registry_path,
        goal_id=goal_id,
        agent_id=agent_id,
        todo_id=todo_id,
        turn_instance_id=turn_instance_id,
        capability_id=CAPABILITY_ID,
        operation="run_scheduler",
        provider_input={
            "context_refs": list(context_refs),
            "input": {
                "experiment_id": experiment_id,
                "profile_digest": digest(validate_profile(profile)),
                "manifest_digest": expected_manifest_digest,
            },
        },
        admission=admission,
        execute=execute,
        environment=_environment(root, scheduler_profile),
    )


def reconcile_scheduler_experiment(
    *,
    run_dir: Path,
    invocation_id: str,
    root: Path,
    scheduler_profile: Path | None = None,
    writeback: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    spend: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    from loopx.extensions.governed_capability_execution import (
        reconcile_governed_external_capability,
    )

    return reconcile_governed_external_capability(
        run_dir=run_dir,
        invocation_id=invocation_id,
        writeback=writeback,
        spend=spend,
        environment=_environment(root, scheduler_runtime=True),
    )


def reconcile_local_experiment(
    *,
    run_dir: Path,
    invocation_id: str,
    root: Path,
    writeback: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    spend: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    from loopx.extensions.governed_capability_execution import (
        reconcile_governed_external_capability,
    )

    return reconcile_governed_external_capability(
        run_dir=run_dir,
        invocation_id=invocation_id,
        writeback=writeback,
        spend=spend,
        environment=_environment(root),
    )
