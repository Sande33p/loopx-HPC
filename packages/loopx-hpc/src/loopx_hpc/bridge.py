"""Optional host-side bridge to upstream LoopX; no duplicate lifecycle authority.

The host supplies a real quota admission packet and typed durable writeback and
spend callbacks. This module never synthesizes them or completes a Todo from an
experiment's process exit code. The independent host must interpret scientific
failure separately from successful recording of a terminal experiment outcome.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .provider import CAPABILITY_ID


def _environment(root: Path) -> dict[str, str]:
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
    return {**environment, "LOOPX_HPC_ROOT": str(resolved)}


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
