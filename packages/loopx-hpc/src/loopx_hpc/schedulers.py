"""Pure PBS/Slurm rendering and conservative single-job observation parsing.

This module never runs SSH, submits/cancels jobs, or reads scheduler state. Its
preview argv are inert data. Site admission, allocation budgets, job identity
binding, checkpoint validation and scientific success are separate boundaries.

Syntax sources (verified 2026-09-04):
https://slurm.schedmd.com/sbatch.html
https://slurm.schedmd.com/sacct.html
https://slurm.schedmd.com/job_state_codes.html
https://docs.alcf.anl.gov/running-jobs/
https://docs.alcf.anl.gov/aurora/running-jobs-aurora/
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import PurePosixPath

from .environments import render_environment_prologue, validate_environment

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_WALLTIME = re.compile(r"([0-9]{1,4}):([0-5][0-9]):([0-5][0-9])\Z")
_SLURM_JOB = re.compile(r"[0-9]+(?:_[0-9]+)?\Z")
_PBS_JOB = re.compile(r"[0-9]+(?:\[[0-9]+\])?(?:\.[A-Za-z0-9][A-Za-z0-9_.-]*)?\Z")
_COMMON = {"job_name", "nodes", "walltime", "account", "queue"}
_SPECIFIC = {
    "slurm": {"ntasks_per_node", "cpus_per_task", "gpus_per_node"},
    "pbs": {"system", "filesystems", "place"},
}


def _scheduler(value: str) -> str:
    if not isinstance(value, str) or value not in _SPECIFIC:
        raise ValueError("scheduler must be pbs or slurm")
    return value


def _literal(value: object, field: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(f"{field} must be a compact literal identifier")
    return value


def _positive(value: object, field: str) -> int:
    if type(value) is not int or value <= 0 or value > 1_000_000:
        raise ValueError(f"{field} must be a positive integer no greater than 1000000")
    return value


def validate_resources(scheduler: str, resources: dict) -> dict:
    """Validate syntax only, never site capacity or permission.

    Required: job_name, nodes, walltime (H+:MM:SS). Optional common account and
    queue. Slurm adds ntasks_per_node/cpus_per_task/gpus_per_node; PBS adds system,
    filesystems (literal list), place (free, pack, scatter or vscatter). No raw
    directives, arbitrary select expressions, or inferred site settings accepted.
    """
    scheduler = _scheduler(scheduler)
    if not isinstance(resources, dict):
        raise ValueError("resources must be an object")
    if any(not isinstance(key, str) for key in resources):
        raise ValueError("resource field names must be strings")
    unknown = set(resources) - _COMMON - _SPECIFIC[scheduler]
    if unknown:
        raise ValueError(f"unsupported {scheduler} resource fields: {sorted(unknown)}")
    result = {"job_name": _literal(resources.get("job_name"), "job_name")}
    result["nodes"] = _positive(resources.get("nodes"), "nodes")
    walltime = resources.get("walltime")
    match = _WALLTIME.fullmatch(walltime) if isinstance(walltime, str) else None
    if not match or not any(int(part) for part in match.groups()):
        raise ValueError("walltime must be a nonzero H+:MM:SS duration")
    result["walltime"] = walltime
    for field in ("account", "queue", "system"):
        if field in resources:
            result[field] = _literal(resources[field], field)
    for field in ("ntasks_per_node", "cpus_per_task", "gpus_per_node"):
        if field in resources:
            result[field] = _positive(resources[field], field)
    if "filesystems" in resources:
        filesystems = resources["filesystems"]
        if (
            not isinstance(filesystems, list)
            or not filesystems
            or len(filesystems) > 32
        ):
            raise ValueError(
                "filesystems must be a nonempty list with at most 32 entries"
            )
        result["filesystems"] = [_literal(value, "filesystem") for value in filesystems]
        if len(set(result["filesystems"])) != len(result["filesystems"]):
            raise ValueError("filesystems must not contain duplicates")
    if "place" in resources:
        if resources["place"] not in ("free", "pack", "scatter", "vscatter"):
            raise ValueError("place must be free, pack, scatter or vscatter")
        result["place"] = resources["place"]
    return result


def render_script(
    scheduler: str, command: list[str], resources: dict, environment: dict | None = None
) -> str:
    """Render a preview-only Bash script. The command is argv, never shell text.

    Scheduler directives precede all executable lines. No MPI launcher, process
    count, working directory, module version, output path, or site policy is
    inferred. Pass a launcher explicitly in command when it is independently
    validated. A rendered script is not permission to execute or submit it.
    """
    checked = validate_resources(scheduler, resources)
    checked_environment = (
        validate_environment(environment) if environment is not None else None
    )
    if not isinstance(command, list) or not command or len(command) > 4096:
        raise ValueError("command must be a nonempty argv list")
    for arg in command:
        if not isinstance(arg, str) or any(
            ord(char) < 32 or ord(char) == 127 for char in arg
        ):
            raise ValueError(
                "command arguments must be strings without control characters"
            )
    if not command[0] or command[0].startswith("-"):
        raise ValueError("command executable must be nonempty and not an option")
    lines = [
        "#!/bin/bash -l"
        if checked_environment and checked_environment["login_shell"]
        else "#!/bin/bash",
        "# Preview only: no submission or site validation has occurred.",
    ]
    if scheduler == "slurm":
        options = {
            "job_name": "job-name",
            "nodes": "nodes",
            "walltime": "time",
            "account": "account",
            "queue": "partition",
            "ntasks_per_node": "ntasks-per-node",
            "cpus_per_task": "cpus-per-task",
            "gpus_per_node": "gpus-per-node",
        }
        lines.extend(
            f"#SBATCH --{option}={checked[field]}"
            for field, option in options.items()
            if field in checked
        )
    else:
        lines.extend(
            [
                f"#PBS -N {checked['job_name']}",
                f"#PBS -l select={checked['nodes']}"
                + (f":system={checked['system']}" if "system" in checked else ""),
                f"#PBS -l walltime={checked['walltime']}",
            ]
        )
        for field, option in (("account", "-A"), ("queue", "-q")):
            if field in checked:
                lines.append(f"#PBS {option} {checked[field]}")
        if "filesystems" in checked:
            lines.append(f"#PBS -l filesystems={':'.join(checked['filesystems'])}")
        if "place" in checked:
            lines.append(f"#PBS -l place={checked['place']}")
    lines.extend(["", "set -euo pipefail", "umask 077"])
    if checked_environment is not None:
        lines.extend(render_environment_prologue(checked_environment))
    lines.append("exec -- " + shlex.join(command))
    return "\n".join(lines) + "\n"


def preview_plan(scheduler: str, script_path: str, job_id: str | None = None) -> dict:
    """Return inert submit/status/cancel argv. There is no execute mode.

    A caller must independently bind one expected scheduler job ID to its attempt
    before accepting observations. Slurm --allocations excludes step rows. PBS
    history retention and scheduler accounting availability remain site-dependent.
    """
    scheduler = _scheduler(scheduler)
    if (
        not isinstance(script_path, str)
        or not script_path
        or any(ord(c) < 32 or ord(c) == 127 for c in script_path)
    ):
        raise ValueError(
            "script_path must be a nonempty path without control characters"
        )
    if (
        not PurePosixPath(script_path).is_absolute()
        or ".." in PurePosixPath(script_path).parts
    ):
        raise ValueError("script_path must be an absolute POSIX path")
    submit = (
        ["sbatch", "--parsable", script_path]
        if scheduler == "slurm"
        else ["qsub", script_path]
    )
    status = cancel = None
    if job_id is not None:
        pattern = _SLURM_JOB if scheduler == "slurm" else _PBS_JOB
        if not isinstance(job_id, str) or not pattern.fullmatch(job_id):
            raise ValueError(
                "job_id must identify exactly one job, not a range or step"
            )
        if scheduler == "slurm":
            status = [
                "sacct",
                "--allocations",
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State%32,ExitCode",
                "--jobs",
                job_id,
            ]
            cancel = ["scancel", job_id]
        else:
            status = ["qstat", "-x", "-f", "-F", "json", job_id]
            cancel = ["qdel", job_id]
    return {
        "schema_version": "loopx_hpc_scheduler_preview_v1",
        "scheduler": scheduler,
        "execution_enabled": False,
        "site_validated": False,
        "submit_argv": submit,
        "status_argv": status,
        "cancel_argv": cancel,
    }


def _observation(
    scheduler: str,
    *,
    status: str = "unknown",
    job_id: str | None = None,
    native_state: str | None = None,
    exit_code: int | None = None,
    signal: int | None = None,
    reason: str = "malformed_or_ambiguous_output",
) -> dict:
    return {
        "scheduler": scheduler,
        "status": status,
        "job_id": job_id,
        "native_state": native_state,
        "exit_code": exit_code,
        "signal": signal,
        "terminal": status in {"succeeded", "failed", "cancelled"},
        "reason": reason,
        "scientific_result_validated": False,
    }


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _pbs_status(output: str) -> dict:
    try:
        value = json.loads(output, object_pairs_hook=_unique_json_object)
    except (ValueError, TypeError, RecursionError):
        return _observation("pbs")
    jobs = value.get("Jobs") if isinstance(value, dict) else None
    if not isinstance(jobs, dict) or len(jobs) != 1:
        return _observation("pbs", reason="missing_or_ambiguous_job")
    job_id, job = next(iter(jobs.items()))
    if not _PBS_JOB.fullmatch(job_id) or not isinstance(job, dict):
        return _observation("pbs")
    state = job.get("job_state")
    if not isinstance(state, str):
        return _observation("pbs", job_id=job_id)
    status = {
        "Q": "queued",
        "H": "queued",
        "W": "queued",
        "T": "queued",
        "R": "running",
        "E": "running",
        "B": "running",
        "S": "running",
        "U": "running",
    }.get(state, "unknown")
    exit_code = job.get("Exit_status")
    if exit_code is not None and type(exit_code) is not int:
        return _observation(
            "pbs", job_id=job_id, native_state=state, reason="malformed_exit_status"
        )
    reason = (
        "scheduler_observation"
        if status != "unknown"
        else "unrecognized_or_unresolved_native_state"
    )
    if state == "F":
        # PBS signal exits do not distinguish qdel, time limit and application
        # signals. Never label one cancelled based on exit code/comment prose.
        status = (
            "unknown"
            if exit_code is None
            else ("succeeded" if exit_code == 0 else "failed")
        )
        reason = (
            "missing_terminal_exit_status"
            if exit_code is None
            else "scheduler_terminal_exit"
        )
    return _observation(
        "pbs",
        status=status,
        job_id=job_id,
        native_state=state,
        exit_code=exit_code,
        reason=reason,
    )


def _slurm_status(output: str) -> dict:
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        return _observation("slurm", reason="missing_or_ambiguous_job")
    fields = rows[0].split("|")
    if len(fields) != 3 or not _SLURM_JOB.fullmatch(fields[0]):
        return _observation("slurm")
    job_id, state, raw_exit = (field.strip() for field in fields)
    exit_match = re.fullmatch(r"([0-9]+):([0-9]+)", raw_exit)
    if not exit_match:
        return _observation(
            "slurm", job_id=job_id, native_state=state, reason="malformed_exit_status"
        )
    exit_code, signal = map(int, exit_match.groups())
    # sacct may append the cancelling user's numeric ID. Do not accept truncated
    # states (trailing '+') or arbitrary suffixes as proof of terminal status.
    canonical = (
        "CANCELLED" if re.fullmatch(r"CANCELLED(?: by [0-9]+)?", state) else state
    )
    states = {
        "PENDING": "queued",
        "REQUEUED": "queued",
        "REQUEUE_FED": "queued",
        "REQUEUE_HOLD": "queued",
        "SPECIAL_EXIT": "queued",
        "RESV_DEL_HOLD": "queued",
        "RUNNING": "running",
        "CONFIGURING": "running",
        "COMPLETING": "running",
        "SUSPENDED": "running",
        "STOPPED": "running",
        "STAGE_OUT": "running",
        "RESIZING": "running",
        "SIGNALING": "running",
        "POWER_UP_NODE": "running",
        "CANCELLED": "cancelled",
        "BOOT_FAIL": "failed",
        "DEADLINE": "failed",
        "FAILED": "failed",
        "NODE_FAIL": "failed",
        "OUT_OF_MEMORY": "failed",
        "TIMEOUT": "failed",
    }
    status = states.get(canonical, "unknown")
    reason = (
        "scheduler_observation"
        if status != "unknown"
        else "unrecognized_or_unresolved_native_state"
    )
    if canonical == "COMPLETED":
        status = "succeeded" if exit_code == 0 and signal == 0 else "failed"
        reason = "scheduler_terminal_exit"
    # PREEMPTED may be followed by requeue. Keep unresolved until the current
    # attempt and scheduler requeue policy have been independently reconciled.
    return _observation(
        "slurm",
        status=status,
        job_id=job_id,
        native_state=state,
        exit_code=exit_code,
        signal=signal,
        reason=reason,
    )


def parse_status(scheduler: str, output: str) -> dict:
    """Parse one Slurm sacct pipe row or one PBS qstat JSON job, conservatively.

    Output status is queued/running/succeeded/failed/cancelled/unknown. ``running``
    includes allocated cleanup/suspension phases: it is not a claim of progress.
    Success is scheduler completion only, never artifact/scientific validation.
    Missing/malformed/ambiguous output stays unknown. Caller binds returned job_id
    to the expected exact job and attempt; this parser cannot establish identity.
    """
    scheduler = _scheduler(scheduler)
    if not isinstance(output, str) or len(output) > 1_000_000:
        return _observation(scheduler)
    return _pbs_status(output) if scheduler == "pbs" else _slurm_status(output)
