"""Run one trusted workload inside an allocation; emit a receipt, never SQLite.

Install this package in the explicit site Python used by the batch script. The
packet, input, output and receipt live on storage shared with the login-node
controller. MPI rank zero alone must publish the workload's result.json.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from .campaign import canonical, digest, now, validate_metrics
from .schedulers import preview_plan
from .worker import _drain_group, _group_exists, _read_artifact, _sync_directory


def run(packet_path: Path) -> int:
    from .scheduler_execution import _write_new

    packet_path = packet_path.absolute()
    if packet_path.resolve() != packet_path:
        raise ValueError("batch packet must not traverse symlinks")
    directory = packet_path.parent
    _, packet = _read_artifact(packet_path)
    expected = {
        "schema_version",
        "experiment_id",
        "token",
        "manifest_digest",
        "config",
        "profile_digest",
        "backend",
        "command",
        "launcher",
        "python",
        "working_directory",
        "timeout_seconds",
        "metric_name",
    }
    if (
        not isinstance(packet, dict)
        or set(packet) != expected
        or packet["schema_version"] != "loopx_hpc_batch_packet_v1"
    ):
        raise ValueError("invalid batch packet")
    backend = packet["backend"]
    job_id = os.environ.get("PBS_JOBID" if backend == "pbs" else "SLURM_JOB_ID", "")
    if not job_id:
        raise ValueError(
            "batch worker requires an actual scheduler allocation identity"
        )
    preview_plan(backend, str(directory / "job.sh"), job_id=job_id)
    if type(packet["timeout_seconds"]) is not int or packet["timeout_seconds"] < 1:
        raise ValueError("invalid batch workload timeout")
    if not isinstance(packet["command"], list) or not isinstance(
        packet["launcher"], list
    ):
        raise ValueError("batch command/launcher must be argv lists")
    # Durable exclusive claim survives a worker kill or scheduler redelivery.
    # Never erase this file or interpret a missing receipt as permission to retry.
    try:
        _write_new(
            directory / "started.json",
            canonical({"token": packet["token"], "job_id": job_id, "started_at": now()})
            + "\n",
        )
    except FileExistsError:
        return 75
    _sync_directory(directory)
    started_at = now()
    replacements = {
        "{config}": str(directory / "config.json"),
        "{result}": str(directory / "result.json"),
        "{python}": packet["python"],
    }
    command = packet["launcher"] + [replacements.get(a, a) for a in packet["command"]]
    metrics, artifacts = {}, []
    failure, process = None, None
    drained = True
    try:
        _, config = _read_artifact(directory / "config.json")
        if canonical(config) != canonical(packet["config"]):
            raise ValueError("frozen input mismatch before execution")
        if (directory / "result.json").exists() or (
            directory / "result.json"
        ).is_symlink():
            raise ValueError("result already exists before workload launch")
        with (
            (directory / "stdout.log").open("xb") as stdout,
            (directory / "stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=packet["working_directory"],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
            deadline = time.monotonic() + packet["timeout_seconds"]
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    failure = "timeout"
                    break
                time.sleep(0.1)
        if _group_exists(process):
            failure = failure or "descendants_after_command_exit"
            drained = _drain_group(process)
        if failure is None and process.returncode != 0:
            failure = f"nonzero_exit:{process.returncode}"
        if failure is None:
            config_bytes, config = _read_artifact(directory / "config.json")
            result_bytes, result = _read_artifact(directory / "result.json")
            if not isinstance(result, dict) or set(result) != {"metrics"}:
                raise ValueError("result schema must be exactly {metrics: {...}}")
            metrics = validate_metrics(result["metrics"], packet["metric_name"])
            if canonical(config) != canonical(packet["config"]):
                raise ValueError("workload modified its frozen input")
            artifacts = [
                {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in (
                    ("config.json", config_bytes),
                    ("result.json", result_bytes),
                )
            ]
    except (OSError, ValueError, TypeError, RecursionError) as error:
        failure = "invalid_execution:" + type(error).__name__
    finally:
        if process is not None and _group_exists(process):
            drained = _drain_group(process)
            failure = failure or "descendants_after_command_exit"
    if not drained:
        failure = "process_group_unresolved:" + (failure or "unknown")
    receipt = {
        "schema_version": "loopx_hpc_batch_receipt_v1",
        "experiment_id": packet["experiment_id"],
        "token": packet["token"],
        "packet_digest": digest(packet),
        "backend": backend,
        "job_id": job_id,
        "started_at": started_at,
        "finished_at": now(),
        "status": "failed" if failure else "succeeded",
        "failure": failure,
        "metrics": {} if failure else metrics,
        "artifacts": [] if failure else artifacts,
    }
    # Publish atomically only after content and file metadata reach shared storage.
    _write_new(directory / "receipt.pending.json", canonical(receipt) + "\n")
    os.link(directory / "receipt.pending.json", directory / "receipt.json")
    _sync_directory(directory)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
