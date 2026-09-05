"""A short-lived controller may exit while this bounded worker completes.

Trusted local workloads stay in one POSIX process group. Group cleanup is not OS
containment: a process that deliberately creates a new session can escape it, and
a killed worker cannot supervise its children. Such workloads need a real runner
containment boundary rather than this local acceptance backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import time

from .campaign import ACTIVE, CampaignStore, canonical, now, validate_metrics

_MAX_ARTIFACT_BYTES = 1024 * 1024
_TERM_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 1.5


def _group_exists(process: subprocess.Popen) -> bool:
    """Reap the owned leader, then conservatively observe its original group.

    A zombie-only group may remain visible on hosts that do not promptly reap
    orphans. We leave it unresolved instead of claiming capacity has been freed.
    """
    process.poll()
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_group_empty(process: subprocess.Popen, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _group_exists(process):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.025)
    return True


def _drain_group(process: subprocess.Popen) -> bool:
    """Best-effort bounded TERM/KILL and readback, never leader-only success."""
    for signal_number, grace in (
        (signal.SIGTERM, _TERM_GRACE_SECONDS),
        (signal.SIGKILL, _KILL_GRACE_SECONDS),
    ):
        if not _group_exists(process):
            return True
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if _wait_group_empty(process, grace):
            return True
    return False


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field in authoritative artifact")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant in authoritative artifact: {value}")


def _read_artifact(path: Path) -> tuple[bytes, object]:
    """Validate, flush and return one bounded byte snapshot for parse AND hash."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError("authoritative artifact must be a bounded regular file")
        data = handle.read(_MAX_ARTIFACT_BYTES + 1)
        after = os.fstat(handle.fileno())
        if len(data) > _MAX_ARTIFACT_BYTES:
            raise ValueError("authoritative artifact exceeds the size limit")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("authoritative artifact changed during read")
        current = path.lstat()
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            raise ValueError("authoritative artifact path was replaced during read")
        value = json.loads(
            data, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        os.fsync(handle.fileno())
        return data, value


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(root: Path, experiment_id: str, token: str) -> None:
    store = CampaignStore(root)
    with store.transaction() as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if row is None or row["token"] != token or row["status"] != "submitting":
            return  # Duplicate worker deliveries do not launch duplicate commands.
        spec = store._manifest(connection)
        record = store._record(row)
        connection.execute(
            "UPDATE experiments SET status='running',process_id=?,updated_at=? WHERE id=?",
            (os.getpid(), now(), experiment_id),
        )
        store.event(
            connection,
            "worker_started",
            experiment_id,
            {
                "token": token,
                "python": platform.python_version(),
                "platform": platform.system(),
            },
        )
    run_dir = store.root / "runs" / experiment_id / token
    config_path, result_path = run_dir / "config.json", run_dir / "result.json"
    replacements = {
        "{config}": str(config_path),
        "{result}": str(result_path),
        "{python}": sys.executable,
    }
    command = [replacements.get(argument, argument) for argument in spec["command"]]
    failure = None
    metrics: dict = {}
    artifacts: list[dict] = []
    process = None
    group_drained = True
    cleanup_attempted = False
    try:
        with config_path.open("xb") as config_file:
            config_file.write((canonical(record["config"]) + "\n").encode())
            config_file.flush()
            os.fsync(config_file.fileno())
        _sync_directory(run_dir)
        with (
            (run_dir / "stdout.log").open("wb") as stdout,
            (run_dir / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            deadline = time.monotonic() + spec["limits"]["timeout_seconds"]
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    failure = "timeout"
                    break
                with store.transaction() as connection:
                    connection.execute(
                        "UPDATE experiments SET updated_at=? WHERE id=? AND token=?",
                        (now(), experiment_id, token),
                    )
                time.sleep(0.1)
        if _group_exists(process):
            if failure is None:
                failure = "descendants_after_command_exit"
            cleanup_attempted = True
            group_drained = _drain_group(process)
        if failure is None and process.returncode != 0:
            failure = f"nonzero_exit:{process.returncode}"
        if failure is None:
            config_bytes, frozen_config = _read_artifact(config_path)
            result_bytes, result = _read_artifact(result_path)
            if not isinstance(result, dict) or set(result) != {"metrics"}:
                raise ValueError("result schema must be exactly {metrics: {...}}")
            metrics = validate_metrics(result["metrics"], spec["metric"]["name"])
            if canonical(frozen_config) != canonical(record["config"]):
                raise ValueError("experiment modified its frozen input configuration")
            for path, data in (
                (config_path, config_bytes),
                (result_path, result_bytes),
            ):
                artifacts.append(
                    {
                        "path": str(path.relative_to(store.root)),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
            _sync_directory(run_dir)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        failure = f"invalid_execution:{type(error).__name__}"
    finally:
        # Unexpected Python errors also receive bounded cleanup before propagating.
        # They cannot silently leave a command running after a failed ingest.
        if process is not None and not cleanup_attempted and _group_exists(process):
            group_drained = _drain_group(process)
            if failure is None:
                failure = "descendants_after_command_exit"
    if not group_drained:
        failure = "process_group_unresolved:" + (failure or "unknown")
    with store.transaction() as connection:
        row = connection.execute(
            "SELECT token,status FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if row["token"] != token or row["status"] not in ACTIVE:
            return
        status = (
            "unknown" if not group_drained else ("failed" if failure else "succeeded")
        )
        connection.execute(
            "UPDATE experiments SET status=?,metrics=?,artifacts=?,failure=?,updated_at=? WHERE id=? AND token=?",
            (
                status,
                canonical(metrics if not failure else {}),
                canonical(artifacts if not failure else []),
                failure,
                now(),
                experiment_id,
                token,
            ),
        )
        store.event(
            connection,
            "attempt_finished",
            experiment_id,
            {
                "token": token,
                "status": status,
                "failure": failure,
                "metrics": metrics if not failure else {},
                "artifacts": artifacts if not failure else [],
            },
        )


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
