"""Detached local attempts with durable admission before process creation."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import subprocess
import sys
import uuid

from .campaign import ACTIVE, CampaignStore, now


class LocalExecutor:
    def __init__(self, store: CampaignStore):
        self.store = store

    def submit(self, experiment_id: str, *, execute: bool = False) -> dict:
        record = self.store.get_experiment(experiment_id)
        if not execute:
            return {
                **record,
                "preview": True,
                "backend": "local",
                "will_launch": record["status"] == "planned",
            }
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if row["status"] != "planned":
                return self.store._record(row)
            occupied = connection.execute(
                "SELECT COUNT(*) FROM experiments WHERE status IN ('submitting','running','unknown')"
            ).fetchone()[0]
            if occupied >= self.store._manifest(connection)["limits"]["max_concurrent"]:
                raise ValueError(
                    "local concurrency budget exhausted; unresolved attempts reserve capacity"
                )
            token = uuid.uuid4().hex
            connection.execute(
                "UPDATE experiments SET status='submitting',token=?,updated_at=? WHERE id=?",
                (token, now(), experiment_id),
            )
            self.store.event(
                connection,
                "attempt_reserved",
                experiment_id,
                {"token": token, "backend": "local"},
            )
        # Never automatically retry an attempt once intent is durable. A crash
        # here is ambiguous, but cannot turn a repeat invocation into a duplicate.
        run_dir = self.store.root / "runs" / experiment_id / token
        run_dir.mkdir(parents=True, mode=0o700)
        try:
            with (run_dir / "worker.log").open("ab") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "loopx_hpc.worker",
                        str(self.store.root),
                        experiment_id,
                        token,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as error:
            with self.store.transaction() as connection:
                connection.execute(
                    "UPDATE experiments SET status='failed',failure=?,updated_at=? WHERE id=? AND token=? AND status='submitting'",
                    (
                        f"worker_spawn_failed:{type(error).__name__}",
                        now(),
                        experiment_id,
                        token,
                    ),
                )
                self.store.event(
                    connection,
                    "worker_spawn_failed",
                    experiment_id,
                    {"error_type": type(error).__name__},
                )
            raise
        with self.store.transaction() as connection:
            connection.execute(
                "UPDATE experiments SET process_id=? WHERE id=? AND token=?",
                (process.pid, experiment_id, token),
            )
        return self.store.get_experiment(experiment_id)

    def reconcile(self, experiment_id: str) -> dict:
        from .scheduler_execution import scheduler_records

        with self.store.transaction() as connection:
            if any(
                a["experiment_id"] == experiment_id
                for a in scheduler_records(connection)
            ):
                raise ValueError(
                    "use SchedulerExecutor to reconcile this scheduler attempt"
                )
        record = self.store.get_experiment(experiment_id)
        if record["status"] not in ACTIVE or record["status"] == "unknown":
            return record
        # This is an observation, not a retry. PID liveness alone is not proof of
        # identity or successful completion; only the worker's exact receipt is.
        stale = (
            datetime.now(timezone.utc) - datetime.fromisoformat(record["updated_at"])
        ).total_seconds() > 15
        absent = False
        if record["process_id"]:
            try:
                os.kill(record["process_id"], 0)
            except ProcessLookupError:
                absent = True
            except PermissionError:
                pass
        if stale and (
            absent or record["status"] == "submitting" or self._beyond_deadline(record)
        ):
            with self.store.transaction() as connection:
                changed = connection.execute(
                    "UPDATE experiments SET status='unknown',failure='worker_receipt_missing',updated_at=? WHERE id=? AND status=? AND updated_at=?",
                    (now(), experiment_id, record["status"], record["updated_at"]),
                ).rowcount
                if changed:
                    self.store.event(
                        connection,
                        "attempt_unresolved",
                        experiment_id,
                        {"reason": "worker_receipt_missing", "retry_allowed": False},
                    )
        return self.store.get_experiment(experiment_id)

    def _beyond_deadline(self, record: dict) -> bool:
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(record["updated_at"])
        ).total_seconds()
        return age > self.store.manifest()["limits"]["timeout_seconds"] + 30
