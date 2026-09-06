"""Opt-in native PBS/Slurm execution from a shared-filesystem controller.

The controller runs on a site login node, not on compute ranks. Compute workers
write bounded receipts, never the SQLite database. A scheduler exit is not a
scientific result; successful ingestion requires both exact terminal accounting
and the matching worker receipt. No SSH, automatic retries, or hidden polling.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import uuid

from .campaign import ACTIVE, CampaignStore, canonical, digest, now, validate_metrics
from .environments import validate_environment
from .schedulers import parse_status, preview_plan, render_script, validate_resources
from .worker import _read_artifact, _sync_directory

_FIELDS = {"scheduler", "resources", "environment", "python", "launcher"}


def validate_profile(profile: dict) -> dict:
    if not isinstance(profile, dict) or set(profile) != _FIELDS:
        raise ValueError(f"scheduler profile fields must be exactly {sorted(_FIELDS)}")
    profile = json.loads(canonical(profile))
    backend = profile["scheduler"]
    profile["resources"] = validate_resources(backend, profile["resources"])
    profile["environment"] = validate_environment(profile["environment"])
    if any(
        key.startswith(("PBS_", "SLURM_", "SBATCH_", "PALS_", "PMI_", "PMIX_"))
        for key in profile["environment"].get("variables", {})
    ):
        raise ValueError(
            "scheduler allocation identity variables must be runtime-owned"
        )
    python = profile["python"]
    if (
        not isinstance(python, str)
        or not python.startswith("/")
        or str(Path(python)) != python
        or ".." in Path(python).parts
        or any(ord(c) < 32 or ord(c) == 127 for c in python)
    ):
        raise ValueError("python must be an explicit normalized absolute site runtime")
    launcher = profile["launcher"]
    if (
        not isinstance(launcher, list)
        or len(launcher) > 128
        or any(
            not isinstance(a, str)
            or not a
            or len(a) > 4096
            or any(ord(c) < 32 or ord(c) == 127 for c in a)
            for a in launcher
        )
    ):
        raise ValueError("launcher must be a bounded literal argv prefix")
    return profile


def scheduler_records(connection: sqlite3.Connection) -> list[dict]:
    if not connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scheduler_attempts'"
    ).fetchone():
        return []
    records = []
    for row in connection.execute(
        "SELECT experiment_id,token,backend,profile_digest,job_id,observation,created_at,updated_at "
        "FROM scheduler_attempts ORDER BY rowid"
    ):
        record = dict(row)
        record["observation"] = json.loads(record["observation"])
        records.append(record)
    return records


def _schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS scheduler_attempts (
        experiment_id TEXT PRIMARY KEY REFERENCES experiments(id), token TEXT UNIQUE NOT NULL,
        backend TEXT NOT NULL, profile TEXT NOT NULL, profile_digest TEXT NOT NULL,
        packet_digest TEXT NOT NULL, job_id TEXT, observation TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""")


def _write_new(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _run_cli(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    # Shell-free, bounded control commands. Do not capture unlimited scheduler
    # output in memory or pass ambient SBATCH_*/PBS_*/BASH_ENV overrides.
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "PBS_CONF_FILE",
        "SLURM_CONF",
    }
    environment = {k: v for k, v in os.environ.items() if k in allowed}
    import tempfile

    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            timeout=30,
            check=False,
        )
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read(1_000_001)
        error_output = stderr.read(1_000_001)
    if len(output) > 1_000_000 or len(error_output) > 1_000_000:
        raise ValueError("scheduler response exceeds size limit")
    return subprocess.CompletedProcess(
        argv,
        completed.returncode,
        output.decode("utf-8"),
        error_output.decode("utf-8"),
    )


class SchedulerExecutor:
    def __init__(self, store: CampaignStore):
        self.store = store

    def _attempt(self, experiment_id: str) -> dict | None:
        with self.store.transaction() as connection:
            if not scheduler_records(connection):
                return None
            row = connection.execute(
                "SELECT * FROM scheduler_attempts WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            return dict(row) if row else None

    def _directory(self, experiment_id: str, token: str) -> Path:
        path = self.store.root / "runs" / experiment_id / token
        if path.resolve() != path or not path.is_relative_to(self.store.root):
            raise ValueError("scheduler run directory must not traverse symlinks")
        return path

    def _script(self, profile: dict, directory: Path, token: str) -> str:
        resources = {**profile["resources"], "job_name": "lx-" + token[:12]}
        command = [
            profile["python"],
            "-m",
            "loopx_hpc.batch_worker",
            str(directory / "packet.json"),
        ]
        script = render_script(
            profile["scheduler"], command, resources, profile["environment"]
        )
        script = script.replace(
            "# Preview only: no submission or site validation has occurred.",
            "# Owner-reviewed scheduler attempt; live site acceptance is separate.",
        )
        directive = (
            "#PBS -r n" if profile["scheduler"] == "pbs" else "#SBATCH --no-requeue"
        )
        shebang, _, body = script.partition("\n")
        return shebang + "\n" + directive + "\n" + body

    def submit(
        self, experiment_id: str, profile: dict, *, execute: bool = False
    ) -> dict:
        if type(execute) is not bool:
            raise ValueError("execute must be a boolean")
        profile = validate_profile(profile)
        record = self.store.get_experiment(experiment_id)
        if not execute:
            token = record["token"] or "0" * 32
            directory = self._directory(experiment_id, token)
            return {
                "preview": True,
                "backend": profile["scheduler"],
                "will_launch": record["status"] == "planned",
                "site_validated": False,
                "profile_digest": digest(profile),
                "script": self._script(profile, directory, token),
                "submit_argv": preview_plan(
                    profile["scheduler"], str(directory / "job.sh")
                )["submit_argv"],
            }
        # Validate the environment/script before recording an attempt or writing files.
        self._script(profile, self._directory(experiment_id, "0" * 32), "0" * 32)
        with self.store.transaction() as connection:
            _schema(connection)
            row = connection.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            previous = connection.execute(
                "SELECT * FROM scheduler_attempts WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row["status"] != "planned":
                if previous is None or previous["profile_digest"] != digest(profile):
                    raise ValueError(
                        "attempt already belongs to another backend or profile"
                    )
                return self.store._record(row)
            occupied = connection.execute(
                "SELECT COUNT(*) FROM experiments WHERE status IN ('submitting','running','unknown')"
            ).fetchone()[0]
            spec = self.store._manifest(connection)
            if occupied >= spec["limits"]["max_concurrent"]:
                raise ValueError(
                    "concurrency budget exhausted; unresolved attempts reserve capacity"
                )
            token = uuid.uuid4().hex
            directory = self._directory(experiment_id, token)
            packet = {
                "schema_version": "loopx_hpc_batch_packet_v1",
                "experiment_id": experiment_id,
                "token": token,
                "manifest_digest": digest(spec),
                "config": json.loads(row["config"]),
                "profile_digest": digest(profile),
                "backend": profile["scheduler"],
                "command": spec["command"],
                "launcher": profile["launcher"],
                "python": profile["python"],
                "working_directory": profile["environment"].get("working_directory")
                or str(directory),
                "timeout_seconds": spec["limits"]["timeout_seconds"],
                "metric_name": spec["metric"]["name"],
            }
            timestamp = now()
            connection.execute(
                "UPDATE experiments SET status='submitting',token=?,updated_at=? WHERE id=?",
                (token, timestamp, experiment_id),
            )
            connection.execute(
                "INSERT INTO scheduler_attempts (experiment_id,token,backend,profile,profile_digest,packet_digest,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    token,
                    profile["scheduler"],
                    canonical(profile),
                    digest(profile),
                    digest(packet),
                    timestamp,
                    timestamp,
                ),
            )
            self.store.event(
                connection,
                "attempt_reserved",
                experiment_id,
                {
                    "token": token,
                    "backend": profile["scheduler"],
                    "profile_digest": digest(profile),
                },
            )
        dispatched = False
        response = None
        try:
            directory.mkdir(parents=True, mode=0o700)
            _write_new(directory / "config.json", canonical(packet["config"]) + "\n")
            _write_new(directory / "packet.json", canonical(packet) + "\n")
            _write_new(directory / "job.sh", self._script(profile, directory, token))
            _sync_directory(directory)
            script_check = subprocess.run(
                ["/bin/bash", "-n", str(directory / "job.sh")],
                capture_output=True,
                timeout=10,
                check=False,
            )
            if script_check.returncode:
                raise ValueError("generated batch script failed bash syntax validation")
            argv = preview_plan(profile["scheduler"], str(directory / "job.sh"))[
                "submit_argv"
            ]
            dispatched = True  # An interrupted/failed submission may still have reached the server.
            response = _run_cli(argv, directory)
            job_id = response.stdout.strip()
            if profile["scheduler"] == "slurm":
                # This controller is site-local, not a federated-cluster router.
                if ";" in job_id:
                    raise ValueError(
                        "federated Slurm response requires an explicit site binding"
                    )
            preview_plan(profile["scheduler"], str(directory / "job.sh"), job_id=job_id)
            if response.returncode or not job_id or "\n" in job_id:
                raise ValueError("scheduler submission acknowledgement is unresolved")
            with self.store.transaction() as connection:
                connection.execute(
                    "UPDATE scheduler_attempts SET job_id=?,updated_at=? WHERE experiment_id=? AND token=?",
                    (job_id, now(), experiment_id, token),
                )
                self.store.event(
                    connection,
                    "scheduler_submitted",
                    experiment_id,
                    {
                        "token": token,
                        "backend": profile["scheduler"],
                        "job_id": job_id,
                        "profile_digest": digest(profile),
                    },
                )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # Native scheduler diagnostics can include private site details. Keep
            # bounded stderr beside the private attempt packet, never in SQLite,
            # status/context, or tracker projections. Diagnostic persistence is
            # subordinate to the conservative unknown/no-retry transition.
            if dispatched and response is not None and response.stderr:
                try:
                    _write_new(
                        directory / "scheduler-submit.stderr", response.stderr
                    )
                    _sync_directory(directory)
                except OSError:
                    pass
            self._unresolved(
                experiment_id,
                token,
                "submission_unresolved:" + type(error).__name__
                if dispatched
                else "staging_failed:" + type(error).__name__,
                failed=not dispatched,
            )
        return self.store.get_experiment(experiment_id)

    def _unresolved(
        self, experiment_id: str, token: str, reason: str, *, failed: bool = False
    ) -> None:
        status = "failed" if failed else "unknown"
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT status,failure FROM experiments WHERE id=? AND token=?",
                (experiment_id, token),
            ).fetchone()
            if (
                row is None
                or row["status"] not in ACTIVE
                or (row["status"] == status and row["failure"] == reason)
            ):
                return
            connection.execute(
                "UPDATE experiments SET status=?,failure=?,updated_at=? WHERE id=? AND token=?",
                (status, reason, now(), experiment_id, token),
            )
            self.store.event(
                connection,
                "attempt_finished" if failed else "attempt_unresolved",
                experiment_id,
                {
                    "token": token,
                    "status": status,
                    "failure": reason,
                    "retry_allowed": False,
                },
            )

    def _receipt(self, attempt: dict, record: dict) -> dict:
        directory = self._directory(record["id"], attempt["token"])
        _, packet = _read_artifact(directory / "packet.json")
        _, receipt = _read_artifact(directory / "receipt.json")
        if digest(packet) != attempt["packet_digest"] or not isinstance(receipt, dict):
            raise ValueError("batch packet/receipt mismatch")
        expected = {
            "schema_version",
            "experiment_id",
            "token",
            "packet_digest",
            "job_id",
            "backend",
            "started_at",
            "finished_at",
            "status",
            "failure",
            "metrics",
            "artifacts",
        }
        if (
            set(receipt) != expected
            or receipt["schema_version"] != "loopx_hpc_batch_receipt_v1"
        ):
            raise ValueError("invalid batch receipt schema")
        if (
            receipt["experiment_id"] != record["id"]
            or receipt["token"] != attempt["token"]
            or receipt["packet_digest"] != attempt["packet_digest"]
            or receipt["backend"] != attempt["backend"]
        ):
            raise ValueError("batch receipt does not bind this exact attempt")
        preview_plan(
            attempt["backend"], str(directory / "job.sh"), job_id=receipt["job_id"]
        )
        if attempt["job_id"] and receipt["job_id"] != attempt["job_id"]:
            raise ValueError("batch receipt job id mismatch")
        times = [
            datetime.fromisoformat(receipt[k]) for k in ("started_at", "finished_at")
        ]
        if any(t.tzinfo is None for t in times) or times[1] < times[0]:
            raise ValueError("invalid batch receipt timestamps")
        if receipt["status"] not in {"succeeded", "failed"}:
            raise ValueError("batch worker did not produce a resolved receipt")
        return receipt

    def reconcile(self, experiment_id: str) -> dict:
        record = self.store.get_experiment(experiment_id)
        attempt = self._attempt(experiment_id)
        if attempt is None:
            raise ValueError("experiment has no scheduler attempt")
        if record["status"] not in ACTIVE:
            return record
        directory = self._directory(experiment_id, attempt["token"])
        if not attempt["job_id"]:
            # A lost qsub/sbatch acknowledgement can be recovered ONLY from an
            # exact compute receipt, then independently checked in accounting.
            try:
                receipt = self._receipt(attempt, record)
                attempt["job_id"] = receipt["job_id"]
            except (OSError, ValueError, TypeError, KeyError):
                self._unresolved(
                    experiment_id,
                    attempt["token"],
                    "submission_ack_and_receipt_missing",
                )
                return self.store.get_experiment(experiment_id)
        try:
            argv = preview_plan(
                attempt["backend"], str(directory / "job.sh"), job_id=attempt["job_id"]
            )["status_argv"]
            response = _run_cli(argv, directory)
            observation = parse_status(attempt["backend"], response.stdout)
            if response.returncode or observation["job_id"] != attempt["job_id"]:
                raise ValueError("scheduler accounting identity unresolved")
        except (OSError, ValueError, subprocess.SubprocessError):
            self._unresolved(
                experiment_id, attempt["token"], "scheduler_accounting_unresolved"
            )
            return self.store.get_experiment(experiment_id)
        receipt = None
        metrics, artifacts = {}, []
        status = {
            "queued": "submitting",
            "running": "running",
            "cancelled": "failed",
        }.get(observation["status"], observation["status"])
        failure = (
            None
            if status in {"submitting", "running", "succeeded"}
            else "scheduler_" + observation["status"]
        )
        if status == "failed":
            # Native terminal failure alone safely frees the allocation. If a
            # bound worker receipt exists, retain its actual execution times;
            # otherwise only the controller's accounting observation is known.
            try:
                receipt = self._receipt(attempt, record)
            except (OSError, ValueError, TypeError, KeyError):
                receipt = None
        if observation["status"] == "succeeded":
            try:
                receipt = self._receipt(attempt, record)
                if receipt["status"] != "succeeded" or receipt["failure"] is not None:
                    raise ValueError("worker did not succeed")
                metrics = validate_metrics(
                    receipt["metrics"], self.store.manifest()["metric"]["name"]
                )
                expected_artifacts = []
                for name, expected in (
                    ("config.json", record["config"]),
                    ("result.json", {"metrics": metrics}),
                ):
                    data, value = _read_artifact(directory / name)
                    if canonical(value) != canonical(expected):
                        raise ValueError("frozen input or result mismatch")
                    expected_artifacts.append(
                        {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
                    )
                if canonical(receipt["artifacts"]) != canonical(expected_artifacts):
                    raise ValueError("worker artifact digests mismatch")
                artifacts = [
                    {
                        **a,
                        "path": str(
                            (directory / a["path"]).relative_to(self.store.root)
                        ),
                    }
                    for a in expected_artifacts
                ]
            except (OSError, ValueError, TypeError, KeyError):
                status, failure, metrics, artifacts = (
                    "unknown",
                    "terminal_receipt_unverified",
                    {},
                    [],
                )
        with self.store.transaction() as connection:
            current = connection.execute(
                "SELECT status,failure FROM experiments WHERE id=? AND token=?",
                (experiment_id, attempt["token"]),
            ).fetchone()
            if current["status"] not in ACTIVE:
                return self.store._record(
                    connection.execute(
                        "SELECT * FROM experiments WHERE id=?", (experiment_id,)
                    ).fetchone()
                )
            previous = connection.execute(
                "SELECT observation,job_id FROM scheduler_attempts WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            changed = (
                previous["observation"] != canonical(observation)
                or previous["job_id"] != attempt["job_id"]
            )
            if changed:
                connection.execute(
                    "UPDATE scheduler_attempts SET job_id=?,observation=?,updated_at=? WHERE experiment_id=?",
                    (attempt["job_id"], canonical(observation), now(), experiment_id),
                )
                self.store.event(
                    connection,
                    "scheduler_observed",
                    experiment_id,
                    {
                        "token": attempt["token"],
                        "backend": attempt["backend"],
                        "job_id": attempt["job_id"],
                        "observation": observation,
                    },
                )
            if changed or current["status"] != status or current["failure"] != failure:
                connection.execute(
                    "UPDATE experiments SET status=?,failure=?,metrics=?,artifacts=?,updated_at=? WHERE id=? AND token=?",
                    (
                        status,
                        failure,
                        canonical(metrics),
                        canonical(artifacts),
                        now(),
                        experiment_id,
                        attempt["token"],
                    ),
                )
                if status in {"succeeded", "failed"}:
                    payload = {
                        "token": attempt["token"],
                        "backend": attempt["backend"],
                        "status": status,
                        "failure": failure,
                        "metrics": metrics,
                        "artifacts": artifacts,
                    }
                    if receipt:
                        payload.update(
                            {k: receipt[k] for k in ("started_at", "finished_at")}
                        )
                    self.store.event(
                        connection, "attempt_finished", experiment_id, payload
                    )
        return self.store.get_experiment(experiment_id)

    def cancel(self, experiment_id: str, *, execute: bool = False) -> dict:
        if type(execute) is not bool:
            raise ValueError("execute must be a boolean")
        record = self.store.get_experiment(experiment_id)
        attempt = self._attempt(experiment_id)
        if not attempt or not attempt["job_id"]:
            raise ValueError("cancel requires a durably bound scheduler job id")
        directory = self._directory(experiment_id, attempt["token"])
        argv = preview_plan(
            attempt["backend"], str(directory / "job.sh"), job_id=attempt["job_id"]
        )["cancel_argv"]
        if not execute or record["status"] not in ACTIVE:
            return {
                "preview": not execute,
                "will_cancel": record["status"] in ACTIVE,
                "cancel_argv": argv,
            }
        # Cancellation intent is durable and repeat calls are observations, not
        # blind repeat side effects. Acknowledgement never proves termination.
        with self.store.transaction() as connection:
            prior = connection.execute(
                "SELECT 1 FROM events WHERE experiment_id=? AND kind='scheduler_cancel_requested'",
                (experiment_id,),
            ).fetchone()
            if prior:
                return {
                    "cancel_requested": True,
                    "replayed": True,
                    "terminal_confirmed": False,
                }
            current = connection.execute(
                "SELECT token,status FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            bound = connection.execute(
                "SELECT token,backend,job_id FROM scheduler_attempts WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if current["status"] not in ACTIVE:
                return {"preview": False, "will_cancel": False, "cancel_argv": argv}
            if current["token"] != attempt["token"] or any(
                bound[field] != attempt[field]
                for field in ("token", "backend", "job_id")
            ):
                raise ValueError("scheduler cancel attempt identity changed")
            expected_name = "lx-" + attempt["token"][:12]
            query = (
                ["qstat", "-f", "-F", "json", attempt["job_id"]]
                if attempt["backend"] == "pbs"
                else [
                    "squeue",
                    "--noheader",
                    "--jobs",
                    attempt["job_id"],
                    "--format=%i|%j",
                ]
            )
            # A numeric ID can be recycled or resolve on the wrong cluster.
            # Reprove our generated name against one active native job before
            # recording any destructive intent. Query failures are read-only and
            # may safely be retried. Holding the local write lock prevents two
            # local cancel callers from both admitting a destructive intent.
            # This check is not an atomic scheduler-side compare-and-cancel;
            # trusted scheduler configuration and credentials remain required.
            try:
                proof = _run_cli(query, directory)
                if (
                    proof.returncode
                    or not isinstance(proof.stdout, str)
                    or len(proof.stdout) > 1_000_000
                ):
                    raise ValueError("cancel identity query failed")
                if attempt["backend"] == "pbs":
                    observed = parse_status("pbs", proof.stdout)
                    if observed["job_id"] != attempt["job_id"] or observed[
                        "status"
                    ] not in {"queued", "running"}:
                        raise ValueError(
                            "cancel identity query lacks one exact active job"
                        )
                    # parse_status already rejects duplicate JSON keys, malformed
                    # objects and ambiguous/multiple job records before this read.
                    job = json.loads(proof.stdout)["Jobs"][attempt["job_id"]]
                    verified = job.get("Job_Name") == expected_name
                else:
                    lines = proof.stdout.splitlines()
                    verified = len(lines) == 1 and lines[0].split("|") == [
                        attempt["job_id"],
                        expected_name,
                    ]
                if not verified:
                    raise ValueError("cancel identity query job name mismatch")
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                RecursionError,
                subprocess.SubprocessError,
            ) as exc:
                raise ValueError(
                    "scheduler cancel identity unresolved; no cancellation requested"
                ) from exc
            self.store.event(
                connection,
                "scheduler_cancel_requested",
                experiment_id,
                {
                    "token": attempt["token"],
                    "backend": attempt["backend"],
                    "job_id": attempt["job_id"],
                    "job_name": expected_name,
                    "identity_checked_at": now(),
                    "identity_query_digest": hashlib.sha256(
                        proof.stdout.encode()
                    ).hexdigest(),
                },
            )
        try:
            response = _run_cli(argv, directory)
            acknowledged = response.returncode == 0
        except (OSError, ValueError, subprocess.SubprocessError):
            acknowledged = False
        return {
            "cancel_requested": True,
            "acknowledged": acknowledged,
            "terminal_confirmed": False,
        }
