"""Exact, local scientific records; not a second LoopX goal/todo authority.

One immutable study per directory. SQLite transactions serialize admission and
evidence-linked proposals. Unknown attempts reserve capacity until reconciled;
v0 deliberately has no automatic retries or remote execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator, Sequence

ACTIVE = ("submitting", "running", "unknown")
TERMINAL = ("succeeded", "failed")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value
    ):
        raise ValueError(f"{label} must be a bounded simple identifier")
    return value


def validate_manifest(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("manifest must be a JSON object")
    spec = json.loads(canonical(spec))
    required = {
        "id",
        "objective",
        "hypothesis",
        "metric",
        "search_space",
        "baseline",
        "command",
        "limits",
        "provenance",
    }
    if set(spec) != required:
        raise ValueError(f"manifest fields must be exactly {sorted(required)}")
    _identifier(spec["id"], "campaign id")
    for key in ("objective", "hypothesis"):
        if not isinstance(spec[key], str) or not spec[key].strip():
            raise ValueError(f"{key} is required")
    metric = spec["metric"]
    if not isinstance(metric, dict) or set(metric) != {"name", "direction"}:
        raise ValueError("metric needs name and direction")
    _identifier(metric["name"], "metric name")
    if metric["direction"] not in ("minimize", "maximize"):
        raise ValueError("metric direction must be minimize or maximize")
    space = spec["search_space"]
    if not isinstance(space, dict) or not space:
        raise ValueError("search_space must be a nonempty finite mapping")
    for name, values in space.items():
        _identifier(name, "parameter")
        if not isinstance(values, list) or not values or len(values) > 10000:
            raise ValueError("each search parameter needs 1..10000 scalar candidates")
        if any(type(v) not in (str, int, float, bool) for v in values):
            raise ValueError("search candidates must be JSON scalars")
        if len({canonical(v) for v in values}) != len(values):
            raise ValueError("duplicate search candidate")
    _validate_config(spec["baseline"], spec)
    command = spec["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(v, str) or not v or "\x00" in v for v in command)
    ):
        raise ValueError(
            "command must be a nonempty argv list; shell strings are not supported"
        )
    if "{config}" not in command or "{result}" not in command:
        raise ValueError(
            "command must contain separate {config} and {result} arguments"
        )
    limits = spec["limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "max_experiments",
        "max_concurrent",
        "timeout_seconds",
    }:
        raise ValueError("limits need max_experiments, max_concurrent, timeout_seconds")
    for key, value in limits.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if limits["max_concurrent"] > limits["max_experiments"]:
        raise ValueError("concurrency cannot exceed experiment budget")
    provenance = spec["provenance"]
    if not isinstance(provenance, dict) or not all(
        isinstance(provenance.get(k), str) and provenance[k].strip()
        for k in ("code_revision", "dataset_revision", "environment")
    ):
        raise ValueError(
            "provenance needs code_revision, dataset_revision and environment"
        )
    return spec


def _validate_config(config: dict, spec: dict) -> None:
    if not isinstance(config, dict) or set(config) != set(spec["search_space"]):
        raise ValueError("configuration must match the frozen search-space keys")
    for key, value in config.items():
        if canonical(value) not in {canonical(v) for v in spec["search_space"][key]}:
            raise ValueError(f"configuration outside allowed search space: {key}")


class CampaignStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.database = self.root / "campaign.sqlite3"

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if not self.database.is_file():
            raise ValueError("campaign not initialized; run init first")
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, spec: dict) -> dict:
        spec = validate_manifest(spec)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with sqlite3.connect(self.database, timeout=30) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL, config_hash TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('planned','submitting','running','succeeded','failed','unknown')),
                    rationale TEXT NOT NULL, parents TEXT NOT NULL, metrics TEXT NOT NULL DEFAULT '{}',
                    artifacts TEXT NOT NULL DEFAULT '[]', token TEXT, process_id INTEGER,
                    failure TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, payload TEXT NOT NULL, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                    experiment_id TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)
        self.database.chmod(0o600)
        with self.transaction() as connection:
            previous = connection.execute(
                "SELECT value FROM metadata WHERE key='manifest'"
            ).fetchone()
            if previous and previous[0] != canonical(spec):
                raise ValueError(
                    "study definition is immutable; use a new campaign directory"
                )
            if not previous:
                connection.execute(
                    "INSERT INTO metadata VALUES ('manifest', ?)", (canonical(spec),)
                )
                self.event(
                    connection,
                    "study_initialized",
                    None,
                    {"manifest_hash": digest(spec)},
                )
        return self.status()

    @staticmethod
    def event(
        connection: sqlite3.Connection,
        kind: str,
        experiment_id: str | None,
        payload: dict,
    ) -> None:
        connection.execute(
            "INSERT INTO events(kind,experiment_id,payload,created_at) VALUES (?,?,?,?)",
            (kind, experiment_id, canonical(payload), now()),
        )

    @staticmethod
    def _manifest(connection: sqlite3.Connection) -> dict:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='manifest'"
        ).fetchone()
        if row is None:
            raise ValueError("campaign initialization incomplete")
        return json.loads(row[0])

    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("config", "parents", "metrics", "artifacts"):
            result[key] = json.loads(result[key])
        return result

    def manifest(self) -> dict:
        with self.transaction() as connection:
            return self._manifest(connection)

    def experiments(self) -> list[dict]:
        with self.transaction() as connection:
            return [
                self._record(row)
                for row in connection.execute(
                    "SELECT * FROM experiments ORDER BY rowid"
                )
            ]

    def get_experiment(self, experiment_id: str) -> dict:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown experiment id")
            return self._record(row)

    def _add(
        self,
        connection: sqlite3.Connection,
        config: dict,
        rationale: str,
        parent_ids: Sequence[str],
    ) -> dict:
        spec = self._manifest(connection)
        _validate_config(config, spec)
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("experiment rationale is required")
        if (
            not isinstance(parent_ids, (list, tuple))
            or any(not isinstance(parent, str) for parent in parent_ids)
            or len(set(parent_ids)) != len(parent_ids)
        ):
            raise ValueError("parent_ids must contain unique experiment ids")
        for parent in parent_ids:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id=?", (parent,)
            ).fetchone()
            if (
                row is None
                or row["status"] != "succeeded"
                or not self.verify_record(self._record(row))
            ):
                raise ValueError(
                    "parent evidence must be a verified successful experiment"
                )
        config_hash = digest(config)
        previous = connection.execute(
            "SELECT * FROM experiments WHERE config_hash=?", (config_hash,)
        ).fetchone()
        if previous:
            record = self._record(previous)
            if record["rationale"] != rationale or record["parents"] != list(
                parent_ids
            ):
                raise ValueError(
                    "same configuration has different immutable decision lineage"
                )
            return record
        count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        if count >= spec["limits"]["max_experiments"]:
            raise ValueError("experiment budget exhausted")
        experiment_id = "exp-" + digest({"campaign": spec["id"], "config": config})[:24]
        timestamp = now()
        connection.execute(
            "INSERT INTO experiments(id,config,config_hash,status,rationale,parents,created_at,updated_at) VALUES (?,?,?,'planned',?,?,?,?)",
            (
                experiment_id,
                canonical(config),
                config_hash,
                rationale,
                canonical(list(parent_ids)),
                timestamp,
                timestamp,
            ),
        )
        self.event(
            connection,
            "experiment_planned",
            experiment_id,
            {
                "config_hash": config_hash,
                "parents": list(parent_ids),
                "rationale": rationale,
            },
        )
        return self._record(
            connection.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
        )

    def add_experiment(
        self, config: dict, *, rationale: str, parent_ids: Sequence[str] = ()
    ) -> dict:
        with self.transaction() as connection:
            return self._add(connection, config, rationale, parent_ids)

    def verify_record(self, record: dict) -> bool:
        if record["status"] != "succeeded" or not record["artifacts"]:
            return False
        if digest(record["config"]) != record["config_hash"]:
            return False
        for artifact in record["artifacts"]:
            path = (self.root / artifact["path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                return False
        config_artifacts = [
            a for a in record["artifacts"] if Path(a["path"]).name == "config.json"
        ]
        result_artifacts = [
            a for a in record["artifacts"] if Path(a["path"]).name == "result.json"
        ]
        if len(config_artifacts) != 1 or len(result_artifacts) != 1:
            return False
        try:
            frozen_config = json.loads(
                (self.root / config_artifacts[0]["path"]).read_text()
            )
            observed_result = json.loads(
                (self.root / result_artifacts[0]["path"]).read_text()
            )
            if digest(frozen_config) != record["config_hash"]:
                return False
            if canonical(observed_result) != canonical({"metrics": record["metrics"]}):
                return False
        except (ValueError, OSError, TypeError):
            return False
        return True

    def status(self) -> dict:
        with self.transaction() as connection:
            records = [
                self._record(row)
                for row in connection.execute(
                    "SELECT * FROM experiments ORDER BY rowid"
                )
            ]
            return {
                "campaign": self._manifest(connection),
                "experiments": records,
                "decisions": [
                    json.loads(row[0])
                    for row in connection.execute(
                        "SELECT payload FROM decisions ORDER BY rowid"
                    )
                ],
            }

    def context(self) -> dict:
        snapshot = self.status()
        metric = snapshot["campaign"]["metric"]
        eligible = [r for r in snapshot["experiments"] if self.verify_record(r)]
        incumbent = None
        if eligible:
            order = sorted(
                eligible,
                key=lambda r: r["metrics"][metric["name"]],
                reverse=metric["direction"] == "maximize",
            )
            incumbent = order[0]["id"]
        return {
            "schema_version": "loopx_hpc_decision_context_v0",
            "context_digest": digest(snapshot),
            **snapshot,
            "eligible_evidence_ids": [r["id"] for r in eligible],
            "incumbent_id": incumbent,
            "remaining_experiments": snapshot["campaign"]["limits"]["max_experiments"]
            - len(snapshot["experiments"]),
            "authority": "proposal_only; execution requires an explicit local execute or governed LoopX admission",
        }

    def apply_proposal(self, proposal: dict) -> dict:
        required = {
            "id",
            "context_digest",
            "runtime",
            "rationale",
            "evidence_ids",
            "configs",
        }
        if not isinstance(proposal, dict) or set(proposal) != required:
            raise ValueError(f"proposal fields must be exactly {sorted(required)}")
        _identifier(proposal["id"], "proposal id")
        _identifier(proposal["runtime"], "runtime label")
        if not isinstance(proposal["configs"], list) or not proposal["configs"]:
            raise ValueError("proposal requires candidate configs")
        if not isinstance(proposal["evidence_ids"], list):
            raise ValueError("evidence_ids must be a list")
        with self.transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM decisions WHERE id=?", (proposal["id"],)
            ).fetchone()
            if previous:
                if previous["payload"] != canonical(proposal):
                    raise ValueError("proposal id reused with different contents")
                return json.loads(previous["result"])
            snapshot = {
                "campaign": self._manifest(connection),
                "experiments": [
                    self._record(row)
                    for row in connection.execute(
                        "SELECT * FROM experiments ORDER BY rowid"
                    )
                ],
                "decisions": [
                    json.loads(row[0])
                    for row in connection.execute(
                        "SELECT payload FROM decisions ORDER BY rowid"
                    )
                ],
            }
            if digest(snapshot) != proposal["context_digest"]:
                raise ValueError(
                    "stale decision context; obtain a fresh context before proposing"
                )
            records = [
                self._add(
                    connection, config, proposal["rationale"], proposal["evidence_ids"]
                )
                for config in proposal["configs"]
            ]
            result = {
                "proposal_id": proposal["id"],
                "experiment_ids": [r["id"] for r in records],
            }
            connection.execute(
                "INSERT INTO decisions VALUES (?,?,?)",
                (proposal["id"], canonical(proposal), canonical(result)),
            )
            self.event(
                connection,
                "proposal_accepted",
                None,
                {
                    **result,
                    "runtime": proposal["runtime"],
                    "context_digest": proposal["context_digest"],
                },
            )
            return result


def validate_metrics(value: Any, metric_name: str) -> dict:
    if not isinstance(value, dict) or not value or metric_name not in value:
        raise ValueError("result must contain the preregistered primary metric")
    for key, number in value.items():
        _identifier(key, "metric name")
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("all metrics must be finite numbers")
    return value
