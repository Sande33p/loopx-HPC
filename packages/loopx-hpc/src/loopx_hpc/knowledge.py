"""Explicit, local sharing of immutable advisory research lessons.

This library never fetches source campaigns or treats imported text as a tool
instruction. Content hashes establish byte-level integrity, not truth,
authorship, scientific comparability, or authority to act. Tags use exact,
case-sensitive AND matching; empty queries list all active lessons.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator
import uuid


SCHEMA_VERSION = "loopx_hpc_lesson_v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_LESSON_ID = re.compile(r"lesson-[0-9a-f]{64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_BOUNDARY = {
    "advisory": True,
    "advisory_only": True,
    "imported_text_trusted": False,
    "content_integrity_verified": True,
    "truth_verified": False,
    "authorship_verified": False,
    "source_accessed": False,
    "action_authority": "none",
    "privacy": "explicit_local_library_not_public_safe",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _text(value: Any, label: str, *, limit: int = 8192) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be nonempty bounded text")


def _identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a bounded simple identifier")


def _tags(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("tags must be a list of at most 32 identifiers")
    for tag in value:
        _identifier(tag, "tag")
    if len(set(value)) != len(value):
        raise ValueError("tags must be unique")
    return value


def _sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 digest")


def _utc_timestamp(value: Any, label: str) -> None:
    try:
        timestamp = datetime.fromisoformat(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} must be an ISO UTC timestamp") from error
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an ISO UTC timestamp")


def _validate_assessment(record: Any, source: dict) -> None:
    if not isinstance(record, dict) or set(record) != {
        "id",
        "kind",
        "recorded_at",
        "payload",
        "digest",
    }:
        raise ValueError("invalid assessment envelope")
    _identifier(record["id"], "assessment id")
    if record["kind"] != "assessment":
        raise ValueError("lesson evidence must contain assessment records")
    _sha256(record["digest"], "assessment digest")
    if record["digest"] != _digest(
        {key: value for key, value in record.items() if key != "digest"}
    ):
        raise ValueError("assessment content hash mismatch")
    _utc_timestamp(record["recorded_at"], "assessment recorded_at")
    payload = record["payload"]
    if not isinstance(payload, dict) or set(payload) != {
        "id",
        "hypothesis",
        "outcome",
        "interpretation",
        "limitations",
        "next_action",
        "experiment_ids",
        "agent",
        "supersedes",
        "source",
        "evidence",
    }:
        raise ValueError("invalid assessment payload")
    if payload["id"] != record["id"]:
        raise ValueError("assessment payload id mismatch")
    if payload["source"] != {
        key: source[key] for key in ("campaign_id", "manifest_digest")
    }:
        raise ValueError("assessment must bind the lesson source campaign and manifest")
    for key in ("hypothesis", "interpretation", "next_action"):
        _text(payload[key], f"assessment {key}")
    limitations = payload["limitations"]
    if not isinstance(limitations, list) or not limitations or len(limitations) > 32:
        raise ValueError("assessment limitations must have 1..32 explicit statements")
    for value in limitations:
        _text(value, "assessment limitation", limit=2048)
    if payload["supersedes"] is not None:
        _identifier(payload["supersedes"], "superseded assessment id")
        if payload["supersedes"] == payload["id"]:
            raise ValueError("assessment cannot supersede itself")
    agent = payload["agent"]
    if not isinstance(agent, dict) or set(agent) != {"runtime", "model"}:
        raise ValueError("assessment agent requires runtime and model labels")
    for key in ("runtime", "model"):
        _text(agent[key], f"assessment agent {key}", limit=256)
    outcome = payload["outcome"]
    if outcome not in ("supports", "refutes", "inconclusive", "infrastructure_failure"):
        raise ValueError("invalid assessment outcome")
    experiment_ids = payload["experiment_ids"]
    if (
        not isinstance(experiment_ids, list)
        or not experiment_ids
        or len(experiment_ids) > 128
    ):
        raise ValueError("assessment requires 1..128 experiment ids")
    for experiment_id in experiment_ids:
        _identifier(experiment_id, "assessment experiment id")
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("assessment experiment ids must be unique")
    evidence = payload["evidence"]
    if not isinstance(evidence, list) or len(evidence) != len(experiment_ids):
        raise ValueError("assessment evidence must match its experiment ids")
    evidence_ids = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {
            "experiment_id",
            "config_digest",
            "attempt_token",
            "status",
            "metrics",
            "artifact_digests",
            "failure",
            "evidence_digest",
            "execution_events",
        }:
            raise ValueError("invalid captured evidence fields")
        _identifier(item["experiment_id"], "evidence experiment id")
        evidence_ids.append(item["experiment_id"])
        _sha256(item["config_digest"], "evidence config digest")
        _identifier(item["attempt_token"], "evidence attempt token")
        _sha256(item["evidence_digest"], "evidence digest")
        if item["evidence_digest"] != _digest(
            {key: value for key, value in item.items() if key != "evidence_digest"}
        ):
            raise ValueError("captured evidence content hash mismatch")
        expected_status = (
            "failed" if outcome == "infrastructure_failure" else "succeeded"
        )
        if item["status"] != expected_status:
            raise ValueError(
                "assessment outcome is incompatible with captured execution status"
            )
        metrics = item["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("captured evidence metrics must be a mapping")
        for name, value in metrics.items():
            _identifier(name, "captured metric name")
            if type(value) not in (int, float):
                raise ValueError("captured evidence metrics must be numeric")
        if expected_status == "succeeded" and source["metric"]["name"] not in metrics:
            raise ValueError("successful evidence requires the primary metric")
        artifact_digests = item["artifact_digests"]
        if not isinstance(artifact_digests, list) or len(artifact_digests) > 128:
            raise ValueError("artifact digests must be a bounded list")
        for artifact_digest in artifact_digests:
            _sha256(artifact_digest, "artifact digest")
        if item["failure"] is not None:
            _text(item["failure"], "captured failure")
        if expected_status == "succeeded" and item["failure"] is not None:
            raise ValueError("successful evidence cannot contain a failure")
        execution_events = item["execution_events"]
        if (
            not isinstance(execution_events, list)
            or not 1 <= len(execution_events) <= 128
        ):
            raise ValueError(
                "captured evidence requires 1..128 original execution events"
            )
        previous_sequence = 0
        for event in execution_events:
            if not isinstance(event, dict) or set(event) != {
                "sequence",
                "kind",
                "recorded_at",
                "event_digest",
            }:
                raise ValueError("invalid execution event fields")
            if (
                type(event["sequence"]) is not int
                or event["sequence"] <= previous_sequence
            ):
                raise ValueError(
                    "execution event sequences must be positive and strictly increasing"
                )
            previous_sequence = event["sequence"]
            _identifier(event["kind"], "execution event kind")
            _utc_timestamp(event["recorded_at"], "execution event recorded_at")
            _sha256(event["event_digest"], "execution event digest")
    if set(evidence_ids) != set(experiment_ids) or len(set(evidence_ids)) != len(
        evidence_ids
    ):
        raise ValueError("assessment evidence identities do not match experiment ids")


def validate_bundle(bundle: dict) -> dict:
    """Validate structure/hash and return detached data, without source access."""
    if not isinstance(bundle, dict):
        raise ValueError("lesson bundle must be a JSON object")
    try:
        raw = _canonical(bundle)
        if len(raw.encode()) > 1_000_000:
            raise ValueError("lesson bundle exceeds 1 MB")
        bundle = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("lesson bundle must be bounded finite JSON data") from error
    if (
        set(bundle) != {"schema_version", "id", "payload"}
        or bundle["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("invalid lesson bundle schema")
    payload = bundle["payload"]
    if not isinstance(payload, dict) or set(payload) != {
        "title",
        "claim",
        "tags",
        "applicability",
        "limitations",
        "source",
        "assessments",
        "replaces",
    }:
        raise ValueError("invalid lesson payload fields")
    if bundle["id"] != "lesson-" + _digest(payload):
        raise ValueError("lesson content hash does not match id")
    _text(payload["title"], "title", limit=256)
    _text(payload["claim"], "claim")
    _tags(payload["tags"])
    for key in ("applicability", "limitations"):
        values = payload[key]
        if not isinstance(values, list) or not values or len(values) > 32:
            raise ValueError(f"{key} must have 1..32 explicit statements")
        for value in values:
            _text(value, key, limit=2048)
    source = payload["source"]
    if not isinstance(source, dict) or set(source) != {
        "campaign_id",
        "manifest_digest",
        "metric",
        "provenance",
    }:
        raise ValueError("invalid lesson source fields")
    _identifier(source["campaign_id"], "source campaign id")
    _sha256(source["manifest_digest"], "source manifest digest")
    metric = source["metric"]
    if not isinstance(metric, dict) or set(metric) != {"name", "direction"}:
        raise ValueError("source metric requires name and direction")
    _identifier(metric["name"], "source metric name")
    if metric["direction"] not in ("minimize", "maximize"):
        raise ValueError("source metric direction must be minimize or maximize")
    provenance = source["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("source provenance must be a mapping")
    for key in ("code_revision", "dataset_revision", "environment"):
        _text(provenance.get(key), f"source provenance {key}", limit=2048)
    assessments = payload["assessments"]
    if not isinstance(assessments, list) or not assessments or len(assessments) > 32:
        raise ValueError("lesson requires 1..32 captured assessments")
    assessment_ids = []
    for assessment in assessments:
        _validate_assessment(assessment, source)
        assessment_ids.append(assessment["id"])
    if len(set(assessment_ids)) != len(assessment_ids):
        raise ValueError("captured assessment ids must be unique")
    if any(record["payload"]["supersedes"] in assessment_ids for record in assessments):
        raise ValueError("lesson cannot include an explicitly superseded assessment")
    replaces = payload["replaces"]
    if replaces is not None and (
        not isinstance(replaces, str) or not _LESSON_ID.fullmatch(replaces)
    ):
        raise ValueError("replaces must be a lesson id or null")
    if replaces == bundle["id"]:
        raise ValueError("a lesson cannot replace itself")
    return bundle


class _KnowledgeSnapshot:
    """Read-only local lifecycle snapshot; valid only inside its context."""

    def __init__(
        self, library: KnowledgeLibrary, connection: sqlite3.Connection | None
    ):
        self._library = library
        self._connection = connection
        self._closed = False
        # This SELECT acquires the SHARED lock now, not on the first get().
        metadata = library._metadata(connection)
        self.identity = self.library_id = metadata["library_id"]
        self.revision = metadata["library_revision"]

    def get(self, lesson_id: str) -> dict:
        if self._closed:
            raise RuntimeError("knowledge snapshot is closed")
        row = self._library._row(self._connection, lesson_id)
        if row is None:
            raise ValueError("lesson not found")
        return self._library._view(self._connection, row)


class KnowledgeLibrary:
    """A user-selected local library; construction and previews never write."""

    def __init__(self, root: Path):
        if "://" in str(root):
            raise ValueError("knowledge root must be an explicit local directory")
        self.root = Path(root).expanduser().resolve()
        self.database = self.root / "knowledge.sqlite3"

    @contextmanager
    def _connection(
        self, *, write: bool = False
    ) -> Iterator[sqlite3.Connection | None]:
        if not write and not self.database.is_file():
            yield None
            return
        if write:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = sqlite3.connect(
                self.database, timeout=30, isolation_level=None
            )
        else:
            connection = sqlite3.connect(
                self.database.as_uri() + "?mode=ro",
                uri=True,
                timeout=30,
                isolation_level=None,
            )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA journal_mode").fetchone()[0] not in {
                "delete",
                "truncate",
                "persist",
            }:
                raise ValueError(
                    "knowledge lifecycle snapshots require rollback-journal mode, not WAL"
                )
            if write:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS library_metadata (
                        key TEXT PRIMARY KEY, value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS lessons (
                        id TEXT PRIMARY KEY, bundle TEXT NOT NULL, published_at TEXT NOT NULL,
                        withdrawn_reason TEXT, superseded_by TEXT REFERENCES lessons(id)
                    );
                    CREATE TABLE IF NOT EXISTS lesson_events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        lesson_id TEXT NOT NULL REFERENCES lessons(id),
                        event TEXT NOT NULL, detail TEXT NOT NULL, recorded_at TEXT NOT NULL
                    );
                    CREATE TRIGGER IF NOT EXISTS lesson_bundle_immutable
                    BEFORE UPDATE OF id,bundle ON lessons BEGIN
                        SELECT RAISE(ABORT, 'lesson bundles are immutable');
                    END;
                    CREATE TRIGGER IF NOT EXISTS lesson_history_retained
                    BEFORE DELETE ON lessons BEGIN
                        SELECT RAISE(ABORT, 'lesson history must be retained');
                    END;
                    CREATE TRIGGER IF NOT EXISTS lesson_events_immutable
                    BEFORE UPDATE ON lesson_events BEGIN
                        SELECT RAISE(ABORT, 'lesson events are immutable');
                    END;
                    CREATE TRIGGER IF NOT EXISTS lesson_events_retained
                    BEFORE DELETE ON lesson_events BEGIN
                        SELECT RAISE(ABORT, 'lesson events must be retained');
                    END;
                    CREATE TRIGGER IF NOT EXISTS library_identity_immutable
                    BEFORE UPDATE ON library_metadata BEGIN
                        SELECT RAISE(ABORT, 'library identity is immutable');
                    END;
                    CREATE TRIGGER IF NOT EXISTS library_identity_retained
                    BEFORE DELETE ON library_metadata BEGIN
                        SELECT RAISE(ABORT, 'library identity must be retained');
                    END;
                """)
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            if write:
                connection.execute(
                    "INSERT OR IGNORE INTO library_metadata(key,value) VALUES ('library_id',?)",
                    ("library-" + uuid.uuid4().hex,),
                )
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _metadata(connection: sqlite3.Connection | None) -> dict:
        if connection is None:
            return {"library_id": None, "library_revision": 0}
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='library_metadata'"
        ).fetchone():
            raise ValueError(
                "library identity is not initialized; explicitly publish before reading"
            )
        row = connection.execute(
            "SELECT value FROM library_metadata WHERE key='library_id'"
        ).fetchone()
        if row is None or not re.fullmatch(r"library-[0-9a-f]{32}", row[0]):
            raise ValueError(
                "library identity is missing or invalid; explicitly publish to initialize"
            )
        revision = connection.execute(
            "SELECT COALESCE(MAX(sequence),0) FROM lesson_events"
        ).fetchone()[0]
        return {"library_id": row[0], "library_revision": revision}

    @contextmanager
    def snapshot(self) -> Iterator[_KnowledgeSnapshot]:
        """Hold lifecycle reads through a caller's campaign commit.

        Acquire the campaign transaction first, then this snapshot; commit the
        campaign before releasing it. Library writers never open a campaign DB.
        Rollback journaling prevents withdrawal/supersession from committing
        while this SHARED lock is held. A missing library remains read-only.
        """
        with self._connection() as connection:
            view = _KnowledgeSnapshot(self, connection)
            try:
                yield view
            finally:
                view._closed = True

    @staticmethod
    def _row(
        connection: sqlite3.Connection | None, lesson_id: str
    ) -> sqlite3.Row | None:
        if not isinstance(lesson_id, str) or not _LESSON_ID.fullmatch(lesson_id):
            raise ValueError("invalid lesson id")
        if connection is None:
            return None
        return connection.execute(
            "SELECT * FROM lessons WHERE id=?", (lesson_id,)
        ).fetchone()

    @staticmethod
    def _state(row: sqlite3.Row) -> str:
        return (
            "withdrawn"
            if row["withdrawn_reason"] is not None
            else "superseded"
            if row["superseded_by"]
            else "active"
        )

    @staticmethod
    def _bundle(row: sqlite3.Row) -> dict:
        bundle = validate_bundle(json.loads(row["bundle"]))
        if bundle["id"] != row["id"]:
            raise ValueError("stored lesson identity does not match its bundle")
        return bundle

    def _view(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
        events = [
            dict(event)
            for event in connection.execute(
                "SELECT sequence,event,detail,recorded_at FROM lesson_events WHERE lesson_id=? ORDER BY sequence",
                (row["id"],),
            )
        ]
        for event in events:
            event["detail"] = json.loads(event["detail"])
        return {
            "bundle": self._bundle(row),
            "state": self._state(row),
            "published_at": row["published_at"],
            "withdrawn_reason": row["withdrawn_reason"],
            "superseded_by": row["superseded_by"],
            "history": events,
            **self._metadata(connection),
            **_BOUNDARY,
        }

    def get(self, lesson_id: str) -> dict:
        with self.snapshot() as view:
            return view.get(lesson_id)

    def search(self, tags: list[str], limit: int = 10) -> list[dict]:
        """Return active lessons matching all tags, ordered by immutable id."""
        tags = _tags(tags)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("search limit must be an integer from 1 to 100")
        result = []
        with self._connection() as connection:
            if connection is None:
                return result
            self._metadata(connection)
            for row in connection.execute(
                "SELECT * FROM lessons WHERE withdrawn_reason IS NULL AND superseded_by IS NULL ORDER BY id"
            ):
                bundle = self._bundle(row)
                if set(tags).issubset(bundle["payload"]["tags"]):
                    result.append(self._view(connection, row))
                    if len(result) == limit:
                        break
        return result

    def publish(self, bundle: dict, execute: bool = False) -> dict:
        bundle = validate_bundle(bundle)
        if type(execute) is not bool:
            raise ValueError("execute must be a boolean")
        lesson_id = bundle["id"]
        replaces = bundle["payload"]["replaces"]
        with self._connection(write=execute) as connection:
            existing = self._row(connection, lesson_id)
            if existing is not None:
                if self._bundle(existing) != bundle:
                    raise ValueError("conflicting immutable lesson identity")
                return {
                    "operation": "publish",
                    "lesson_id": lesson_id,
                    "execute": execute,
                    "would_change": False,
                    "changed": False,
                    "reused": True,
                    "state": self._state(existing),
                    **self._metadata(connection),
                    **_BOUNDARY,
                }
            predecessor = self._row(connection, replaces) if replaces else None
            if replaces:
                if predecessor is None or self._state(predecessor) != "active":
                    raise ValueError("replacement requires an existing active lesson")
                original_source = self._bundle(predecessor)["payload"]["source"]
                source = bundle["payload"]["source"]
                if source != original_source:
                    raise ValueError(
                        "replacement must preserve exact source campaign, manifest, metric and provenance"
                    )
            if execute:
                timestamp = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    "INSERT INTO lessons(id,bundle,published_at) VALUES (?,?,?)",
                    (lesson_id, _canonical(bundle), timestamp),
                )
                connection.execute(
                    "INSERT INTO lesson_events(lesson_id,event,detail,recorded_at) VALUES (?,?,?,?)",
                    (
                        lesson_id,
                        "published",
                        _canonical({"replaces": replaces}),
                        timestamp,
                    ),
                )
                if replaces:
                    connection.execute(
                        "UPDATE lessons SET superseded_by=? WHERE id=?",
                        (lesson_id, replaces),
                    )
                    connection.execute(
                        "INSERT INTO lesson_events(lesson_id,event,detail,recorded_at) VALUES (?,?,?,?)",
                        (
                            replaces,
                            "superseded",
                            _canonical({"superseded_by": lesson_id}),
                            timestamp,
                        ),
                    )
            return {
                "operation": "publish",
                "lesson_id": lesson_id,
                "execute": execute,
                "would_change": True,
                "changed": execute,
                "reused": False,
                "state": "active" if execute else "not_published",
                "proposed_state": "active",
                "replaces": replaces,
                **self._metadata(connection),
                **_BOUNDARY,
            }

    def withdraw(self, lesson_id: str, reason: str, execute: bool = False) -> dict:
        _text(reason, "withdrawal reason", limit=2048)
        if type(execute) is not bool:
            raise ValueError("execute must be a boolean")
        with self._connection(write=execute) as connection:
            row = self._row(connection, lesson_id)
            if row is None:
                raise ValueError("lesson not found")
            self._bundle(row)
            repeated = row["withdrawn_reason"] is not None
            if repeated and row["withdrawn_reason"] != reason:
                raise ValueError(
                    "withdrawal reason is immutable; repeated withdrawal must match"
                )
            if execute and not repeated:
                timestamp = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    "UPDATE lessons SET withdrawn_reason=? WHERE id=?",
                    (reason, lesson_id),
                )
                connection.execute(
                    "INSERT INTO lesson_events(lesson_id,event,detail,recorded_at) VALUES (?,?,?,?)",
                    (lesson_id, "withdrawn", _canonical({"reason": reason}), timestamp),
                )
            return {
                "operation": "withdraw",
                "lesson_id": lesson_id,
                "execute": execute,
                "would_change": not repeated,
                "changed": execute and not repeated,
                "reused": repeated,
                "state": "withdrawn" if execute or repeated else self._state(row),
                "reason": reason,
                **self._metadata(connection),
                **_BOUNDARY,
            }
