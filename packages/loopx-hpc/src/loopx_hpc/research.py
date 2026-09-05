"""Append-only scientific interpretations and explicit advisory lesson reuse.

These are authored, evidence-linked rationales, not model chain-of-thought or
automatically proven scientific conclusions. No operation here executes work.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import sqlite3
from typing import Any

from .campaign import (
    CampaignStore,
    _identifier,
    _validate_config,
    canonical,
    digest,
    now,
)
from .knowledge import KnowledgeLibrary, validate_bundle


def _text(value: Any, label: str, limit: int = 8192) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or "\x00" in value
    ):
        raise ValueError(f"{label} requires bounded nonempty text")


def _list(
    value: Any, label: str, *, identifiers: bool = False, minimum: int = 1
) -> None:
    if not isinstance(value, list) or not minimum <= len(value) <= 32:
        raise ValueError(f"{label} requires {minimum}..32 entries")
    for item in value:
        if identifiers:
            _identifier(item, label)
        else:
            _text(item, label, 2048)
    if identifiers and len(set(value)) != len(value):
        raise ValueError(f"{label} must be unique")


def _agent(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"runtime", "model"}:
        raise ValueError("agent requires runtime and model labels")
    for key in value:
        _text(value[key], key, 256)


def _request(value: Any, required: set[str], optional: set[str] = frozenset()) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) - required - optional
        or required - set(value)
    ):
        raise ValueError(
            f"request requires {sorted(required)}; optional {sorted(optional)}"
        )
    raw = canonical(value)
    if len(raw.encode()) > 1_000_000:
        raise ValueError("research request exceeds 1 MB")
    return json.loads(raw)


def research_records(connection: sqlite3.Connection) -> list[dict]:
    """Old campaigns stay byte-compatible; reading does not migrate their schema."""
    if not connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_records'"
    ).fetchone():
        return []
    records = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT record FROM research_records ORDER BY rowid"
        )
    ]
    for record in records:
        if record.get("digest") != digest(
            {k: v for k, v in record.items() if k != "digest"}
        ):
            raise ValueError("research record content hash mismatch")
    return records


def _source(spec: dict) -> dict:
    return {"campaign_id": spec["id"], "manifest_digest": digest(spec)}


def _events(connection: sqlite3.Connection) -> list[dict]:
    return [
        {**dict(row), "payload": json.loads(row["payload"])}
        for row in connection.execute(
            "SELECT sequence,kind,experiment_id,payload,created_at FROM events ORDER BY sequence"
        )
    ]


def _evidence(record: dict, events: list[dict] = ()) -> dict:
    # No artifact paths, raw outputs, PID, or source file bodies leave the study.
    # Failure strings are local diagnostic labels, not copied stdout/stderr.
    value = {
        "experiment_id": record["id"],
        "config_digest": record["config_hash"],
        "attempt_token": record["token"],
        "status": record["status"],
        "metrics": record["metrics"],
        "artifact_digests": [a["sha256"] for a in record["artifacts"]],
        "failure": "execution_failed" if record["status"] == "failed" else None,
        "execution_events": [
            {
                "sequence": event["sequence"],
                "kind": event["kind"],
                "recorded_at": event["created_at"],
                "event_digest": digest(event),
            }
            for event in events
            if event["experiment_id"] == record["id"]
            and event["kind"]
            in {
                "attempt_reserved",
                "worker_started",
                "attempt_finished",
                "worker_spawn_failed",
                "attempt_unresolved",
                "scheduler_submitted",
                "scheduler_observed",
                "scheduler_cancel_requested",
            }
            and event["payload"].get("token", record["token"]) == record["token"]
        ],
    }
    return {**value, "evidence_digest": digest(value)}


class ResearchJournal:
    def __init__(self, store: CampaignStore):
        self.store = store

    @staticmethod
    def _previous(
        connection: sqlite3.Connection, request: dict, kind: str
    ) -> dict | None:
        records = research_records(connection)
        if not any(r["id"] == request["id"] for r in records):
            return None
        row = connection.execute(
            "SELECT kind,request,record FROM research_records WHERE id=?",
            (request["id"],),
        ).fetchone()
        if row["kind"] != kind or row["request"] != canonical(request):
            raise ValueError("research id reused with different contents")
        return json.loads(row["record"])

    def _append(
        self, connection: sqlite3.Connection, kind: str, request: dict, payload: dict
    ) -> dict:
        connection.execute("""CREATE TABLE IF NOT EXISTS research_records (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, request TEXT NOT NULL, record TEXT NOT NULL
        )""")
        connection.execute("""CREATE TRIGGER IF NOT EXISTS research_records_no_update
            BEFORE UPDATE ON research_records BEGIN SELECT RAISE(ABORT,'research records are immutable'); END""")
        connection.execute("""CREATE TRIGGER IF NOT EXISTS research_records_no_delete
            BEFORE DELETE ON research_records BEGIN SELECT RAISE(ABORT,'research records are immutable'); END""")
        body = {
            "id": request["id"],
            "kind": kind,
            "recorded_at": now(),
            "payload": payload,
        }
        record = {**body, "digest": digest(body)}
        connection.execute(
            "INSERT INTO research_records VALUES (?,?,?,?)",
            (
                request["id"],
                kind,
                canonical(request),
                canonical(record),
            ),
        )
        self.store.event(
            connection,
            "research_recorded",
            None,
            {
                "record_id": record["id"],
                "record_kind": kind,
                "record_digest": record["digest"],
            },
        )
        return record

    def record_assessment(self, request: dict) -> dict:
        request = _request(
            request,
            {
                "id",
                "hypothesis",
                "outcome",
                "interpretation",
                "limitations",
                "next_action",
                "experiment_ids",
                "agent",
                "source",
            },
            {"supersedes"},
        )
        request.setdefault("supersedes", None)
        _identifier(request["id"], "assessment id")
        for key in ("hypothesis", "interpretation", "next_action"):
            _text(request[key], key)
        _list(request["limitations"], "limitations")
        _list(request["experiment_ids"], "experiment_ids", identifiers=True)
        _agent(request["agent"])
        if request["outcome"] not in (
            "supports",
            "refutes",
            "inconclusive",
            "infrastructure_failure",
        ):
            raise ValueError("invalid assessment outcome")
        with self.store.transaction() as connection:
            previous = self._previous(connection, request, "assessment")
            if previous:
                return previous
            if request["source"] != _source(self.store._manifest(connection)):
                raise ValueError(
                    "assessment source manifest mismatch; use the exact intended study"
                )
            records = research_records(connection)
            supersedes = request["supersedes"]
            if supersedes is not None:
                _identifier(supersedes, "supersedes")
                if not any(
                    r["id"] == supersedes and r["kind"] == "assessment" for r in records
                ):
                    raise ValueError("supersedes requires an existing local assessment")
                if any(r["payload"].get("supersedes") == supersedes for r in records):
                    raise ValueError("assessment already superseded")
            evidence = []
            events = _events(connection)
            for experiment_id in request["experiment_ids"]:
                row = connection.execute(
                    "SELECT * FROM experiments WHERE id=?", (experiment_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("unknown assessment experiment")
                record = self.store._record(row)
                if request["outcome"] == "infrastructure_failure":
                    if record["status"] != "failed" or not record["token"]:
                        raise ValueError(
                            "infrastructure_failure requires an actual failed attempt"
                        )
                elif not self.store.verify_record(record):
                    raise ValueError(
                        "scientific assessment requires verified successful execution, not failed/unknown jobs"
                    )
                captured = _evidence(record, events)
                if not captured["execution_events"]:
                    raise ValueError(
                        "assessment requires original recorded execution events"
                    )
                evidence.append(captured)
            return self._append(
                connection,
                "assessment",
                request,
                {
                    **request,
                    "source": _source(self.store._manifest(connection)),
                    "evidence": evidence,
                },
            )

    def make_lesson(self, request: dict) -> dict:
        """Read-only export construction; revalidate exact source artifacts now."""
        request = _request(
            request,
            {
                "title",
                "claim",
                "tags",
                "applicability",
                "limitations",
                "assessment_ids",
            },
            {"replaces"},
        )
        _list(request["assessment_ids"], "assessment_ids", identifiers=True)
        with self.store.transaction() as connection:
            spec = self.store._manifest(connection)
            records = research_records(connection)
            events = _events(connection)
            by_id = {r["id"]: r for r in records if r["kind"] == "assessment"}
            superseded = {r["payload"].get("supersedes") for r in by_id.values()}
            selected = []
            for assessment_id in request["assessment_ids"]:
                if assessment_id not in by_id or assessment_id in superseded:
                    raise ValueError(
                        "lesson requires current, non-superseded assessments"
                    )
                assessment = by_id[assessment_id]
                if assessment["payload"]["source"] != _source(spec):
                    raise ValueError("assessment source manifest mismatch")
                for evidence in assessment["payload"]["evidence"]:
                    row = connection.execute(
                        "SELECT * FROM experiments WHERE id=?",
                        (evidence["experiment_id"],),
                    ).fetchone()
                    if row is None:
                        raise ValueError("source evidence missing")
                    record = self.store._record(row)
                    if canonical(_evidence(record, events)) != canonical(evidence):
                        raise ValueError("source evidence changed since assessment")
                    if record["status"] == "succeeded" and not self.store.verify_record(
                        record
                    ):
                        raise ValueError("source artifacts no longer verified")
                selected.append(assessment)
            payload = {
                k: request[k]
                for k in ("title", "claim", "tags", "applicability", "limitations")
            }
            payload.update(
                {
                    "source": {
                        **_source(spec),
                        "metric": spec["metric"],
                        "provenance": {
                            k: spec["provenance"][k]
                            for k in (
                                "code_revision",
                                "dataset_revision",
                                "environment",
                            )
                        },
                    },
                    "assessments": selected,
                    "replaces": request.get("replaces"),
                }
            )
            return validate_bundle(
                {
                    "schema_version": "loopx_hpc_lesson_v1",
                    "id": "lesson-" + digest(payload),
                    "payload": payload,
                }
            )

    def record_lesson_use(self, request: dict, library: KnowledgeLibrary) -> dict:
        request = _request(
            request,
            {
                "id",
                "lesson_id",
                "action",
                "rationale",
                "adaptation",
                "checks",
                "agent",
                "source",
            },
        )
        _identifier(request["id"], "lesson use id")
        if request["action"] not in ("adopt", "adapt", "reject"):
            raise ValueError("lesson action must be adopt, adapt, or reject")
        for key in ("rationale", "adaptation"):
            _text(request[key], key)
        _list(request["checks"], "applicability checks")
        _agent(request["agent"])
        # Hold the local library's consistent read lock until the campaign commits.
        with ExitStack() as locks, self.store.transaction() as connection:
            previous = self._previous(connection, request, "lesson_use")
            if previous:
                return previous
            if request["source"] != _source(self.store._manifest(connection)):
                raise ValueError("lesson use target manifest mismatch")
            observation = locks.enter_context(library.snapshot())
            lesson = observation.get(request["lesson_id"])
            if request["action"] != "reject" and lesson["state"] != "active":
                raise ValueError("cannot adopt a withdrawn or superseded lesson")
            return self._append(
                connection,
                "lesson_use",
                request,
                {
                    **request,
                    "source": _source(self.store._manifest(connection)),
                    "lesson_state_at_use": lesson["state"],
                    "lesson": lesson["bundle"],
                    "library_id": lesson["library_id"],
                },
            )


def validate_reasoning(
    proposal: dict, snapshot: dict, library: KnowledgeLibrary | None
) -> None:
    """Optional richer proposal contract; existing v0 proposals remain valid."""
    if "reasoning" not in proposal:
        return
    reasoning = _request(
        proposal["reasoning"],
        {
            "hypothesis",
            "prediction",
            "selection_basis",
            "alternatives",
            "uncertainty",
            "lesson_use_ids",
            "agent",
        },
    )
    for key in ("hypothesis", "prediction", "selection_basis", "uncertainty"):
        _text(reasoning[key], key)
    _agent(reasoning["agent"])
    if reasoning["agent"]["runtime"] != proposal["runtime"]:
        raise ValueError("reasoning agent runtime must match proposal runtime")
    _list(reasoning["lesson_use_ids"], "lesson_use_ids", identifiers=True, minimum=0)
    alternatives = reasoning["alternatives"]
    if not isinstance(alternatives, list) or len(alternatives) > 32:
        raise ValueError("alternatives must be a bounded list")
    seen = {canonical(config) for config in proposal["configs"]}
    for alternative in alternatives:
        alternative = _request(alternative, {"config", "reason_not_selected"})
        _validate_config(alternative["config"], snapshot["campaign"])
        _text(alternative["reason_not_selected"], "reason_not_selected")
        encoded = canonical(alternative["config"])
        if encoded in seen:
            raise ValueError(
                "alternative duplicates a selected or other alternative config"
            )
        seen.add(encoded)
    by_id = {r["id"]: r for r in snapshot.get("research_records", [])}
    for use_id in reasoning["lesson_use_ids"]:
        use = by_id.get(use_id)
        if (
            not use
            or use["kind"] != "lesson_use"
            or use["payload"]["action"] == "reject"
        ):
            raise ValueError(
                "proposal requires an explicit adopted/adapted local lesson use"
            )
        if use["payload"]["source"] != _source(snapshot["campaign"]):
            raise ValueError("lesson use target manifest mismatch")
        if library is None:
            raise ValueError(
                "a fresh explicit knowledge library is required for lesson-informed proposals"
            )
        current = library.get(use["payload"]["lesson_id"])
        if current["library_id"] != use["payload"]["library_id"]:
            raise ValueError("lesson library changed; record a new explicit adoption")
        if current["state"] != "active" or canonical(current["bundle"]) != canonical(
            use["payload"]["lesson"]
        ):
            raise ValueError(
                "lesson is withdrawn, superseded, or changed; reassess the proposal"
            )
