"""Read-only scientific lineage from recorded decisions, never inferred causality.

The graph deliberately omits commands, artifact paths, raw logs and raw failure
messages. Numerical differences are not significance tests or scientific
assessments. Missing historical reasoning stays missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
import html
import json
import math
import re
from typing import Any

from .campaign import CampaignStore, canonical, digest


SCHEMA = "loopx_hpc_evolution_v0"


def _ref(kind: str, identifier: Any) -> str:
    return f"{kind}:{identifier}"


def _safe_value(value: Any) -> Any:
    """Keep typed values; replace path-like strings with exact digest references."""
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return {
            "redacted": "path_like_text",
            "value_digest": digest(value),
            "type": "string",
        }
    if isinstance(value, Mapping):
        return {
            str(key): (
                {"redacted": "private_detail", "value_digest": digest(item)}
                if key
                in {"command", "log", "logs", "stdout", "stderr", "path", "failure"}
                else _safe_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return value


def _config_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict]:
    changes = []
    for key in sorted(set(before) | set(after)):
        if (
            key in before
            and key in after
            and canonical(before[key]) == canonical(after[key])
        ):
            continue
        changes.append(
            {
                "parameter": key,
                "before_present": key in before,
                "after_present": key in after,
                "before": _safe_value(before.get(key)),
                "after": _safe_value(after.get(key)),
                "before_type": type(before[key]).__name__ if key in before else None,
                "after_type": type(after[key]).__name__ if key in after else None,
            }
        )
    return changes


def _verified(store: CampaignStore, record: dict) -> bool:
    try:
        return bool(store.verify_record(record))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _read_snapshot(store: CampaignStore) -> tuple[dict, list[dict], dict[str, dict]]:
    # One transaction keeps event high-watermark and decision acceptance results
    # aligned with the projected experiment state, even while workers finish.
    with store.transaction() as connection:
        snapshot = {
            "campaign": store._manifest(connection),
            "experiments": [
                store._record(row)
                for row in connection.execute(
                    "SELECT * FROM experiments ORDER BY rowid"
                )
            ],
            "decisions": [],
        }
        results = {}
        for row in connection.execute(
            "SELECT id,payload,result FROM decisions ORDER BY rowid"
        ):
            snapshot["decisions"].append(json.loads(row["payload"]))
            results[row["id"]] = json.loads(row["result"])
        events = [
            {**dict(row), "payload": json.loads(row["payload"])}
            for row in connection.execute(
                "SELECT sequence,kind,experiment_id,payload,created_at FROM events ORDER BY sequence"
            )
        ]
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_records'"
        ).fetchone():
            records = [
                json.loads(row["record"])
                for row in connection.execute(
                    "SELECT record FROM research_records ORDER BY rowid"
                )
            ]
            if records:
                snapshot["research_records"] = records
    return snapshot, events, results


def _execution_outcome(record: dict, evidence_valid: bool) -> str:
    status = record["status"]
    if status == "succeeded":
        return "validated_result" if evidence_valid else "invalid_or_missing_evidence"
    if status == "failed":
        # Nonzero process exit is not, by itself, an infrastructure diagnosis or
        # a refuted scientific hypothesis. Only the known launcher failure is.
        if str(record.get("failure") or "").startswith("worker_spawn_failed:"):
            return "infrastructure_failure"
        return "execution_failure"
    return {
        "planned": "planned",
        "submitting": "submission_pending",
        "running": "in_progress",
        "unknown": "unresolved_execution",
    }.get(status, "unrecognized_execution_state")


def _numeric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _metric_delta(
    before: int | float, after: int | float
) -> tuple[int | float | None, str, Decimal]:
    # Preserve decimal values as logged in JSON, rather than displaying binary
    # float subtraction artifacts. The exact string also handles float overflow.
    with localcontext() as context:
        context.prec = max(1000, len(str(before)) + len(str(after)) + 4)
        exact = Decimal(str(after)) - Decimal(str(before))
    numeric = (
        after - before if type(before) is int and type(after) is int else float(exact)
    )
    if not math.isfinite(numeric):
        numeric = None
    return numeric, format(exact, "f"), exact


def _assessment_evidence(
    payload: dict, experiments: dict[str, dict], events: list[dict]
) -> tuple[bool, dict[str, bool]]:
    from .research import _evidence

    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False, {}
    indexed = {
        item.get("experiment_id"): item for item in evidence if isinstance(item, dict)
    }
    valid = len(indexed) == len(evidence) and set(indexed) == set(
        payload.get("experiment_ids", [])
    )
    unchanged = {}
    for identifier, item in indexed.items():
        valid = valid and item.get("evidence_digest") == digest(
            {key: value for key, value in item.items() if key != "evidence_digest"}
        )
        unchanged[identifier] = identifier in experiments and canonical(
            _evidence(experiments[identifier], events)
        ) == canonical(item)
    return valid, unchanged


def _captured_lesson(record: dict) -> tuple[list[dict], list[dict], dict]:
    """Expand only the immutable imported snapshot; never access its source."""
    from .knowledge import validate_bundle

    payload = record["payload"]
    bundle = validate_bundle(payload.get("lesson"))
    if payload.get("lesson_id") != bundle["id"]:
        raise ValueError("lesson use and captured bundle identity differ")
    lesson = bundle["payload"]
    namespace = {
        key: lesson["source"][key] for key in ("campaign_id", "manifest_digest")
    }
    capture = {
        "snapshot_only": True,
        "content_integrity_verified": True,
        "source_accessed": False,
        "current_source_artifacts": "not_checked",
        "scientific_truth_verified": False,
    }

    def captured_ref(kind: str, *identity: str) -> str:
        # Hash the tuple, not concatenated identifiers: names may contain colons.
        return (
            f"captured:{namespace['manifest_digest']}:{kind}:{digest(list(identity))}"
        )

    lesson_ref = captured_ref("lesson", bundle["id"])
    nodes = [
        {
            "id": lesson_ref,
            "kind": "imported_lesson",
            "lesson_id": bundle["id"],
            "payload": _safe_value(
                {key: value for key, value in lesson.items() if key != "assessments"}
            ),
            "source": {
                **namespace,
                "lesson_id": bundle["id"],
                "content_digest": digest(lesson),
            },
            "capture": capture,
        }
    ]
    edges = []
    for assessment in lesson["assessments"]:
        assessment_payload = assessment["payload"]
        assessment_ref = captured_ref(
            "assessment", assessment["id"], assessment["digest"]
        )
        evidence_refs = []
        for evidence in assessment_payload["evidence"]:
            evidence_ref = captured_ref(
                "experiment",
                evidence["experiment_id"],
                evidence["attempt_token"],
                evidence["evidence_digest"],
            )
            evidence_refs.append(evidence_ref)
            nodes.append(
                {
                    "id": evidence_ref,
                    "kind": "captured_experiment",
                    "experiment_id": evidence["experiment_id"],
                    "attempt_token": evidence["attempt_token"],
                    "captured_status": evidence["status"],
                    "captured_metrics": evidence["metrics"],
                    "config_digest": evidence["config_digest"],
                    "configuration": "not_captured",
                    "artifact_digests": evidence["artifact_digests"],
                    "execution_events": evidence["execution_events"],
                    "source": {
                        **namespace,
                        "experiment_id": evidence["experiment_id"],
                        "attempt_token": evidence["attempt_token"],
                        "evidence_digest": evidence["evidence_digest"],
                    },
                    "capture": capture,
                }
            )
            edges.append(
                {
                    "source": evidence_ref,
                    "target": assessment_ref,
                    "kind": "captured_assessed_evidence",
                    "provenance": {
                        **namespace,
                        "lesson_id": bundle["id"],
                        "assessment_digest": assessment["digest"],
                        "evidence_digest": evidence["evidence_digest"],
                    },
                }
            )
        nodes.append(
            {
                "id": assessment_ref,
                "kind": "captured_assessment",
                "record_id": assessment["id"],
                "payload": _safe_value(
                    {
                        key: value
                        for key, value in assessment_payload.items()
                        if key != "evidence"
                    }
                ),
                "evidence_refs": evidence_refs,
                "recorded_at": assessment["recorded_at"],
                "source": {
                    **namespace,
                    "assessment_id": assessment["id"],
                    "record_digest": assessment["digest"],
                },
                "capture": capture,
            }
        )
        edges.append(
            {
                "source": assessment_ref,
                "target": lesson_ref,
                "kind": "captured_assessment_in_lesson",
                "provenance": {
                    **namespace,
                    "lesson_id": bundle["id"],
                    "assessment_digest": assessment["digest"],
                },
            }
        )
    edges.append(
        {
            "source": lesson_ref,
            "target": _ref("lesson_use", record["id"]),
            "kind": "imported_for_lesson_use",
            "provenance": {
                "table": "research_records",
                "id": record["id"],
                "digest": record["digest"],
                "source": namespace,
                "target": payload["source"],
                "lesson_id": bundle["id"],
                "action": payload.get("action"),
                "library_id": payload.get("library_id"),
                "lesson_state_at_use": payload.get("lesson_state_at_use"),
                "recorded_at": record["recorded_at"],
                "current_library_state": "not_checked",
            },
        }
    )
    return (
        nodes,
        edges,
        {
            "id": bundle["id"],
            "node_ref": lesson_ref,
            "content_digest": digest(lesson),
            "snapshot_only": True,
            "content_integrity_verified": True,
        },
    )


def build_evolution(store: CampaignStore) -> dict:
    """Project exact recorded lineage and comparisons within one frozen study."""
    snapshot, events, accepted = _read_snapshot(store)
    manifest = snapshot["campaign"]
    manifest_digest = digest(manifest)
    source = {
        "campaign_id": manifest["id"],
        "manifest_digest": manifest_digest,
        "snapshot_digest": digest(snapshot),
        "event_high_watermark": events[-1]["sequence"] if events else 0,
    }
    experiments = {record["id"]: record for record in snapshot["experiments"]}
    verified = {key: _verified(store, row) for key, row in experiments.items()}
    nodes, edges, comparisons, warnings = [], [], [], []
    edge_keys = set()

    def edge(start: str, end: str, kind: str, provenance: dict) -> None:
        key = (start, end, kind, canonical(provenance))
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append(
                {"source": start, "target": end, "kind": kind, "provenance": provenance}
            )

    for identifier, row in experiments.items():
        config_valid = digest(row["config"]) == row["config_hash"]
        nodes.append(
            {
                "id": _ref("experiment", identifier),
                "kind": "experiment",
                "experiment_id": identifier,
                "status": row["status"],
                "execution_outcome": _execution_outcome(row, verified[identifier]),
                "scientific_assessments": [],
                "config": _safe_value(row["config"]),
                "config_hash": row["config_hash"],
                "config_hash_valid": config_valid,
                "evidence_valid": verified[identifier],
                "metrics": row["metrics"],
                "attempt_ref": _ref("attempt", row["token"])
                if row.get("token")
                else None,
                "artifact_digests": [a.get("sha256") for a in row.get("artifacts", [])],
                "rationale": _safe_value(row["rationale"]),
                "baseline_config_diff": _config_diff(
                    manifest["baseline"], row["config"]
                ),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "source": {
                    "table": "experiments",
                    "id": identifier,
                    "record_digest": digest(row),
                },
            }
        )
        for parent_id in dict.fromkeys(row.get("parents", [])):
            if parent_id not in experiments:
                warnings.append(
                    {
                        "kind": "missing_parent",
                        "experiment_id": identifier,
                        "parent_id": parent_id,
                    }
                )
                continue
            edge(
                _ref("experiment", parent_id),
                _ref("experiment", identifier),
                "recorded_parent",
                {
                    "table": "experiments",
                    "id": identifier,
                    "record_digest": digest(row),
                },
            )
            parent = experiments[parent_id]
            metric_name = manifest["metric"]["name"]
            left, right = (
                parent["metrics"].get(metric_name),
                row["metrics"].get(metric_name),
            )
            comparable = (
                verified[parent_id]
                and verified[identifier]
                and _numeric(left)
                and _numeric(right)
            )
            delta, delta_exact, exact_value = (
                _metric_delta(left, right) if comparable else (None, None, None)
            )
            comparisons.append(
                {
                    "source": _ref("experiment", parent_id),
                    "target": _ref("experiment", identifier),
                    "manifest_digest": manifest_digest,
                    "config_diff": _config_diff(parent["config"], row["config"]),
                    "metric": metric_name,
                    "direction": manifest["metric"]["direction"],
                    "comparable": comparable,
                    "parent_value": left if comparable else None,
                    "next_value": right if comparable else None,
                    "delta": delta,
                    "delta_exact": delta_exact,
                    "numerically_improved": (
                        (exact_value < 0)
                        if manifest["metric"]["direction"] == "minimize"
                        else (exact_value > 0)
                    )
                    if comparable
                    else None,
                    "interpretation": "numerical_difference_only"
                    if comparable
                    else "requires_two_verified_results",
                }
            )

    experiment_nodes = {node["experiment_id"]: node for node in nodes}
    for decision in snapshot["decisions"]:
        decision_id = decision["id"]
        ref = _ref("decision", decision_id)
        provenance = {
            "table": "decisions",
            "id": decision_id,
            "payload_digest": digest(decision),
        }
        result = accepted.get(decision_id)
        output_ids = (
            list(dict.fromkeys(result.get("experiment_ids", []))) if result else []
        )
        nodes.append(
            {
                "id": ref,
                "kind": "decision",
                "decision_id": decision_id,
                "runtime": decision.get("runtime"),
                "rationale": _safe_value(decision.get("rationale")),
                "context_digest": decision.get("context_digest"),
                "reasoning": _safe_value(decision.get("reasoning")),
                "reasoning_status": "recorded"
                if decision.get("reasoning") is not None
                else "not_recorded",
                "experiment_ids": output_ids,
                "source": {
                    **provenance,
                    "acceptance_digest": digest(result) if result else None,
                },
            }
        )
        for experiment_id in dict.fromkeys(decision.get("evidence_ids", [])):
            if experiment_id in experiments:
                edge(
                    _ref("experiment", experiment_id), ref, "cited_evidence", provenance
                )
            else:
                warnings.append(
                    {
                        "kind": "missing_decision_evidence",
                        "decision_id": decision_id,
                        "experiment_id": experiment_id,
                    }
                )
        for experiment_id in output_ids:
            if experiment_id in experiments:
                edge(
                    ref,
                    _ref("experiment", experiment_id),
                    "accepted_configuration",
                    {
                        **provenance,
                        "acceptance_digest": digest(result),
                    },
                )
            else:
                warnings.append(
                    {
                        "kind": "missing_accepted_experiment",
                        "decision_id": decision_id,
                        "experiment_id": experiment_id,
                    }
                )

    valid_research_refs = set()
    captured_nodes = {}
    superseded_by: dict[str, list[str]] = {}
    for record in snapshot.get("research_records", []):
        payload = record["payload"]
        ref = _ref(record["kind"], record["id"])
        digest_valid = record.get("digest") == digest(
            {key: value for key, value in record.items() if key != "digest"}
        )
        namespace_valid = payload.get("source") == {
            "campaign_id": manifest["id"],
            "manifest_digest": manifest_digest,
        }
        if digest_valid and namespace_valid:
            valid_research_refs.add(ref)
        else:
            warnings.append(
                {
                    "kind": "invalid_research_record_source",
                    "record_id": record["id"],
                    "digest_valid": digest_valid,
                    "namespace_valid": namespace_valid,
                }
            )
        research_node = {
            "id": ref,
            "kind": record["kind"],
            "record_id": record["id"],
            "payload": _safe_value(payload),
            "recorded_at": record["recorded_at"],
            "source": {
                "table": "research_records",
                "id": record["id"],
                "digest": record["digest"],
                "digest_valid": digest_valid,
                "namespace_valid": namespace_valid,
            },
        }
        if record["kind"] == "lesson_use":
            # Keep one content-addressed copy of source history, not the entire
            # nested bundle repeated for each use. Invalid snapshots stay inert.
            research_node["payload"]["lesson"] = {
                "id": payload.get("lesson_id"),
                "bundle_digest": digest(payload.get("lesson")),
                "content_integrity_verified": False,
            }
            if ref in valid_research_refs:
                try:
                    imported, links, reference = _captured_lesson(record)
                except (ValueError, TypeError, KeyError):
                    warnings.append(
                        {"kind": "invalid_captured_lesson", "record_id": record["id"]}
                    )
                else:
                    research_node["payload"]["lesson"] = reference
                    for node in imported:
                        captured_nodes.setdefault(node["id"], node)
                    for link in links:
                        edge(
                            link["source"],
                            link["target"],
                            link["kind"],
                            link["provenance"],
                        )
        nodes.append(research_node)
        if record["kind"] == "assessment" and ref in valid_research_refs:
            recorded_evidence_valid, evidence_unchanged = _assessment_evidence(
                payload, experiments, events
            )
            nodes[-1]["source"]["recorded_evidence_valid"] = recorded_evidence_valid
            if not recorded_evidence_valid:
                warnings.append(
                    {
                        "kind": "invalid_or_missing_assessment_evidence",
                        "record_id": record["id"],
                    }
                )
            for experiment_id, unchanged in evidence_unchanged.items():
                if not unchanged:
                    warnings.append(
                        {
                            "kind": "assessment_evidence_changed",
                            "record_id": record["id"],
                            "experiment_id": experiment_id,
                        }
                    )
            evidence_ids = payload.get(
                "experiment_ids", payload.get("evidence_ids", [])
            )
            if payload.get("experiment_id"):
                evidence_ids = [payload["experiment_id"], *evidence_ids]
            for experiment_id in dict.fromkeys(evidence_ids):
                if experiment_id not in experiments:
                    warnings.append(
                        {
                            "kind": "missing_assessment_evidence",
                            "record_id": record["id"],
                            "experiment_id": experiment_id,
                        }
                    )
                    continue
                experiment_nodes[experiment_id]["scientific_assessments"].append(
                    {
                        "ref": ref,
                        "outcome": payload.get("outcome"),
                        "recorded_at": record["recorded_at"],
                        "record_digest": record["digest"],
                        "recorded_evidence_valid": recorded_evidence_valid,
                        "evidence_unchanged": evidence_unchanged.get(
                            experiment_id, False
                        ),
                        "current_evidence_valid": verified[experiment_id],
                    }
                )
                edge(
                    _ref("experiment", experiment_id),
                    ref,
                    "assessed_evidence",
                    {
                        "table": "research_records",
                        "id": record["id"],
                        "digest": record["digest"],
                        "recorded_evidence_valid": recorded_evidence_valid,
                        "evidence_unchanged": evidence_unchanged.get(
                            experiment_id, False
                        ),
                    },
                )

    nodes.extend(captured_nodes.values())

    for decision in snapshot["decisions"]:
        for record_id in dict.fromkeys(
            (decision.get("reasoning") or {}).get("lesson_use_ids", [])
        ):
            ref = _ref("lesson_use", record_id)
            if ref in valid_research_refs:
                edge(
                    ref,
                    _ref("decision", decision["id"]),
                    "cited_lesson_use",
                    {
                        "table": "decisions",
                        "id": decision["id"],
                        "payload_digest": digest(decision),
                    },
                )
            else:
                warnings.append(
                    {
                        "kind": "missing_or_invalid_lesson_use",
                        "decision_id": decision["id"],
                        "record_id": record_id,
                    }
                )

    for record in snapshot.get("research_records", []):
        ref = _ref(record["kind"], record["id"])
        supersedes = record["payload"].get("supersedes")
        if supersedes and ref in valid_research_refs:
            previous = _ref(record["kind"], supersedes)
            if previous in valid_research_refs:
                superseded_by.setdefault(previous, []).append(ref)
                edge(
                    previous,
                    ref,
                    "explicitly_superseded_by",
                    {
                        "table": "research_records",
                        "id": record["id"],
                        "digest": record["digest"],
                    },
                )
            else:
                warnings.append(
                    {
                        "kind": "missing_superseded_record",
                        "record_id": record["id"],
                        "supersedes": supersedes,
                    }
                )

    for node in nodes:
        if node["kind"] == "assessment":
            node["superseded_by"] = superseded_by.get(node["id"], [])
        if node["kind"] == "experiment":
            for assessment in node["scientific_assessments"]:
                assessment["superseded_by"] = superseded_by.get(assessment["ref"], [])

    timeline = [
        {
            "id": _ref("event", event["sequence"]),
            "sequence": event["sequence"],
            "kind": event["kind"],
            "recorded_at": event["created_at"],
            "experiment_ref": _ref("experiment", event["experiment_id"])
            if event["experiment_id"]
            else None,
            "decision_ref": _ref("decision", event["payload"]["proposal_id"])
            if event["kind"] == "proposal_accepted"
            and "proposal_id" in event["payload"]
            else None,
            "research_ref": _ref(
                event["payload"]["record_kind"], event["payload"]["record_id"]
            )
            if event["kind"] == "research_recorded"
            and "record_id" in event["payload"]
            and "record_kind" in event["payload"]
            else None,
            "source_digest": digest(event),
        }
        for event in events
    ]
    projection = {
        "schema_version": SCHEMA,
        "source": source,
        "study": {
            "objective": _safe_value(manifest["objective"]),
            "hypothesis": _safe_value(manifest["hypothesis"]),
            "metric": manifest["metric"],
            "provenance": _safe_value(manifest["provenance"]),
        },
        "nodes": nodes,
        "edges": edges,
        "comparisons": comparisons,
        "timeline": timeline,
        "warnings": warnings,
        "boundaries": {
            "read_only": True,
            "causality": "recorded_links_only",
            "metric_deltas": "same_frozen_study_verified_results_only",
            "scientific_verdicts": "recorded_assessments_only",
            "imported_history": "validated_captured_snapshots_only",
            "current_imported_source_artifacts": "not_checked",
            "current_library_state": "not_checked",
        },
    }
    projection["projection_digest"] = digest(projection)
    return projection


def _md(value: Any) -> str:
    text = value if isinstance(value, str) else canonical(value)
    text = " ".join(text.splitlines())
    text = html.escape(text, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


def _field_lines(
    payload: Mapping[str, Any], fields: tuple[tuple[str, str], ...]
) -> list[str]:
    lines = []
    for key, title in fields:
        value = payload.get(key)
        if isinstance(value, list):
            lines.append(
                f"- {title}: "
                + ("; ".join(_md(item) for item in value) or "none recorded")
            )
        else:
            lines.append(
                f"- {title}: {_md(value) if value is not None else 'not recorded'}"
            )
    return lines


def _assessment_lines(payload: Mapping[str, Any]) -> list[str]:
    return _field_lines(
        payload,
        (
            ("outcome", "Outcome"),
            ("hypothesis", "Hypothesis"),
            ("interpretation", "Interpretation"),
            ("limitations", "Limitations"),
            ("next_action", "Next action"),
        ),
    )


def render_evolution(projection: Mapping[str, Any]) -> str:
    """Render plain, escaped Markdown; no raw HTML, links, diagrams, or commands."""
    source = projection["source"]
    nodes_by_ref = {node["id"]: node for node in projection["nodes"]}

    def label(reference: str) -> str:
        node = nodes_by_ref.get(reference, {})
        if node.get("kind") == "imported_lesson":
            return "lesson " + node["lesson_id"][:19]
        if node.get("kind") in ("captured_experiment", "captured_assessment"):
            origin = node["source"]
            identifier = node.get("experiment_id", node.get("record_id"))
            return f"captured {origin['campaign_id']}@{origin['manifest_digest'][:12]} {identifier}"
        return reference

    lines = [
        f"# Experiment evolution: {_md(source['campaign_id'])}",
        "",
        f"Study digest: {_md(source['manifest_digest'])}",
        "",
        f"Objective: {_md(projection['study']['objective'])}",
        "",
        f"Study hypothesis: {_md(projection['study']['hypothesis'])}",
        "",
        "Execution status and recorded scientific assessments are separate. Metric deltas are numerical comparisons, not significance tests.",
        "",
        "## Experiments",
        "",
        "| Experiment | Execution | Evidence valid | Config | Metrics | Scientific assessments |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for node in projection["nodes"]:
        if node["kind"] == "experiment":
            verdicts = [
                str(item["outcome"])
                + (" (superseded)" if item.get("superseded_by") else "")
                + (
                    " (evidence changed or unverified)"
                    if not item["recorded_evidence_valid"]
                    or not item["evidence_unchanged"]
                    else ""
                )
                for item in node["scientific_assessments"]
            ]
            cells = [
                node["experiment_id"],
                node["execution_outcome"],
                node["evidence_valid"],
                node["config"],
                node["metrics"],
                verdicts or "not recorded",
            ]
            lines.append("| " + " | ".join(_md(value) for value in cells) + " |")
    lines.extend(["", "## Decisions and recorded reasoning", ""])
    decisions = [node for node in projection["nodes"] if node["kind"] == "decision"]
    if not decisions:
        lines.append("No accepted decisions are recorded.")
    for node in decisions:
        lines.extend(
            [
                f"### {_md(node['decision_id'])}",
                "",
                f"Rationale: {_md(node['rationale'])}",
                "",
                f"Accepted experiment IDs: {_md(node['experiment_ids'])}",
                "",
            ]
        )
        reasoning = node["reasoning"]
        if reasoning is None:
            lines.extend(["Reasoning: not recorded (legacy decision)", ""])
        else:
            lines.extend(
                _field_lines(
                    reasoning,
                    (
                        ("hypothesis", "Hypothesis"),
                        ("prediction", "Prediction"),
                        ("selection_basis", "Selection basis"),
                        ("uncertainty", "Uncertainty"),
                        ("lesson_use_ids", "Cited lesson uses"),
                    ),
                )
            )
            for alternative in reasoning.get("alternatives", []):
                lines.append(
                    f"- Alternative {_md(alternative['config'])}: {_md(alternative['reason_not_selected'])}"
                )
            lines.append("")
        lines.extend([f"Source digest: {_md(node['source']['payload_digest'])}", ""])
    lines.extend(["## Recorded scientific assessments", ""])
    assessments = [node for node in projection["nodes"] if node["kind"] == "assessment"]
    if not assessments:
        lines.append(
            "No scientific assessment is recorded; execution success is not hypothesis support."
        )
    for node in assessments:
        lines.extend(
            [
                f"### {_md(node['record_id'])}",
                "",
                *_assessment_lines(node["payload"]),
                f"- Recorded at: {_md(node['recorded_at'])}",
                f"- Superseded by: {_md(node['superseded_by'] or 'none recorded')}",
            ]
        )
        for evidence in node["payload"].get("evidence", []):
            lines.append(
                f"- Evidence: {_md(evidence.get('experiment_id'))}; metrics {_md(evidence.get('metrics'))}; evidence digest {_md(evidence.get('evidence_digest'))}"
            )
        lines.extend(
            [
                f"- Source checks: record digest {_md(node['source']['digest_valid'])}; namespace {_md(node['source']['namespace_valid'])}; captured evidence {_md(node['source'].get('recorded_evidence_valid', False))}",
                "",
            ]
        )
    lessons = [node for node in projection["nodes"] if node["kind"] == "lesson_use"]
    if lessons:
        lines.extend(["## Recorded lesson use", ""])
        for node in lessons:
            payload = node["payload"]
            imported = nodes_by_ref.get(payload.get("lesson", {}).get("node_ref"), {})
            lesson = imported.get("payload", {})
            origin = imported.get("source", {})
            lines.extend(
                [
                    f"### {_md(node['record_id'])}",
                    "",
                    *_field_lines(lesson, (("title", "Lesson"), ("claim", "Claim"))),
                    *_field_lines(
                        payload,
                        (
                            ("action", "Action"),
                            ("rationale", "Rationale"),
                            ("adaptation", "Adaptation"),
                            ("checks", "Applicability checks"),
                            ("lesson_id", "Lesson content ID"),
                            ("library_id", "Library ID"),
                            ("lesson_state_at_use", "Historical state at use"),
                        ),
                    ),
                    f"- Captured source: {_md(origin.get('campaign_id', 'not validated'))}; manifest {_md(origin.get('manifest_digest', 'not validated'))}",
                    f"- Use recorded at: {_md(node['recorded_at'])}; current library state not checked.",
                    "",
                ]
            )
    captured = [
        node
        for node in projection["nodes"]
        if node["kind"] in ("captured_assessment", "captured_experiment")
    ]
    if captured:
        lines.extend(
            [
                "## Captured source history (snapshot only)",
                "",
                "Imported content hashes are validated. Current source artifacts were not accessed or verified; captured scientific claims are not independently established. Source configurations and decision reasoning are not reconstructed from hashes.",
                "",
            ]
        )
        for node in captured:
            lines.extend([f"### {_md(label(node['id']))}", ""])
            if node["kind"] == "captured_assessment":
                lines.extend(_assessment_lines(node["payload"]))
                lines.extend(
                    [
                        f"- Recorded at source: {_md(node['recorded_at'])}",
                        f"- Captured evidence: {'; '.join(_md(label(ref)) for ref in node['evidence_refs'])}",
                        f"- Source record digest: {_md(node['source']['record_digest'])}",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"- Captured execution: {_md(node['captured_status'])}; metrics {_md(node['captured_metrics'])}",
                        f"- Attempt token: {_md(node['attempt_token'])}",
                        f"- Config digest: {_md(node['config_digest'])}; values not captured.",
                    ]
                )
                for event in node["execution_events"]:
                    lines.append(
                        f"- Source event {_md(event['sequence'])}: {_md(event['recorded_at'])}, {_md(event['kind'])}"
                    )
            lines.append("")
    lines.extend(["## Recorded lineage", ""])
    if captured:
        lines.extend(
            [
                "Captured labels abbreviate hashes; exact namespaced IDs and provenance are preserved in the JSON graph.",
                "",
            ]
        )
    for edge in projection["edges"]:
        lines.append(
            f"- {_md(label(edge['source']))} → {_md(label(edge['target']))}: {_md(edge['kind'])}"
        )
    if not projection["edges"]:
        lines.append("No causal links are recorded.")
    lines.extend(["", "## Comparable metric changes", ""])
    for comparison in projection["comparisons"]:
        value = (
            comparison["delta_exact"]
            if comparison["comparable"]
            else "not comparable: evidence incomplete or invalid"
        )
        lines.append(
            f"- {_md(comparison['source'])} → {_md(comparison['target'])}: {_md(comparison['metric'])} delta {_md(value)}; config changes {_md(comparison['config_diff'])}"
        )
    if not projection["comparisons"]:
        lines.append("No parent-linked comparisons are recorded.")
    lines.extend(["", "## Event timeline", ""])
    for event in projection["timeline"]:
        lines.append(
            f"- {_md(event['recorded_at'])}: {_md(event['kind'])} ({_md(event['experiment_ref'] or event['decision_ref'] or event.get('research_ref') or event['id'])})"
        )
    if projection["warnings"]:
        lines.extend(
            [
                "",
                "## Source warnings",
                "",
                *[f"- {_md(item)}" for item in projection["warnings"]],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
