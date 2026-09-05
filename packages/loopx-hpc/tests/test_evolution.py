"""Read-model semantics: recorded causal edges, exact types, and evidence gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loopx_hpc.campaign import CampaignStore, canonical, digest, now
from loopx_hpc.cli import demo_spec
from loopx_hpc.evolution import build_evolution, render_evolution


def _store(tmp_path: Path) -> CampaignStore:
    store = CampaignStore(tmp_path / "campaign")
    spec = demo_spec()
    spec["limits"]["max_experiments"] = 10
    store.initialize(spec)
    return store


def _finish(store: CampaignStore, experiment_id: str, value: int | float) -> None:
    """Supply real checksum-verifiable fixture artifacts, not a verifier mock."""
    record = store.get_experiment(experiment_id)
    directory = store.root / "evidence" / experiment_id
    directory.mkdir(parents=True)
    artifacts = []
    for name, payload in (
        ("config.json", record["config"]),
        ("result.json", {"metrics": {"loss": value}}),
    ):
        path = directory / name
        path.write_text(canonical(payload))
        artifacts.append(
            {
                "path": str(path.relative_to(store.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    with store.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET status='succeeded',metrics=?,artifacts=?,token=?,updated_at=? WHERE id=?",
            (
                canonical({"loss": value}),
                canonical(artifacts),
                "fixture-attempt-" + experiment_id,
                now(),
                experiment_id,
            ),
        )
        store.event(
            connection,
            "attempt_finished",
            experiment_id,
            {
                "status": "succeeded",
                "metrics": {"loss": value},
                "token": "fixture-attempt-" + experiment_id,
            },
        )
    assert store.verify_record(store.get_experiment(experiment_id))


def _lineage(store: CampaignStore) -> tuple[str, str, dict]:
    parent = store.add_experiment({"x": 0}, rationale="Measure the baseline")
    _finish(store, parent["id"], 4)
    proposal = {
        "id": "decision-1",
        "context_digest": store.context()["context_digest"],
        "runtime": "test-runner",
        "rationale": "Use measured baseline evidence",
        "evidence_ids": [parent["id"]],
        "configs": [{"x": 1}],
    }
    result = store.apply_proposal(proposal)
    child_id = result["experiment_ids"][0]
    _finish(store, child_id, 1)
    return parent["id"], child_id, proposal


def _research_record(
    store: CampaignStore, identifier: str, kind: str, payload: dict
) -> dict:
    # Direct fixture insertion also exercises reading historical data independent
    # of new record command validation, without altering the production schema.
    if kind == "assessment":
        from loopx_hpc.research import _evidence

        with store.transaction() as connection:
            events = [
                {**dict(row), "payload": json.loads(row["payload"])}
                for row in connection.execute("SELECT * FROM events ORDER BY sequence")
            ]
        payload = {
            **payload,
            "evidence": [
                _evidence(store.get_experiment(key), events)
                for key in payload["experiment_ids"]
            ],
        }
    record = {
        "id": identifier,
        "kind": kind,
        "recorded_at": "2026-01-02T00:00:00+00:00",
        "payload": payload,
    }
    record["digest"] = digest(record)
    with store.transaction() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS research_records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, request TEXT NOT NULL, record TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO research_records VALUES (?,?,?,?)",
            (identifier, kind, canonical(payload), canonical(record)),
        )
        store.event(
            connection,
            "research_record_added",
            None,
            {"record_id": identifier, "kind": kind},
        )
    return record


def _namespace(store: CampaignStore) -> dict:
    manifest = store.manifest()
    return {"campaign_id": manifest["id"], "manifest_digest": digest(manifest)}


def test_legacy_decision_exact_lineage_deltas_and_deterministic_read_only(tmp_path):
    store = _store(tmp_path)
    parent_id, child_id, proposal = _lineage(store)
    assert store.apply_proposal(proposal)["experiment_ids"] == [child_id]
    before = store.database.read_bytes()
    projection = build_evolution(store)
    assert store.database.read_bytes() == before
    assert build_evolution(store) == projection
    assert projection["source"]["snapshot_digest"] == digest(store.status())
    experiments = [node for node in projection["nodes"] if node["kind"] == "experiment"]
    assert len(experiments) == 2
    decision = next(node for node in projection["nodes"] if node["kind"] == "decision")
    assert decision["reasoning"] is None
    assert decision["reasoning_status"] == "not_recorded"
    assert {edge["kind"] for edge in projection["edges"]} == {
        "recorded_parent",
        "cited_evidence",
        "accepted_configuration",
    }
    comparison = projection["comparisons"][0]
    assert comparison["source"] == f"experiment:{parent_id}"
    assert comparison["target"] == f"experiment:{child_id}"
    assert comparison["delta"] == -3
    assert comparison["numerically_improved"] is True
    assert comparison["interpretation"] == "numerical_difference_only"
    assert comparison["config_diff"] == [
        {
            "parameter": "x",
            "before_present": True,
            "after_present": True,
            "before": 0,
            "after": 1,
            "before_type": "int",
            "after_type": "int",
        }
    ]
    assert all(node["scientific_assessments"] == [] for node in experiments)
    assert any(
        event["decision_ref"] == "decision:decision-1"
        for event in projection["timeline"]
    )
    assert str(store.root) not in canonical(projection)
    assert "evidence/" not in canonical(projection)


def test_missing_or_tampered_evidence_blocks_numeric_comparison(tmp_path):
    store = _store(tmp_path)
    _, child_id, _ = _lineage(store)
    record = store.get_experiment(child_id)
    artifact = store.root / record["artifacts"][-1]["path"]
    artifact.write_text('{"metrics":{"loss":-999}}')
    projection = build_evolution(store)
    node = next(
        node for node in projection["nodes"] if node["id"] == f"experiment:{child_id}"
    )
    assert node["execution_outcome"] == "invalid_or_missing_evidence"
    assert node["metrics"] == {"loss": 1}  # Stored result, not the tampered file.
    assert projection["comparisons"][0]["delta"] is None
    assert projection["comparisons"][0]["comparable"] is False
    artifact.unlink()
    assert build_evolution(store)["comparisons"][0]["delta"] is None


def test_type_sensitive_configs_and_stable_deduplication(tmp_path):
    store = CampaignStore(tmp_path)
    spec = demo_spec()
    spec["search_space"] = {"x": [True, 1, 1.0, "1"]}
    spec["baseline"] = {"x": 1}
    spec["limits"]["max_experiments"] = 4
    store.initialize(spec)
    for value in (True, 1, 1.0, "1"):
        store.add_experiment({"x": value}, rationale="Type-sensitive candidate")
    store.add_experiment({"x": 1}, rationale="Type-sensitive candidate")
    projection = build_evolution(store)
    experiments = [node for node in projection["nodes"] if node["kind"] == "experiment"]
    assert len(experiments) == 4
    assert len({node["config_hash"] for node in experiments}) == 4
    int_node = next(node for node in experiments if type(node["config"]["x"]) is int)
    assert int_node["baseline_config_diff"] == []
    other_types = {
        node["baseline_config_diff"][0]["after_type"]
        for node in experiments
        if node is not int_node
    }
    assert other_types == {"bool", "float", "str"}


def test_reasoning_assessments_and_lesson_use_are_recorded_not_inferred(tmp_path):
    store = _store(tmp_path)
    parent_id, child_id, proposal = _lineage(store)
    source = _namespace(store)
    _research_record(
        store,
        "assessment-1",
        "assessment",
        {
            "id": "assessment-1",
            "source": source,
            "experiment_ids": [child_id],
            "hypothesis": "A lower x increases loss",
            "outcome": "refutes",
            "interpretation": "The preregistered directional prediction was contradicted",
            "limitations": "A deterministic fixture is not a general study",
            "next_action": "Test another point",
            "agent": {"runtime": "claude-code", "model": "test-model"},
            "supersedes": None,
        },
    )
    _research_record(
        store,
        "use-1",
        "lesson_use",
        {
            "id": "use-1",
            "source": source,
            "lesson_id": "lesson-1",
            "action": "adapt",
            "rationale": "Match the current objective",
            "adaptation": "Use the current metric",
            "checks": ["Same study"],
            "agent": {"runtime": "pi", "model": "test-model"},
            "lesson_state_at_use": "accepted",
            "lesson": {"digest": "a" * 64},
        },
    )
    reasoning = {
        "hypothesis": "An interior point reduces the objective",
        "prediction": "Loss below baseline",
        "selection_basis": "Compare the recorded baseline",
        "alternatives": [
            {"config": {"x": 2}, "reason_not_selected": "Reserve for follow-up"}
        ],
        "uncertainty": "No stochastic uncertainty estimated",
        "lesson_use_ids": ["use-1"],
        "agent": {"runtime": "codex", "model": "test-model"},
    }
    with store.transaction() as connection:
        connection.execute(
            "UPDATE decisions SET payload=? WHERE id=?",
            (canonical({**proposal, "reasoning": reasoning}), proposal["id"]),
        )
    projection = build_evolution(store)
    child = next(
        node for node in projection["nodes"] if node["id"] == f"experiment:{child_id}"
    )
    assert child["execution_outcome"] == "validated_result"
    assert child["scientific_assessments"][0]["outcome"] == "refutes"
    decision = next(node for node in projection["nodes"] if node["kind"] == "decision")
    assert decision["reasoning"] == reasoning
    assert any(
        edge["source"] == "lesson_use:use-1" and edge["target"] == "decision:decision-1"
        for edge in projection["edges"]
    )
    assert any(edge["kind"] == "assessed_evidence" for edge in projection["edges"])


def test_execution_failure_is_not_automatic_scientific_refutation(tmp_path):
    store = _store(tmp_path)
    a = store.add_experiment({"x": 0}, rationale="First")
    b = store.add_experiment({"x": 1}, rationale="Second")
    with store.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET status='failed',failure='worker_spawn_failed:OSError' WHERE id=?",
            (a["id"],),
        )
        connection.execute(
            "UPDATE experiments SET status='failed',failure='nonzero_exit:2' WHERE id=?",
            (b["id"],),
        )
    nodes = [
        node for node in build_evolution(store)["nodes"] if node["kind"] == "experiment"
    ]
    assert {node["execution_outcome"] for node in nodes} == {
        "infrastructure_failure",
        "execution_failure",
    }
    assert all(node["scientific_assessments"] == [] for node in nodes)


def test_forged_or_cross_campaign_assessment_is_not_used(tmp_path):
    store = _store(tmp_path)
    experiment = store.add_experiment({"x": 0}, rationale="Baseline")
    _finish(store, experiment["id"], 4)
    payload = {
        "experiment_ids": [experiment["id"]],
        "outcome": "supports",
        "source": _namespace(store),
    }
    first = _research_record(store, "bad-digest", "assessment", payload)
    first["payload"]["outcome"] = "refutes"
    with store.transaction() as connection:
        connection.execute(
            "UPDATE research_records SET record=? WHERE id=?",
            (canonical(first), "bad-digest"),
        )
    _research_record(
        store,
        "wrong-study",
        "assessment",
        {**payload, "source": {**_namespace(store), "manifest_digest": "c" * 64}},
    )
    projection = build_evolution(store)
    assert not [
        edge for edge in projection["edges"] if edge["kind"] == "assessed_evidence"
    ]
    assert (
        len(
            [
                warning
                for warning in projection["warnings"]
                if warning["kind"] == "invalid_research_record_source"
            ]
        )
        == 2
    )


def test_markdown_escapes_markup_and_graph_omits_paths(tmp_path):
    store = CampaignStore(tmp_path)
    spec = demo_spec()
    spec["objective"] = "<img src=x onerror=alert(1)>"
    spec["search_space"] = {"x": ["/private/model"]}
    spec["baseline"] = {"x": "/private/model"}
    store.initialize(spec)
    context = store.context()
    store.apply_proposal(
        {
            "id": "escaped",
            "context_digest": context["context_digest"],
            "runtime": "pi",
            "rationale": "[danger](javascript:alert(1)) | *not formatting* `code`",
            "evidence_ids": [],
            "configs": [{"x": "/private/model"}],
        }
    )
    projection = build_evolution(store)
    report = render_evolution(projection)
    assert "/private/model" not in canonical(projection)
    assert "<img" not in report
    assert "&lt;img" in report
    assert "[danger](javascript:" not in report
    assert r"\[danger\]" in report
    assert r"\|" in report
    assert r"\*not formatting\*" in report


def test_actual_journal_supersession_and_exact_captured_evidence(tmp_path):
    from loopx_hpc.research import ResearchJournal

    store = _store(tmp_path)
    parent_id, child_id, _ = _lineage(store)
    journal = ResearchJournal(store)
    request = {
        "id": "assessment-original",
        "hypothesis": "A measured change reduces loss",
        "outcome": "inconclusive",
        "interpretation": "Insufficient scope for a general claim",
        "limitations": ["Only two deterministic points"],
        "next_action": "Review the bounded directional prediction",
        "experiment_ids": [parent_id, child_id],
        "agent": {"runtime": "pi", "model": "fixture-model"},
        "source": _namespace(store),
    }
    first = journal.record_assessment(request)
    second = journal.record_assessment(
        {
            **request,
            "id": "assessment-revised",
            "outcome": "supports",
            "interpretation": "The local directional prediction holds for these points",
            "supersedes": first["id"],
        }
    )
    before = store.database.read_bytes()
    projection = build_evolution(store)
    assert store.database.read_bytes() == before
    assert projection["source"]["snapshot_digest"] == digest(store.status())
    first_node = next(
        node
        for node in projection["nodes"]
        if node["id"] == "assessment:assessment-original"
    )
    assert first_node["source"]["recorded_evidence_valid"] is True
    assert first_node["superseded_by"] == ["assessment:assessment-revised"]
    assert any(
        edge["kind"] == "explicitly_superseded_by" for edge in projection["edges"]
    )
    assert any(
        event["research_ref"] == f"assessment:{second['id']}"
        for event in projection["timeline"]
    )
    experiment = next(
        node for node in projection["nodes"] if node["id"] == f"experiment:{child_id}"
    )
    assert len(experiment["scientific_assessments"]) == 2
    assert all(
        item["recorded_evidence_valid"] and item["evidence_unchanged"]
        for item in experiment["scientific_assessments"]
    )


def test_decimal_metric_difference_is_not_rounded_or_binary_noise(tmp_path):
    store = _store(tmp_path)
    parent = store.add_experiment({"x": 0}, rationale="Baseline")
    _finish(store, parent["id"], 0.1)
    child = store.add_experiment(
        {"x": 1}, rationale="Compare decimal measurements", parent_ids=[parent["id"]]
    )
    _finish(store, child["id"], 0.3)
    comparison = build_evolution(store)["comparisons"][0]
    assert comparison["delta"] == 0.2
    assert comparison["delta_exact"] == "0.2"
    assert comparison["numerically_improved"] is False


def test_original_execution_timestamp_provenance_detects_event_drift(tmp_path):
    from loopx_hpc.research import ResearchJournal

    store = _store(tmp_path)
    experiment = store.add_experiment({"x": 0}, rationale="Timestamped baseline")
    _finish(store, experiment["id"], 4)
    assessment = ResearchJournal(store).record_assessment(
        {
            "id": "timestamp-assessment",
            "source": _namespace(store),
            "hypothesis": "A baseline can be measured",
            "outcome": "inconclusive",
            "interpretation": "Only baseline measured",
            "limitations": ["No comparison"],
            "next_action": "Plan a contrasting configuration",
            "experiment_ids": [experiment["id"]],
            "agent": {"runtime": "pi", "model": "fixture-model"},
        }
    )
    captured = assessment["payload"]["evidence"][0]["execution_events"]
    assert len(captured) == 1
    with store.transaction() as connection:
        connection.execute(
            "UPDATE events SET created_at='2099-01-01T00:00:00+00:00' WHERE sequence=?",
            (captured[0]["sequence"],),
        )
    projection = build_evolution(store)
    node = next(
        node
        for node in projection["nodes"]
        if node["id"] == f"experiment:{experiment['id']}"
    )
    assert node["evidence_valid"] is True  # Artifact bytes still match.
    assert node["scientific_assessments"][0]["evidence_unchanged"] is False
    assert any(
        warning["kind"] == "assessment_evidence_changed"
        for warning in projection["warnings"]
    )
    assessment_node = next(
        node
        for node in projection["nodes"]
        if node["id"] == "assessment:timestamp-assessment"
    )
    assert assessment_node["payload"]["evidence"][0]["execution_events"] == captured


def _source_lesson(tmp_path, objective):
    from loopx_hpc.research import ResearchJournal

    source = CampaignStore(tmp_path)
    spec = demo_spec()
    spec["objective"] = objective
    source.initialize(spec)
    experiment = source.add_experiment({"x": 0}, rationale="Measure source baseline")
    _finish(source, experiment["id"], 4)
    journal = ResearchJournal(source)
    assessment = journal.record_assessment(
        {
            "id": "shared-assessment-id",
            "source": _namespace(source),
            "hypothesis": "A baseline can be measured",
            "outcome": "inconclusive",
            "interpretation": "Only the source baseline is measured",
            "limitations": ["No comparison"],
            "next_action": "Test a contrasting configuration",
            "experiment_ids": [experiment["id"]],
            "agent": {"runtime": "pi", "model": "fixture-model"},
        }
    )
    bundle = journal.make_lesson(
        {
            "title": "<img src=x onerror=alert(1)>",
            "claim": "The bounded source result motivates a local test",
            "tags": ["baseline"],
            "applicability": ["Check the target study independently"],
            "limitations": ["No transfer guarantee"],
            "assessment_ids": [assessment["id"]],
        }
    )
    return source, bundle, assessment


def test_real_cross_project_history_is_namespaced_deduplicated_snapshot_only(tmp_path):
    from loopx_hpc.knowledge import KnowledgeLibrary
    from loopx_hpc.research import ResearchJournal

    first, bundle_a, assessment_a = _source_lesson(tmp_path / "source-a", "Study A")
    second, bundle_b, _ = _source_lesson(tmp_path / "source-b", "Study B")
    library = KnowledgeLibrary(tmp_path / "library")
    library.publish(bundle_a, execute=True)
    library.publish(bundle_b, execute=True)
    target = _store(tmp_path / "target")
    journal = ResearchJournal(target)
    for use_id, bundle in (
        ("use-a", bundle_a),
        ("use-a-repeat", bundle_a),
        ("use-b", bundle_b),
    ):
        journal.record_lesson_use(
            {
                "id": use_id,
                "source": _namespace(target),
                "lesson_id": bundle["id"],
                "action": "adapt",
                "rationale": "Run an independent target test",
                "adaptation": "Keep the target study metric",
                "checks": ["Dataset scope checked"],
                "agent": {"runtime": "codex", "model": "fixture-model"},
            },
            library,
        )
    result = target.apply_proposal(
        {
            "id": "target-decision",
            "context_digest": target.context()["context_digest"],
            "runtime": "codex",
            "rationale": "Explicitly adapt captured source lessons",
            "evidence_ids": [],
            "configs": [{"x": 0}],
            "reasoning": {
                "hypothesis": "The baseline transfers only as a testable starting point",
                "prediction": "Obtain a finite target loss",
                "selection_basis": "Test source advice locally",
                "alternatives": [
                    {
                        "config": {"x": 1},
                        "reason_not_selected": "Measure baseline first",
                    }
                ],
                "uncertainty": "Transfer remains untested",
                "lesson_use_ids": ["use-a", "use-b"],
                "agent": {"runtime": "codex", "model": "fixture-model"},
            },
        },
        knowledge_library=library,
    )
    target_id = result["experiment_ids"][0]
    _finish(target, target_id, 2)
    journal.record_assessment(
        {
            **{
                key: assessment_a["payload"][key]
                for key in (
                    "id",
                    "hypothesis",
                    "outcome",
                    "interpretation",
                    "limitations",
                    "next_action",
                    "agent",
                )
            },
            "experiment_ids": [target_id],
            "source": _namespace(target),
        }
    )
    library.withdraw(
        bundle_a["id"], "Later withdrawal cannot rewrite a past use", execute=True
    )
    # Imported projection must work with neither source campaign available.
    first.root.rename(tmp_path / "offline-source-a")
    second.root.rename(tmp_path / "offline-source-b")
    before = target.database.read_bytes()
    graph = build_evolution(target)
    assert target.database.read_bytes() == before
    assert build_evolution(target) == graph
    assert not graph["warnings"]
    assert graph["comparisons"] == []  # No comparison across frozen studies.
    groups = {}
    for node in graph["nodes"]:
        groups.setdefault(node["kind"], []).append(node)
    assert len(groups["imported_lesson"]) == 2
    assert len(groups["captured_assessment"]) == 2
    assert len(groups["captured_experiment"]) == 2
    assert len(groups["lesson_use"]) == 3
    captured = groups["captured_experiment"]
    assert {node["experiment_id"] for node in captured} == {target_id}
    assert {node["captured_metrics"]["loss"] for node in captured} == {4}
    assert len({node["id"] for node in graph["nodes"]}) == len(graph["nodes"])
    for node in captured + groups["captured_assessment"] + groups["imported_lesson"]:
        assert node["source"]["manifest_digest"] in node["id"]
        assert node["capture"]["snapshot_only"] is True
        assert node["capture"]["source_accessed"] is False
        assert node["capture"]["current_source_artifacts"] == "not_checked"
        assert "evidence_valid" not in node
    assert (
        captured[0]["execution_events"]
        == assessment_a["payload"]["evidence"][0]["execution_events"]
    )
    assert (
        groups["captured_assessment"][0]["recorded_at"] == assessment_a["recorded_at"]
    )
    assert all(
        "assessments" not in node["payload"]["lesson"] for node in groups["lesson_use"]
    )
    assert all(
        node["payload"]["lesson_state_at_use"] == "active"
        for node in groups["lesson_use"]
    )
    assert {edge["kind"] for edge in graph["edges"]} == {
        "captured_assessed_evidence",
        "captured_assessment_in_lesson",
        "imported_for_lesson_use",
        "cited_lesson_use",
        "accepted_configuration",
        "assessed_evidence",
    }
    import_edges = [
        edge for edge in graph["edges"] if edge["kind"] == "imported_for_lesson_use"
    ]
    assert len(import_edges) == 3
    assert all(edge["provenance"]["library_id"] for edge in import_edges)
    assert all(
        edge["provenance"]["current_library_state"] == "not_checked"
        for edge in import_edges
    )
    report = render_evolution(graph)
    assert "Captured source history (snapshot only)" in report
    assert "Current source artifacts were not accessed or verified" in report
    assert "Selection basis:" in report and "Alternative" in report
    assert "Interpretation:" in report and "Applicability checks:" in report
    assert "<img" not in report and "&lt;img" in report
    assert '"execution_events"' not in report  # No nested provenance JSON dump.
    assert str(tmp_path) not in canonical(graph)


def test_tampered_captured_lesson_does_not_expand_despite_valid_local_envelope(
    tmp_path,
):
    _, bundle, _ = _source_lesson(tmp_path / "source", "Original study")
    bundle["payload"]["assessments"][0]["payload"]["evidence"][0]["metrics"][
        "loss"
    ] = -999
    target = _store(tmp_path / "target")
    _research_record(
        target,
        "bad-use",
        "lesson_use",
        {
            "id": "bad-use",
            "source": _namespace(target),
            "lesson_id": bundle["id"],
            "lesson": bundle,
            "action": "adapt",
            "lesson_state_at_use": "active",
            "library_id": "captured-library",
            "rationale": "Tampered fixture",
        },
    )
    graph = build_evolution(target)
    assert [node["kind"] for node in graph["nodes"]] == ["lesson_use"]
    assert graph["nodes"][0]["source"]["digest_valid"] is True
    assert graph["nodes"][0]["payload"]["lesson"]["content_integrity_verified"] is False
    assert graph["edges"] == []
    assert graph["warnings"] == [
        {"kind": "invalid_captured_lesson", "record_id": "bad-use"}
    ]
