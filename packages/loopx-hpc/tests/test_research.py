from copy import deepcopy
import json
import sqlite3

import pytest

from loopx_hpc.campaign import CampaignStore, digest
from loopx_hpc.cli import demo, demo_spec, main
from loopx_hpc.knowledge import KnowledgeLibrary
from loopx_hpc.research import ResearchJournal
from loopx_hpc.research_demo import learning_demo


@pytest.fixture
def source(tmp_path):
    demo(tmp_path / "source", execute=True)
    return CampaignStore(tmp_path / "source")


def assessment(store, **updates):
    return {
        "id": "finding-1",
        "hypothesis": "The midpoint reduces loss",
        "outcome": "supports",
        "interpretation": "Measured loss decreases from 4 to 0",
        "limitations": ["Synthetic deterministic workload only"],
        "next_action": "Retest on a separate qualified study",
        "experiment_ids": store.context()["eligible_evidence_ids"],
        "agent": {"runtime": "test", "model": "none"},
        "source": {
            "campaign_id": store.manifest()["id"],
            "manifest_digest": digest(store.manifest()),
        },
        **updates,
    }


def lesson_request(**updates):
    return {
        "title": "Midpoint observation",
        "claim": "Midpoint is better in this exact workload",
        "tags": ["synthetic"],
        "applicability": ["Same workload and metric"],
        "limitations": ["Not a general optimization theorem"],
        "assessment_ids": ["finding-1"],
        **updates,
    }


def publish(source, library):
    journal = ResearchJournal(source)
    journal.record_assessment(assessment(source))
    bundle = journal.make_lesson(lesson_request())
    library.publish(bundle, execute=True)
    return bundle


def target_and_use(tmp_path, source):
    library = KnowledgeLibrary(tmp_path / "shared")
    bundle = publish(source, library)
    target = CampaignStore(tmp_path / "target")
    spec = demo_spec()
    spec["id"] = "target-study"
    target.initialize(spec)
    use = ResearchJournal(target).record_lesson_use(
        {
            "id": "use-1",
            "lesson_id": bundle["id"],
            "action": "adapt",
            "rationale": "Retest the candidate with matching provenance",
            "adaptation": "Independent local trial",
            "checks": ["Same workload hash and metric direction"],
            "agent": {"runtime": "test", "model": "none"},
            "source": {"campaign_id": spec["id"], "manifest_digest": digest(spec)},
        },
        library,
    )
    return target, library, bundle, use


def proposal(store, use_id="use-1", **updates):
    return {
        "id": "decision-1",
        "context_digest": store.context()["context_digest"],
        "runtime": "test",
        "rationale": "Test a lesson-derived candidate",
        "evidence_ids": [],
        "configs": [{"x": 2}],
        "reasoning": {
            "hypothesis": "A midpoint may transfer",
            "prediction": "loss below 4",
            "selection_basis": "Reviewed exact source applicability",
            "alternatives": [
                {
                    "config": {"x": 1},
                    "reason_not_selected": "One candidate suffices for this test",
                }
            ],
            "uncertainty": "Independent verification is required",
            "lesson_use_ids": [use_id],
            "agent": {"runtime": "test", "model": "none"},
        },
        **updates,
    }


def test_assessment_replay_correction_and_context_fence(source):
    journal = ResearchJournal(source)
    before = source.context()["context_digest"]
    request = assessment(source)
    record = journal.record_assessment(request)
    assert record["digest"] == digest(
        {k: v for k, v in record.items() if k != "digest"}
    )
    assert source.context()["context_digest"] != before
    assert journal.record_assessment(request) == record
    with pytest.raises(ValueError, match="reused"):
        journal.record_assessment(
            {**request, "interpretation": "A different conclusion"}
        )
    journal.record_assessment(
        {
            **request,
            "id": "finding-2",
            "supersedes": "finding-1",
            "outcome": "inconclusive",
        }
    )
    with pytest.raises(ValueError, match="non-superseded"):
        journal.make_lesson(lesson_request())
    assert journal.make_lesson(lesson_request(assessment_ids=["finding-2"]))
    with pytest.raises(ValueError, match="already superseded"):
        journal.record_assessment(
            {**request, "id": "finding-3", "supersedes": "finding-1"}
        )
    with source.transaction() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM events WHERE kind='research_recorded'"
            ).fetchone()[0]
            == 2
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM research_records")


@pytest.mark.parametrize("state", ["planned", "unknown", "failed", "running"])
def test_execution_outcome_is_not_a_scientific_conclusion(tmp_path, state):
    store = CampaignStore(tmp_path)
    store.initialize(demo_spec())
    experiment = store.add_experiment({"x": 0}, rationale="Test evidence qualification")
    with store.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET status=?,token='attempt-1' WHERE id=?",
            (state, experiment["id"]),
        )
        if state == "failed":
            store.event(
                connection,
                "attempt_finished",
                experiment["id"],
                {"token": "attempt-1", "status": "failed"},
            )
    journal = ResearchJournal(store)
    request = assessment(store, outcome="refutes", experiment_ids=[experiment["id"]])
    with pytest.raises(ValueError, match="verified successful"):
        journal.record_assessment(request)
    request["outcome"] = "infrastructure_failure"
    if state == "failed":
        assert (
            journal.record_assessment(request)["payload"]["outcome"]
            == "infrastructure_failure"
        )
        assert journal.make_lesson(lesson_request())["payload"]["assessments"]
    else:
        with pytest.raises(ValueError, match="actual failed"):
            journal.record_assessment(request)


def test_tampered_evidence_keeps_history_but_blocks_new_lesson(source):
    journal = ResearchJournal(source)
    record = journal.record_assessment(assessment(source))
    experiment = source.experiments()[0]
    result = next(
        a for a in experiment["artifacts"] if a["path"].endswith("result.json")
    )
    (source.root / result["path"]).write_text('{"metrics":{"loss":-999}}')
    assert source.status()["research_records"] == [record]
    with pytest.raises(ValueError, match="no longer verified"):
        journal.make_lesson(lesson_request())


def test_source_namespace_and_no_raw_output_export(source, tmp_path):
    journal = ResearchJournal(source)
    journal.record_assessment(assessment(source))
    (source.root / "private.log").write_text("SENTINEL-SECRET")
    bundle = journal.make_lesson(lesson_request())
    raw = json.dumps(bundle)
    assert "SENTINEL-SECRET" not in raw and str(source.root) not in raw
    assert '"path"' not in raw and '"process_id"' not in raw
    assert bundle["payload"]["source"]["manifest_digest"] == digest(source.manifest())
    other = CampaignStore(tmp_path / "same-name-different-study")
    spec = demo_spec()
    spec["provenance"]["dataset_revision"] = "a-different-dataset"
    other.initialize(spec)
    duplicate_id = other.add_experiment(
        {"x": 0}, rationale="same config different source"
    )["id"]
    assert duplicate_id == source.experiments()[0]["id"]
    assert digest(other.manifest()) != bundle["payload"]["source"]["manifest_digest"]


def test_lesson_reuse_is_advisory_and_requires_fresh_library(source, tmp_path):
    target, library, bundle, use = target_and_use(tmp_path, source)
    assert target.context()["eligible_evidence_ids"] == []
    assert target.context()["incumbent_id"] is None
    assert target.experiments() == [] and not (target.root / "runs").exists()
    request = proposal(target)
    with pytest.raises(ValueError, match="fresh explicit"):
        target.apply_proposal(request)
    result = target.apply_proposal(request, knowledge_library=library)
    assert target.get_experiment(result["experiment_ids"][0])["status"] == "planned"
    assert not (target.root / "runs").exists()
    library.withdraw(bundle["id"], "Source conclusion needs correction", execute=True)
    assert (
        target.apply_proposal(request, knowledge_library=library) == result
    )  # Historical replay.
    with pytest.raises(ValueError, match="withdrawn"):
        target.apply_proposal(
            proposal(target, id="decision-2", configs=[{"x": 3}]),
            knowledge_library=library,
        )
    assert target.status()["research_records"] == [use]


def test_research_update_stales_proposal_and_invalid_reasoning_is_atomic(
    source, tmp_path
):
    target, library, bundle, use = target_and_use(tmp_path, source)
    request = proposal(target)
    original = target.context()
    invalid = deepcopy(request)
    invalid["reasoning"]["agent"]["runtime"] = "different"
    with pytest.raises(ValueError, match="runtime"):
        target.apply_proposal(invalid, knowledge_library=library)
    invalid = deepcopy(request)
    invalid["reasoning"]["alternatives"][0]["config"] = {"x": True}
    with pytest.raises(ValueError, match="search space"):
        target.apply_proposal(invalid, knowledge_library=library)
    assert target.context() == original
    use_request = {
        k: use["payload"][k]
        for k in (
            "id",
            "lesson_id",
            "action",
            "rationale",
            "adaptation",
            "checks",
            "agent",
            "source",
        )
    }
    ResearchJournal(target).record_lesson_use(
        {**use_request, "id": "reject-2", "action": "reject"}, library
    )
    with pytest.raises(ValueError, match="stale"):
        target.apply_proposal(request, knowledge_library=library)
    with pytest.raises(ValueError, match="adopted/adapted"):
        target.apply_proposal(
            proposal(target, use_id="reject-2"), knowledge_library=library
        )


def test_legacy_snapshot_unchanged_and_read_cli_has_no_schema_effect(tmp_path, capsys):
    store = CampaignStore(tmp_path / "study")
    store.initialize(demo_spec())
    before = store.context()
    assert set(store.status()) == {"campaign", "experiments", "decisions"}
    assert main(["--root", str(store.root), "evolution"]) == 0
    assert json.loads(capsys.readouterr().out)["source"]["manifest_digest"] == digest(
        store.manifest()
    )
    assert store.context() == before
    with store.transaction() as connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='research_records'"
        ).fetchone()
    assert main(["lesson-search", "--library", str(tmp_path / "absent")]) == 0
    assert not (tmp_path / "absent").exists()


def test_real_cross_project_learning_demo_and_replay(tmp_path):
    assert learning_demo(tmp_path)["preview"]
    assert not (tmp_path / "source").exists()
    result = learning_demo(tmp_path, execute=True)
    assert len(result["source"]["experiments"]) == 3
    assert len(result["target"]["experiments"]) == 1
    assert result["target"]["experiments"][0]["metrics"] == {"loss": 0}
    assert result["target"]["experiments"][0]["parents"] == []
    assert result["target"]["decisions"][0]["reasoning"]["lesson_use_ids"] == [
        "check-midpoint-lesson"
    ]
    assert result["source"]["research_records"][0]["payload"]["evidence"]
    assert result["target_evolution"]["edges"]
    assert result["model_calls"] == result["remote_jobs"] == 0
    assert learning_demo(tmp_path, execute=True) == result
