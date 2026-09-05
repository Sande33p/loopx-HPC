"""Regression tests for provenance binding and advisory reuse boundaries.

Source evidence comes from actual bounded local workers. Library tests inspect
only explicit local SQLite stores; no model, remote job, or upload is involved.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
import time

import pytest

from loopx_hpc.campaign import CampaignStore
from loopx_hpc.cli import demo, demo_spec
from loopx_hpc.knowledge import KnowledgeLibrary
from loopx_hpc.local import LocalExecutor
from loopx_hpc.research import ResearchJournal

from test_research import (
    assessment,
    lesson_request,
    proposal,
    target_and_use,
)


@pytest.fixture
def source(tmp_path):
    demo(tmp_path / "source", execute=True)
    return CampaignStore(tmp_path / "source")


def completed_study(path, spec):
    store = CampaignStore(path)
    store.initialize(spec)
    row = store.add_experiment({"x": 0}, rationale="Provenance regression fixture")
    LocalExecutor(store).submit(row["id"], execute=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = LocalExecutor(store).reconcile(row["id"])
        if record["status"] in {"succeeded", "failed", "unknown"}:
            assert record["status"] == "succeeded"
            assert store.verify_record(record)
            return store
        time.sleep(0.025)
    pytest.fail("bounded provenance fixture did not finish")


def use_request(store, lesson_id, *, use_id="new-use", **changes):
    return {
        "id": use_id,
        "lesson_id": lesson_id,
        "action": "adapt",
        "rationale": "Review relevance to this exact target study",
        "adaptation": "Run an independent local verification",
        "checks": ["Metric and code provenance reviewed; transfer is not assumed"],
        "agent": {"runtime": "test", "model": "none"},
        "source": store.context()["source"],
        **changes,
    }


def test_same_experiment_id_does_not_authorize_wrong_source_assessment(
    source, tmp_path
):
    spec = source.manifest()
    spec["provenance"]["dataset_revision"] = "independent-dataset-revision"
    other = completed_study(tmp_path / "same-name-other-manifest", spec)
    experiment_id = other.experiments()[0]["id"]
    assert experiment_id == source.experiments()[0]["id"]
    assert other.context()["source"] != source.context()["source"]
    before = other.status()
    request = assessment(source, experiment_ids=[experiment_id])
    with pytest.raises(ValueError, match="source manifest mismatch"):
        ResearchJournal(other).record_assessment(request)
    assert other.status() == before
    accepted = ResearchJournal(other).record_assessment(
        {**request, "source": other.context()["source"]}
    )
    assert accepted["payload"]["source"] == other.context()["source"]


def test_wrong_target_source_cannot_record_lesson_use(source, tmp_path):
    target, library, bundle, _ = target_and_use(tmp_path, source)
    before = target.status()
    with pytest.raises(ValueError, match="target manifest mismatch"):
        ResearchJournal(target).record_lesson_use(
            use_request(target, bundle["id"], source=source.context()["source"]),
            library,
        )
    assert target.status() == before
    assert not (target.root / "runs").exists()


def test_an_active_copy_in_another_library_cannot_refresh_a_withdrawn_adoption(
    source, tmp_path
):
    target, library_a, bundle, use = target_and_use(tmp_path, source)
    library_b = KnowledgeLibrary(tmp_path / "other-library")
    library_b.publish(bundle, execute=True)
    library_a.withdraw(bundle["id"], "Source needs review", execute=True)
    assert library_b.get(bundle["id"])["state"] == "active"
    assert library_b.get(bundle["id"])["library_id"] != use["payload"]["library_id"]
    before = target.status()
    with pytest.raises(ValueError, match="library changed"):
        target.apply_proposal(proposal(target), knowledge_library=library_b)
    assert target.status() == before
    assert not (target.root / "runs").exists()
    # Choosing a different local curation authority is explicit and recorded,
    # not silently treated as a refresh of the first library.
    replacement_use = ResearchJournal(target).record_lesson_use(
        use_request(target, bundle["id"], use_id="explicit-library-b"), library_b
    )
    assert (
        replacement_use["payload"]["library_id"]
        == library_b.get(bundle["id"])["library_id"]
    )
    accepted = target.apply_proposal(
        proposal(target, use_id="explicit-library-b"), knowledge_library=library_b
    )
    assert target.get_experiment(accepted["experiment_ids"][0])["status"] == "planned"
    assert not (target.root / "runs").exists()


def test_withdrawal_cannot_commit_inside_proposal_acceptance_window(
    source, tmp_path, monkeypatch
):
    target, library, bundle, _ = target_and_use(tmp_path, source)
    entered_add = threading.Event()
    release_add = threading.Event()
    withdrawal_started = threading.Event()
    original_add = target._add

    def held_add(*args, **kwargs):
        entered_add.set()
        if not release_add.wait(5):
            raise RuntimeError("test acceptance barrier timed out")
        return original_add(*args, **kwargs)

    def withdraw():
        withdrawal_started.set()
        return library.withdraw(
            bundle["id"], "Concurrent owner withdrawal", execute=True
        )

    monkeypatch.setattr(target, "_add", held_add)
    request = proposal(target)
    with ThreadPoolExecutor(max_workers=2) as pool:
        accept = pool.submit(target.apply_proposal, request, knowledge_library=library)
        try:
            assert entered_add.wait(5)
            withdrawal = pool.submit(withdraw)
            assert withdrawal_started.wait(5)
            time.sleep(0.15)
            assert not withdrawal.done(), (
                "withdrawal committed while the proposal still held its library snapshot"
            )
        finally:
            release_add.set()
        accepted = accept.result(timeout=5)
        assert withdrawal.result(timeout=5)["changed"]
    assert target.get_experiment(accepted["experiment_ids"][0])["status"] == "planned"
    assert library.get(bundle["id"])["state"] == "withdrawn"
    # Commit ordering permits the historical accepted proposal, but not a later
    # proposal using the lesson after withdrawal.
    with pytest.raises(ValueError, match="withdrawn"):
        target.apply_proposal(
            proposal(target, id="after-withdrawal", configs=[{"x": 3}]),
            knowledge_library=library,
        )


def test_corrected_lesson_preserves_history_and_invalidates_old_proposal(
    source, tmp_path
):
    target, library, bundle, original_use = target_and_use(tmp_path, source)
    pending = proposal(target)
    journal = ResearchJournal(source)
    original_assessment = source.status()["research_records"][0]
    correction = journal.record_assessment(
        assessment(
            source,
            id="finding-2",
            supersedes="finding-1",
            outcome="inconclusive",
            interpretation="The evidence is narrower than the initial authored claim",
        )
    )
    with pytest.raises(ValueError, match="non-superseded"):
        journal.make_lesson(lesson_request())
    replacement = journal.make_lesson(
        lesson_request(
            assessment_ids=["finding-2"],
            claim="A narrower observation requiring independent verification",
            replaces=bundle["id"],
        )
    )
    library.publish(replacement, execute=True)
    before = target.status()
    with pytest.raises(ValueError, match="superseded"):
        target.apply_proposal(pending, knowledge_library=library)
    assert target.status() == before
    ResearchJournal(target).record_lesson_use(
        use_request(target, replacement["id"], use_id="review-correction"), library
    )
    with pytest.raises(ValueError, match="stale"):
        target.apply_proposal(pending, knowledge_library=library)
    assert source.status()["research_records"] == [original_assessment, correction]
    assert target.status()["research_records"][0] == original_use


@pytest.mark.parametrize(
    "extra",
    [
        {"raw_logs": "DO-NOT-EXPORT"},
        {"source_path": "/private/example"},
        {"execute": True},
    ],
)
def test_unsupported_assessment_fields_fail_without_schema_or_event_mutation(
    source, extra
):
    before = source.status()
    with source.transaction() as connection:
        events_before = connection.execute("SELECT count(*) FROM events").fetchone()[0]
    with pytest.raises(ValueError, match="request requires"):
        ResearchJournal(source).record_assessment({**assessment(source), **extra})
    assert source.status() == before
    with source.transaction() as connection:
        assert (
            connection.execute("SELECT count(*) FROM events").fetchone()[0]
            == events_before
        )
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='research_records'"
        ).fetchone()


def test_late_invalid_candidate_rolls_back_whole_lesson_informed_proposal(
    source, tmp_path
):
    target, library, _, _ = target_and_use(tmp_path, source)
    before = target.status()
    request = proposal(target, configs=[{"x": 2}, {"x": True}])
    with pytest.raises(ValueError, match="search space"):
        target.apply_proposal(request, knowledge_library=library)
    assert target.status() == before
    assert not (target.root / "runs").exists()


def test_export_selects_provenance_fields_but_does_not_claim_text_is_public_safe(
    tmp_path,
):
    spec = demo_spec()
    spec["provenance"]["private_notes"] = {"raw_log": "PRIVATE-PROVENANCE-SENTINEL"}
    store = completed_study(tmp_path / "source", spec)
    (store.root / "private.log").write_text("PRIVATE-LOG-SENTINEL")
    journal = ResearchJournal(store)
    authored = (
        "Owner-authored interpretation; this is not an automatically proven theorem"
    )
    journal.record_assessment(
        assessment(store, interpretation=authored, outcome="inconclusive")
    )
    bundle = journal.make_lesson(lesson_request())
    assert set(bundle["payload"]["source"]["provenance"]) == {
        "code_revision",
        "dataset_revision",
        "environment",
    }
    serialized = json.dumps(bundle)
    assert "PRIVATE-PROVENANCE-SENTINEL" not in serialized
    assert "PRIVATE-LOG-SENTINEL" not in serialized
    assert str(store.root) not in serialized
    assert authored in serialized  # Authored text is retained, not auto-redacted.
    library = KnowledgeLibrary(tmp_path / "explicit-private-library")
    library.publish(bundle, execute=True)
    view = library.get(bundle["id"])
    assert view["privacy"] == "explicit_local_library_not_public_safe"
    assert view["action_authority"] == "none"
    assert not view["truth_verified"] and not view["authorship_verified"]
    assert not view["imported_text_trusted"] and not view["source_accessed"]


def test_captured_execution_times_are_source_events_not_assessment_time(source):
    journal = ResearchJournal(source)
    accepted = journal.record_assessment(assessment(source))
    with source.transaction() as connection:
        original_events = {
            row["sequence"]: dict(row)
            for row in connection.execute("SELECT * FROM events")
        }
    for evidence in accepted["payload"]["evidence"]:
        kinds = {event["kind"] for event in evidence["execution_events"]}
        assert {"attempt_reserved", "worker_started", "attempt_finished"} <= kinds
        for captured in evidence["execution_events"]:
            original = original_events[captured["sequence"]]
            assert captured["recorded_at"] == original["created_at"]
            assert original["experiment_id"] == evidence["experiment_id"]
            assert json.loads(original["payload"])["token"] == evidence["attempt_token"]
            assert captured["recorded_at"] != accepted["recorded_at"]
    before = deepcopy(accepted)
    assert journal.record_assessment(assessment(source)) == before
