"""Deterministic source-to-target transfer acceptance, not autonomous discovery."""

from pathlib import Path

from .campaign import CampaignStore, digest
from .cli import _wait, demo, demo_spec
from .evolution import build_evolution
from .knowledge import KnowledgeLibrary
from .local import LocalExecutor
from .research import ResearchJournal


def learning_demo(root: Path, *, execute: bool = False) -> dict:
    if not execute:
        return {
            "preview": True,
            "execution_enabled": False,
            "source_experiments": 3,
            "target_experiments": 1,
            "model_calls": 0,
            "remote_jobs": 0,
            "description": "Measure a synthetic bracket, assess the midpoint, publish locally, explicitly check and retest in a new study",
        }
    root = Path(root)
    demo(root / "source", execute=True)
    source = CampaignStore(root / "source")
    journal = ResearchJournal(source)
    source_context = source.context()
    measured = {
        r["config"]["x"]: r["metrics"].get("loss")
        for r in source_context["experiments"]
        if r["id"] in source_context["eligible_evidence_ids"]
    }
    if measured != {0: 4, 4: 4, 2: 0}:
        raise ValueError(
            "synthetic source results differ; do not issue the demonstration conclusion"
        )
    journal.record_assessment(
        {
            "id": "midpoint-assessment",
            "hypothesis": source_context["campaign"]["hypothesis"],
            "outcome": "supports",
            "interpretation": "Measured endpoint losses are 4 and the midpoint loss is 0 in this synthetic function",
            "limitations": [
                "Three deterministic measurements on one synthetic function; no uncertainty estimate or generalization claim"
            ],
            "next_action": "Offer a conditional midpoint candidate for a separate study with the exact same workload",
            "experiment_ids": source_context["eligible_evidence_ids"],
            "agent": {"runtime": "deterministic-demo", "model": "none"},
            "source": {
                "campaign_id": source.manifest()["id"],
                "manifest_digest": digest(source.manifest()),
            },
        }
    )
    bundle = journal.make_lesson(
        {
            "title": "Synthetic bracket midpoint candidate",
            "claim": "In this exact quadratic workload, x=2 produced loss=0 versus loss=4 at x=0 and x=4",
            "tags": ["synthetic", "quadratic", "midpoint"],
            "applicability": [
                "Same workload hash, dataset revision, metric direction and finite x search space"
            ],
            "limitations": [
                "A conditional candidate, not a generally optimal configuration or scientific discovery"
            ],
            "assessment_ids": ["midpoint-assessment"],
        }
    )
    library = KnowledgeLibrary(root / "library")
    library.publish(bundle, execute=True)
    target = CampaignStore(root / "target")
    spec = demo_spec()
    spec["id"] = "independent-transfer-study"
    spec["objective"] = "Independently retest a conditionally transferred candidate"
    spec["limits"] = {"max_experiments": 1, "max_concurrent": 1, "timeout_seconds": 10}
    target.initialize(spec)
    target_journal = ResearchJournal(target)
    use = target_journal.record_lesson_use(
        {
            "id": "check-midpoint-lesson",
            "lesson_id": bundle["id"],
            "action": "adopt",
            "rationale": "The source lesson applies to the identical synthetic workload; independently rerun the candidate",
            "adaptation": "No parameter adaptation; source observations are not imported as target results",
            "checks": [
                "Compared exact code and dataset revisions",
                "Checked metric name, direction and allowed x candidates",
                "Target has one explicitly authorized local trial",
            ],
            "agent": {"runtime": "deterministic-demo", "model": "none"},
            "source": {"campaign_id": spec["id"], "manifest_digest": digest(spec)},
        },
        library,
    )
    context = target.context()
    previous = next(
        (p for p in context["decisions"] if p["id"] == "transfer-midpoint"), None
    )
    proposal = previous or {
        "id": "transfer-midpoint",
        "context_digest": context["context_digest"],
        "runtime": "deterministic-demo",
        "rationale": "Retest the explicit, conditionally adopted midpoint candidate",
        "evidence_ids": [],
        "configs": [{"x": 2}],
        "reasoning": {
            "hypothesis": "The same exact synthetic workload reproduces the source midpoint loss",
            "prediction": "x=2 yields loss=0",
            "selection_basis": "Explicit lesson applicability checks and a fresh local trial",
            "alternatives": [
                {
                    "config": {"x": 1},
                    "reason_not_selected": "This trial tests transfer of the source candidate, not a new search",
                }
            ],
            "uncertainty": "No evidence for transfer beyond the identical deterministic workload",
            "lesson_use_ids": [use["id"]],
            "agent": {"runtime": "deterministic-demo", "model": "none"},
        },
    }
    accepted = target.apply_proposal(proposal, knowledge_library=library)
    for experiment_id in accepted["experiment_ids"]:
        LocalExecutor(target).submit(experiment_id, execute=True)
    _wait(target, accepted["experiment_ids"])
    if any(
        target.get_experiment(identifier)["metrics"] != {"loss": 0}
        for identifier in accepted["experiment_ids"]
    ):
        raise ValueError(
            "target result differs; do not issue the demonstration transfer conclusion"
        )
    target_journal.record_assessment(
        {
            "id": "transfer-assessment",
            "hypothesis": proposal["reasoning"]["hypothesis"],
            "outcome": "supports",
            "interpretation": "The independent local target trial measured loss=0 at x=2",
            "limitations": [
                "Same deterministic workload, not an independent scientific dataset or a model-driven planner"
            ],
            "next_action": "Use this as integration acceptance only; design a real preregistered study before scientific claims",
            "experiment_ids": accepted["experiment_ids"],
            "agent": {"runtime": "deterministic-demo", "model": "none"},
            "source": {"campaign_id": spec["id"], "manifest_digest": digest(spec)},
        }
    )
    return {
        "demo": "completed",
        "model_calls": 0,
        "remote_jobs": 0,
        "lesson_id": bundle["id"],
        "source": source.context(),
        "target": target.context(),
        "source_evolution": build_evolution(source),
        "target_evolution": build_evolution(target),
        "boundary": "Actual local execution and explicit advisory reuse; no autonomous agent or scientific generalization claim",
    }
