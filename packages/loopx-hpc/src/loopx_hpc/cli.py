"""Local operator CLI. Remote job rendering never dispatches a command."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from .campaign import CampaignStore
from .local import LocalExecutor


def demo_spec() -> dict:
    workload = Path(__file__).with_name("demo_workload.py")
    return {
        "id": "local-quadratic-study",
        "objective": "Validate an evidence-linked local experiment loop",
        "hypothesis": "The midpoint of the initial bracket reduces the synthetic loss",
        "metric": {"name": "loss", "direction": "minimize"},
        "search_space": {"x": [0, 1, 2, 3, 4]},
        "baseline": {"x": 0},
        "command": [
            "{python}",
            "-m",
            "loopx_hpc.demo_workload",
            "{config}",
            "{result}",
        ],
        "limits": {"max_experiments": 3, "max_concurrent": 2, "timeout_seconds": 10},
        "provenance": {
            "code_revision": "sha256:"
            + hashlib.sha256(workload.read_bytes()).hexdigest(),
            "dataset_revision": "synthetic-quadratic-v1",
            "environment": f"Python {platform.python_version()} on {platform.system()} {platform.machine()}",
        },
    }


def _wait(store: CampaignStore, ids: list[str], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while True:
        records = [
            LocalExecutor(store).reconcile(experiment_id) for experiment_id in ids
        ]
        if any(r["status"] in ("failed", "unknown") for r in records):
            raise ValueError(
                "demo attempt failed or is unresolved; inspect status and private logs"
            )
        if all(r["status"] == "succeeded" for r in records):
            return
        if time.monotonic() >= deadline:
            raise ValueError(
                "demo observation deadline reached; jobs were not resubmitted"
            )
        time.sleep(0.1)


def demo(root: Path, *, execute: bool = False) -> dict:
    spec = demo_spec()
    if not execute:
        return {"preview": True, "campaign": spec, "execution_enabled": False}
    store = CampaignStore(root)
    store.initialize(spec)
    initial = [
        store.add_experiment({"x": x}, rationale="Initial preregistered bracket")
        for x in (0, 4)
    ]
    for experiment in initial:
        LocalExecutor(store).submit(experiment["id"], execute=True)
    _wait(store, [r["id"] for r in initial])
    context = store.context()
    decisions = context["decisions"]
    if not decisions:
        # Deliberately transparent demonstration policy, not an LLM or a
        # general-purpose optimizer. Evidence comes from actual worker results.
        ranked = sorted(
            [
                r
                for r in context["experiments"]
                if r["id"] in context["eligible_evidence_ids"]
            ],
            key=lambda r: r["metrics"]["loss"],
        )
        midpoint = (ranked[0]["config"]["x"] + ranked[1]["config"]["x"]) // 2
        proposal = {
            "id": "midpoint-followup",
            "context_digest": context["context_digest"],
            "runtime": "deterministic-demo",
            "rationale": "Evaluate the midpoint of the two measured initial configurations",
            "evidence_ids": [r["id"] for r in ranked[:2]],
            "configs": [{"x": midpoint}],
            "reasoning": {
                "hypothesis": spec["hypothesis"],
                "prediction": "The midpoint has lower loss than both measured endpoints",
                "selection_basis": "Both measured endpoints have equal loss; test the center of this synthetic bracket",
                "alternatives": [
                    {
                        "config": {"x": 1},
                        "reason_not_selected": "One remaining trial favors the bracket midpoint",
                    }
                ],
                "uncertainty": "This deterministic synthetic demonstration establishes neither general optimization nor scientific validity",
                "lesson_use_ids": [],
                "agent": {"runtime": "deterministic-demo", "model": "none"},
            },
        }
        outcome = store.apply_proposal(proposal)
        followup_ids = outcome["experiment_ids"]
    else:
        followup_ids = [r["id"] for r in context["experiments"] if r["parents"]]
    for experiment_id in followup_ids:
        LocalExecutor(store).submit(experiment_id, execute=True)
    _wait(store, followup_ids)
    from .tracking import export_tracking

    tracking = export_tracking(store.status(), root / "tracking", "json")
    return {
        "demo": "completed",
        "model_calls": 0,
        "remote_jobs": 0,
        "context": store.context(),
        "tracking": tracking,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".local/hpc-campaign"))
    sub = parser.add_subparsers(dest="action", required=True)
    initialize = sub.add_parser(
        "init", help="Freeze a trusted, owner-reviewed study manifest"
    )
    initialize.add_argument("--manifest", type=Path, required=True)
    add = sub.add_parser("add", help="Record an experiment; does not launch it")
    add.add_argument("--config", required=True, help="JSON object")
    add.add_argument("--rationale", required=True)
    add.add_argument("--parent", action="append", default=[])
    for name in ("status", "context", "reconcile"):
        sub.add_parser(name)
    propose = sub.add_parser(
        "propose", help="Validate an evidence-linked runtime-neutral proposal"
    )
    propose.add_argument("--input", required=True, type=Path)
    propose.add_argument(
        "--library",
        type=Path,
        help="Explicit local library; required when citing lessons",
    )
    evolution = sub.add_parser(
        "evolution", help="Read the evidence-linked evolution graph and timeline"
    )
    evolution.add_argument("--format", choices=("json", "markdown"), default="json")
    assess = sub.add_parser(
        "assess", help="Append an authored, evidence-verified scientific assessment"
    )
    assess.add_argument("--input", type=Path, required=True)
    publish = sub.add_parser(
        "lesson-publish", help="Preview or explicitly share a verified lesson locally"
    )
    publish.add_argument("--input", type=Path, required=True)
    publish.add_argument("--library", type=Path, required=True)
    publish.add_argument("--execute", action="store_true")
    search = sub.add_parser(
        "lesson-search", help="Retrieve active advisory lessons by exact AND tags"
    )
    search.add_argument("--library", type=Path, required=True)
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--limit", type=int, default=10)
    use = sub.add_parser(
        "lesson-use",
        help="Record an explicit applicability assessment in the target study",
    )
    use.add_argument("--input", type=Path, required=True)
    use.add_argument("--library", type=Path, required=True)
    withdraw = sub.add_parser(
        "lesson-withdraw", help="Preview or explicitly withdraw a shared lesson"
    )
    withdraw.add_argument("--library", type=Path, required=True)
    withdraw.add_argument("--lesson", required=True)
    withdraw.add_argument("--reason", required=True)
    withdraw.add_argument("--execute", action="store_true")
    submit = sub.add_parser(
        "submit", help="Preview or explicitly launch one local experiment"
    )
    submit.add_argument("--experiment", required=True)
    submit.add_argument("--execute", action="store_true")
    track = sub.add_parser(
        "track", help="Export a local tracking projection; never uploads"
    )
    track.add_argument(
        "--backend",
        choices=("json", "mlflow", "wandb", "wandb-offline"),
        default="json",
    )
    track.add_argument("--destination", type=Path, required=True)
    render = sub.add_parser(
        "render-job", help="Print a scheduler script only; never submits"
    )
    render.add_argument("--scheduler", choices=("pbs", "slurm"), required=True)
    render.add_argument("--input", type=Path, required=True)
    demonstration = sub.add_parser(
        "demo", help="Run the bounded synthetic acceptance study"
    )
    demonstration.add_argument("--execute", action="store_true")
    learning = sub.add_parser(
        "learning-demo",
        help="Run a bounded source-to-target lesson transfer acceptance study",
    )
    learning.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    store = CampaignStore(args.root)
    try:
        if args.action == "init":
            result = store.initialize(json.loads(args.manifest.read_text()))
        elif args.action == "add":
            result = store.add_experiment(
                json.loads(args.config),
                rationale=args.rationale,
                parent_ids=args.parent,
            )
        elif args.action == "context":
            result = store.context()
        elif args.action == "status":
            result = store.status()
        elif args.action == "propose":
            from .knowledge import KnowledgeLibrary

            result = store.apply_proposal(
                json.loads(args.input.read_text()),
                knowledge_library=KnowledgeLibrary(args.library)
                if args.library
                else None,
            )
        elif args.action == "evolution":
            from .evolution import build_evolution, render_evolution

            result = build_evolution(store)
            if args.format == "markdown":
                print(render_evolution(result))
                return 0
        elif args.action == "assess":
            from .research import ResearchJournal

            result = ResearchJournal(store).record_assessment(
                json.loads(args.input.read_text())
            )
        elif args.action == "lesson-publish":
            from .knowledge import KnowledgeLibrary
            from .research import ResearchJournal

            bundle = ResearchJournal(store).make_lesson(
                json.loads(args.input.read_text())
            )
            result = KnowledgeLibrary(args.library).publish(
                bundle, execute=args.execute
            )
        elif args.action == "lesson-search":
            from .knowledge import KnowledgeLibrary

            result = KnowledgeLibrary(args.library).search(
                tags=args.tag, limit=args.limit
            )
        elif args.action == "lesson-use":
            from .knowledge import KnowledgeLibrary
            from .research import ResearchJournal

            result = ResearchJournal(store).record_lesson_use(
                json.loads(args.input.read_text()), KnowledgeLibrary(args.library)
            )
        elif args.action == "lesson-withdraw":
            from .knowledge import KnowledgeLibrary

            result = KnowledgeLibrary(args.library).withdraw(
                args.lesson, args.reason, execute=args.execute
            )
        elif args.action == "learning-demo":
            from .research_demo import learning_demo

            result = learning_demo(args.root, execute=args.execute)
        elif args.action == "submit":
            result = LocalExecutor(store).submit(args.experiment, execute=args.execute)
        elif args.action == "reconcile":
            result = {
                "experiments": [
                    LocalExecutor(store).reconcile(r["id"]) for r in store.experiments()
                ]
            }
        elif args.action == "track":
            from .tracking import export_tracking

            result = export_tracking(store.status(), args.destination, args.backend)
        elif args.action == "render-job":
            from .schedulers import render_script

            request = json.loads(args.input.read_text())
            if not isinstance(request, dict) or set(request) - {
                "command",
                "resources",
                "environment",
            }:
                raise ValueError(
                    "render input must contain command, resources and optional environment"
                )
            print(
                render_script(
                    args.scheduler,
                    request["command"],
                    request["resources"],
                    request.get("environment"),
                )
            )
            return 0
        else:
            result = demo(args.root, execute=args.execute)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
