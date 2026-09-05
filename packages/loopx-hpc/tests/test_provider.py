"""Provider contract checks, including real LoopX install/bind/read-only dispatch.

The governed acceptance case obtains a real CLI admission in an isolated test
goal and uses real durable writeback, quota spend, and terminal closeout APIs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import pytest

from loopx_hpc.campaign import CampaignStore
from loopx_hpc import provider
from loopx_hpc.bridge import _environment, observe_experiment, start_local_experiment

PACKAGE = Path(__file__).resolve().parents[1]


def _campaign(tmp_path: Path, *, fail: bool = False) -> tuple[CampaignStore, str]:
    command = [sys.executable, "-m", "loopx_hpc.demo_workload", "{config}", "{result}"]
    if fail:
        command = [sys.executable, "-c", "raise SystemExit(3)", "{config}", "{result}"]
    store = CampaignStore(tmp_path / "campaign")
    store.initialize(
        {
            "id": "provider-test",
            "objective": "Validate provider boundaries",
            "hypothesis": "A fixed local objective can be measured",
            "metric": {"name": "loss", "direction": "minimize"},
            "search_space": {"x": [0, 2, 4]},
            "baseline": {"x": 0},
            "command": command,
            "limits": {
                "max_experiments": 3,
                "max_concurrent": 1,
                "timeout_seconds": 10,
            },
            "provenance": {
                "code_revision": "fixture-v1",
                "dataset_revision": "synthetic-v1",
                "environment": "test-python",
            },
        }
    )
    row = store.add_experiment({"x": 0}, rationale="Measure the baseline")
    return store, row["id"]


def _request(
    experiment_id: str, *, operation: str = "observe", phase: str = "start"
) -> dict:
    request = {
        "schema_version": provider.REQUEST_SCHEMA,
        "invocation_id": "capability-" + "a" * 24,
        "capability_id": provider.CAPABILITY_ID,
        "operation": operation,
        "input": {"experiment_id": experiment_id},
        "goal": {"goal_id": "provider-test-goal"},
    }
    if operation == "run_local":
        request["authority"] = {
            "effect_id": "test-effect-1",
            "goal_id": "provider-test-goal",
        }
        request["lifecycle"] = {"phase": phase, "idempotency_key": "test-effect-1"}
    return request


def _terminal(store: CampaignStore, request: dict) -> dict:
    request = {**request, "lifecycle": {**request["lifecycle"], "phase": "reconcile"}}
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = provider.handle_request(request, root=store.root)
        if result["status"] != "running":
            return result
        time.sleep(0.02)
    pytest.fail("local provider did not reach a terminal outcome")


def test_observe_has_no_executor_effect_and_redacts_private_data(tmp_path, monkeypatch):
    store, experiment_id = _campaign(tmp_path)
    before = store.status()

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only provider attempted executor access")

    monkeypatch.setattr(provider, "LocalExecutor", forbidden)
    result = provider.handle_request(_request(experiment_id), root=store.root)
    assert result["observations"][0]["experiment_status"] == "planned"
    assert result["effect_receipt"] is None
    assert result["transition_proposals"] == []
    assert result["domain_state_mutations"] == []
    assert store.status() == before
    encoded = json.dumps(result)
    assert str(tmp_path) not in encoded
    assert "test-python" not in encoded
    assert "Measure the baseline" not in encoded


@pytest.mark.parametrize(
    "bad_input",
    [
        {"experiment_id": "exp-1", "command": ["arbitrary"]},
        {"experiment_id": "../other-campaign"},
        {"experiment_id": "exp-1", "root": "/private/campaign"},
    ],
)
def test_rejects_arbitrary_commands_paths_and_extra_input(tmp_path, bad_input):
    store, experiment_id = _campaign(tmp_path)
    request = _request(experiment_id)
    request["input"] = bad_input
    with pytest.raises(ValueError):
        provider.handle_request(request, root=store.root)


def test_write_requires_matching_authority_and_known_phase(tmp_path):
    store, experiment_id = _campaign(tmp_path)
    request = _request(experiment_id, operation="run_local")
    request["authority"]["effect_id"] = "wrong-effect"
    with pytest.raises(ValueError, match="identity"):
        provider.handle_request(request, root=store.root)
    request = _request(experiment_id, operation="run_local", phase="cancel")
    with pytest.raises(ValueError, match="phase"):
        provider.handle_request(request, root=store.root)
    assert store.get_experiment(experiment_id)["status"] == "planned"


def test_real_local_execution_replay_and_upstream_terminal_validation(tmp_path):
    from loopx.extensions.governed_capability_execution import (
        validate_governed_capability_result,
    )

    store, experiment_id = _campaign(tmp_path)
    request = _request(experiment_id, operation="run_local")
    provider.handle_request(request, root=store.root)
    first = store.get_experiment(experiment_id)
    provider.handle_request(request, root=store.root)
    assert store.get_experiment(experiment_id)["token"] == first["token"]
    result = _terminal(store, request)
    assert result["observations"][0]["experiment_status"] == "succeeded"
    assert result["observations"][0]["evidence_valid"] is True
    validated = validate_governed_capability_result(
        result,
        invocation_id=request["invocation_id"],
        effect_id="test-effect-1",
        operation={
            "effect_class": "external_write",
            "result_schema": provider.RESULT_SCHEMA,
        },
    )
    assert validated["effect_receipt"]["status"] == "committed"
    assert str(store.root) not in json.dumps(result)


def test_failed_scientific_attempt_is_not_reported_as_successful_experiment(tmp_path):
    store, experiment_id = _campaign(tmp_path, fail=True)
    request = _request(experiment_id, operation="run_local")
    provider.handle_request(request, root=store.root)
    result = _terminal(store, request)
    # Protocol success is durable recording, never scientific success.
    assert result["status"] == "succeeded"
    assert result["observations"][0]["experiment_status"] == "failed"
    assert result["observations"][0]["needs_attention"] is True
    assert result["effect_receipt"]["status"] == "committed"


def test_unknown_attempt_stays_unresolved_without_resubmission(tmp_path):
    store, experiment_id = _campaign(tmp_path)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET status='unknown',token='original-attempt' WHERE id=?",
            (experiment_id,),
        )
    request = _request(experiment_id, operation="run_local")
    result = provider.handle_request(request, root=store.root)
    assert result["status"] == "running"
    assert result["observations"][0]["experiment_status"] == "unknown"
    assert result["effect_receipt"] is None
    assert store.get_experiment(experiment_id)["token"] == "original-attempt"
    assert not (store.root / "runs").exists()


def test_tampered_terminal_evidence_is_observable_but_cannot_settle(tmp_path):
    store, experiment_id = _campaign(tmp_path)
    request = _request(experiment_id, operation="run_local")
    provider.handle_request(request, root=store.root)
    _terminal(store, request)
    artifact = store.get_experiment(experiment_id)["artifacts"][0]
    (store.root / artifact["path"]).write_text("tampered test evidence")
    result = provider.handle_request(_request(experiment_id), root=store.root)
    assert result["observations"][0]["evidence_valid"] is False
    with pytest.raises(ValueError, match="evidence"):
        provider.handle_request(request, root=store.root)


def test_bridge_requires_selected_todo_for_exact_experiment(tmp_path):
    store, experiment_id = _campaign(tmp_path)
    with pytest.raises(ValueError, match="exact experiment"):
        start_local_experiment(
            state_file=tmp_path / "extensions.json",
            run_dir=tmp_path / "governed",
            registry_path=tmp_path / "registry.json",
            goal_id="test-goal",
            agent_id="test-agent",
            todo_id="test-todo",
            turn_instance_id="test-turn",
            root=store.root,
            experiment_id=experiment_id,
            context_refs=[],
            admission={"selected_todo": {"target_key": "experiment:different"}},
            execute=True,
        )
    assert store.get_experiment(experiment_id)["status"] == "planned"


def test_bridge_does_not_inherit_ambient_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLE_PROVIDER_API_KEY", "not-a-real-secret")
    environment = _environment(tmp_path)
    assert "EXAMPLE_PROVIDER_API_KEY" not in environment
    assert environment["LOOPX_HPC_ROOT"] == str(tmp_path.resolve())


def test_actual_extension_install_binding_read_only_dispatch_and_write_refusal(
    tmp_path,
):
    from loopx.extensions.capability_admission import (
        bind_external_capability_to_goal,
        invoke_external_capability,
    )
    from loopx.extensions.runtime import install_extension, run_standalone_extension

    store, experiment_id = _campaign(tmp_path)
    state_file = tmp_path / "extensions.json"
    installed = install_extension(
        PACKAGE / "extension.toml", state_file=state_file, execute=True
    )
    assert installed["doctor"]["verified"] is True
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "loopx_registry_v1",
                "common_runtime_root": str(tmp_path / "runtime"),
                "goals": [
                    {
                        "id": "provider-test-goal",
                        "title": "Provider integration test",
                        "repo": str(tmp_path),
                    }
                ],
            }
        )
    )
    bind_external_capability_to_goal(
        registry_path=registry,
        state_file=state_file,
        goal_id="provider-test-goal",
        capability_id=provider.CAPABILITY_ID,
        operations=["observe", "run_local"],
        execute=True,
    )
    context_refs = [
        {
            "kind": "experiment",
            "ref": f"experiment:{experiment_id}",
            "digest": "sha256:" + "b" * 64,
        }
    ]
    receipt = observe_experiment(
        state_file=state_file,
        registry_path=registry,
        goal_id="provider-test-goal",
        root=store.root,
        experiment_id=experiment_id,
        context_refs=context_refs,
        execute=True,
    )
    assert receipt["executed"] is True
    assert (
        receipt["provider_result"]["observations"][0]["experiment_status"] == "planned"
    )
    with pytest.raises(ValueError, match="governed Turn"):
        invoke_external_capability(
            state_file=state_file,
            registry_path=registry,
            goal_id="provider-test-goal",
            capability_id=provider.CAPABILITY_ID,
            operation="run_local",
            provider_input={
                "context_refs": context_refs,
                "input": {"experiment_id": experiment_id},
            },
            execute=True,
            environment={**os.environ, "LOOPX_HPC_ROOT": str(store.root)},
        )
    with pytest.raises(ValueError, match="standalone"):
        run_standalone_extension(
            "loopx-hpc", state_file=state_file, request={}, execute=True
        )
    assert store.get_experiment(experiment_id)["status"] == "planned"


def test_governed_bridge_real_admission_durable_writeback_spend_and_replay(tmp_path):
    """One real local attempt traverses upstream admission and settlement APIs."""
    from loopx.control_plane.testing.canary_harness import (
        run_json_cli,
        write_fixture_registry,
    )
    from loopx.extensions.capability_admission import bind_external_capability_to_goal
    from loopx.extensions.runtime import install_extension
    from loopx_hpc.bridge import reconcile_local_experiment

    store, experiment_id = _campaign(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    goal_id, agent_id, turn_id = "hpc-acceptance", "local-host", "local-turn-1"
    state_doc = project / ".codex" / "goals" / goal_id / "ACTIVE_GOAL_STATE.md"
    state_doc.parent.mkdir(parents=True)
    state_doc.write_text(
        "---\nstatus: active\n---\n\n# Active Goal State\n\n## Objective\n\nValidate one local baseline.\n\n## Agent Todo\n"
    )
    registry, runtime = project / ".loopx" / "registry.json", tmp_path / "runtime"
    write_fixture_registry(
        project=project,
        runtime_root=runtime,
        registry_path=registry,
        goal_id=goal_id,
        domain="experiment-acceptance",
        adapter_kind="generic_project_goal_v0",
        registered_agents=[agent_id],
        quota_allowed_slots=5,
    )

    def cli(*args):
        return run_json_cli(
            *args, registry_path=registry, runtime_root=runtime, cwd=project
        )

    todo = cli(
        "todo",
        "add",
        "--goal-id",
        goal_id,
        "--role",
        "agent",
        "--text",
        "Execute and validate the local baseline.",
        "--action-kind",
        "run_experiment",
        "--target-key",
        f"experiment:{experiment_id}",
    )
    todo_id = todo["todo_id"]
    cli(
        "todo",
        "claim",
        "--goal-id",
        goal_id,
        "--todo-id",
        todo_id,
        "--claimed-by",
        agent_id,
        "--agent-id",
        agent_id,
    )
    admission = cli(
        "quota",
        "should-run",
        "--goal-id",
        goal_id,
        "--agent-id",
        agent_id,
        "--todo-id",
        todo_id,
        "--turn-instance-id",
        turn_id,
        "--runtime-profile",
        "generic_cli",
        "--scan-path",
        str(project),
    )
    assert admission["should_run"] is True, admission
    assert admission["selected_todo"]["todo_id"] == todo_id
    extensions = tmp_path / "extensions.json"
    install_extension(PACKAGE / "extension.toml", state_file=extensions, execute=True)
    bind_external_capability_to_goal(
        registry_path=registry,
        state_file=extensions,
        goal_id=goal_id,
        capability_id=provider.CAPABILITY_ID,
        operations=["run_local"],
        execute=True,
    )
    context_refs = [
        {
            "kind": "experiment",
            "ref": f"experiment:{experiment_id}",
            "digest": "sha256:" + store.get_experiment(experiment_id)["config_hash"],
        }
    ]
    arguments = dict(
        state_file=extensions,
        run_dir=tmp_path / "governed",
        registry_path=registry,
        goal_id=goal_id,
        agent_id=agent_id,
        todo_id=todo_id,
        turn_instance_id=turn_id,
        root=store.root,
        experiment_id=experiment_id,
        context_refs=context_refs,
        admission=admission,
    )
    preview = start_local_experiment(**arguments)
    assert preview["executed"] is False
    assert store.get_experiment(experiment_id)["status"] == "planned"
    started = start_local_experiment(**arguments, execute=True)
    attempt_token = store.get_experiment(experiment_id)["token"]
    repeated = start_local_experiment(**arguments, execute=True)
    assert repeated["invocation_id"] == started["invocation_id"]
    assert store.get_experiment(experiment_id)["token"] == attempt_token
    callbacks = []

    def writeback(context):
        callbacks.append("writeback")
        assert store.verify_record(store.get_experiment(experiment_id))
        digest = context["effect_receipt_digest"]
        refreshed = cli(
            "refresh-state",
            "--goal-id",
            goal_id,
            "--todo-id",
            todo_id,
            "--agent-id",
            agent_id,
            "--turn-instance-id",
            turn_id,
            "--classification",
            "local_experiment_validated",
            "--delivery-outcome",
            "outcome_progress",
            "--delivery-boundary",
            "in_flight_continuation",
            "--recommended-action",
            f"Validated local experiment; external receipt {digest}",
            "--no-global-sync",
        )
        assert refreshed.get("settlement_identity") == context["settlement_identity"], (
            refreshed
        )
        persisted = json.loads(Path(refreshed["json_path"]).read_text())
        assert digest in persisted["recommended_action"]
        assert persisted["settlement_identity"] == context["settlement_identity"]
        return {**refreshed, "effect_receipt_digest": digest}

    def spend(context):
        callbacks.append("spend")
        spent = cli(
            "quota",
            "spend-slot",
            "--goal-id",
            goal_id,
            "--agent-id",
            agent_id,
            "--todo-id",
            todo_id,
            "--turn-instance-id",
            turn_id,
            "--source",
            "heartbeat",
            "--execute",
            "--scan-path",
            str(project),
            "--runtime-profile",
            "generic_cli",
        )
        assert spent.get("settlement_identity") == context["settlement_identity"], spent
        assert spent["quota_event"]["after"]["spent_slots"] == 1
        return spent

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        settled = reconcile_local_experiment(
            run_dir=tmp_path / "governed",
            invocation_id=started["invocation_id"],
            root=store.root,
            writeback=writeback,
            spend=spend,
        )
        if settled["status"] != "running":
            break
        time.sleep(0.02)
    assert settled["status"] == "committed", settled
    assert callbacks == ["writeback", "spend"]
    replay = reconcile_local_experiment(
        run_dir=tmp_path / "governed",
        invocation_id=started["invocation_id"],
        root=store.root,
        writeback=writeback,
        spend=spend,
    )
    assert replay["status"] == "committed"
    assert callbacks == ["writeback", "spend"]
    completed = cli(
        "todo",
        "complete",
        "--goal-id",
        goal_id,
        "--todo-id",
        todo_id,
        "--claimed-by",
        agent_id,
        "--agent-id",
        agent_id,
        "--turn-instance-id",
        turn_id,
        "--evidence",
        "Validated local baseline and settled its exact provider receipt.",
        "--no-follow-up",
    )
    assert completed["completed"] is True
