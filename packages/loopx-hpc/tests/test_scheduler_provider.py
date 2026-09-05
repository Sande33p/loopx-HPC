"""Governed scheduler boundaries: real LoopX control plane, fixture native CLIs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from loopx_hpc import provider
from loopx_hpc.bridge import (
    _environment,
    reconcile_scheduler_experiment,
    start_scheduler_experiment,
)
from loopx_hpc.campaign import CampaignStore, digest
from loopx_hpc.cli import demo_spec
from loopx_hpc.scheduler_execution import validate_profile
import loopx_hpc.scheduler_execution as execution


PACKAGE = Path(__file__).resolve().parents[1]


def campaign(tmp_path):
    store = CampaignStore(tmp_path / "campaign")
    store.initialize(demo_spec())
    record = store.add_experiment({"x": 0}, rationale="Measure a fixture baseline")
    profile = {
        "scheduler": "slurm",
        "resources": {"job_name": "fixture", "nodes": 1, "walltime": "00:01:00"},
        "environment": {"name": "private-site-recipe", "configured": True},
        "python": sys.executable,
        "launcher": [],
    }
    profile_path = tmp_path / "site-profile.json"
    profile_path.write_text(json.dumps(profile))
    return store, record["id"], profile_path, digest(validate_profile(profile))


def request(experiment_id, profile_digest, manifest_digest, *, phase="start"):
    return {
        "schema_version": provider.REQUEST_SCHEMA,
        "invocation_id": "capability-" + "a" * 24,
        "capability_id": provider.CAPABILITY_ID,
        "operation": "run_scheduler",
        "input": {
            "experiment_id": experiment_id,
            "profile_digest": profile_digest,
            "manifest_digest": manifest_digest,
        },
        "goal": {"goal_id": "scheduler-provider-fixture"},
        "authority": {
            "effect_id": "scheduler-effect-1",
            "goal_id": "scheduler-provider-fixture",
        },
        "lifecycle": {"phase": phase, "idempotency_key": "scheduler-effect-1"},
    }


def test_observe_does_not_read_scheduler_profile_or_invoke_cli(tmp_path, monkeypatch):
    store, experiment_id, _, _ = campaign(tmp_path)
    before = store.status()

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only observe reached the native scheduler")

    monkeypatch.setattr(execution, "_run_cli", forbidden)
    observed = provider.handle_request(
        {
            "schema_version": provider.REQUEST_SCHEMA,
            "invocation_id": "capability-" + "a" * 24,
            "capability_id": provider.CAPABILITY_ID,
            "operation": "observe",
            "input": {"experiment_id": experiment_id},
        },
        root=store.root,
        scheduler_profile=tmp_path / "nonexistent-profile.json",
    )
    assert observed["effect_receipt"] is None
    assert observed["observations"][0]["experiment_status"] == "planned"
    assert store.status() == before


@pytest.mark.parametrize(
    "change",
    [
        "missing_profile",
        "digest_mismatch",
        "bad_digest",
        "missing_authority",
        "effect_mismatch",
        "phase",
        "injected_profile",
        "missing_manifest",
        "invalid_manifest",
    ],
)
def test_bad_scheduler_admission_inputs_make_zero_native_calls(
    tmp_path, monkeypatch, change
):
    store, experiment_id, profile_path, profile_digest = campaign(tmp_path)
    incoming = request(experiment_id, profile_digest, digest(store.manifest()))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request invoked native scheduler")

    monkeypatch.setattr(execution, "_run_cli", forbidden)
    if change == "missing_profile":
        profile_path = None
    elif change == "digest_mismatch":
        incoming["input"]["profile_digest"] = "0" * 64
    elif change == "bad_digest":
        incoming["input"]["profile_digest"] = "a-site-name"
    elif change == "missing_authority":
        del incoming["authority"]
    elif change == "effect_mismatch":
        incoming["authority"]["effect_id"] = "different-effect"
    elif change == "phase":
        incoming["lifecycle"]["phase"] = "cancel"
    elif change == "missing_manifest":
        del incoming["input"]["manifest_digest"]
    elif change == "invalid_manifest":
        incoming["input"]["manifest_digest"] = "not-a-digest"
    else:
        incoming["input"]["profile"] = {"command": ["arbitrary"]}
    with pytest.raises(ValueError):
        provider.handle_request(
            incoming, root=store.root, scheduler_profile=profile_path
        )
    assert store.get_experiment(experiment_id)["status"] == "planned"
    assert not (store.root / "runs").exists()


def colliding_study(store, tmp_path):
    other = CampaignStore(tmp_path / "other-study")
    manifest = store.manifest()
    manifest["provenance"]["dataset_revision"] = "different-frozen-dataset"
    other.initialize(manifest)
    record = other.add_experiment({"x": 0}, rationale="A distinct source study")
    return other, record


@pytest.mark.parametrize("phase", ["start", "reconcile"])
def test_provider_rejects_root_redirect_despite_colliding_experiment_id(
    tmp_path, monkeypatch, phase
):
    store, experiment_id, profile_path, profile_digest = campaign(tmp_path)
    other, duplicate = colliding_study(store, tmp_path)
    assert duplicate["id"] == experiment_id
    assert digest(other.manifest()) != digest(store.manifest())

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong source reached native scheduler")

    monkeypatch.setattr(execution, "_run_cli", forbidden)
    incoming = request(
        experiment_id, profile_digest, digest(store.manifest()), phase=phase
    )
    with pytest.raises(ValueError, match="source manifest"):
        provider.handle_request(
            incoming, root=other.root, scheduler_profile=profile_path
        )
    assert other.get_experiment(experiment_id)["status"] == "planned"
    assert not (other.root / "runs").exists()


def test_bridge_checks_callers_expected_study_before_upstream_dispatch(
    tmp_path, monkeypatch
):
    store, experiment_id, profile_path, _ = campaign(tmp_path)
    other, duplicate = colliding_study(store, tmp_path)
    assert duplicate["id"] == experiment_id

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong source reached upstream admission")

    monkeypatch.setattr(
        "loopx.extensions.governed_capability_execution.start_governed_external_capability",
        forbidden,
    )
    with pytest.raises(ValueError, match="source manifest"):
        start_scheduler_experiment(
            state_file=tmp_path / "extensions.json",
            run_dir=tmp_path / "governed",
            registry_path=tmp_path / "registry.json",
            goal_id="fixture-goal",
            agent_id="fixture-agent",
            todo_id="fixture-todo",
            turn_instance_id="fixture-turn",
            root=other.root,
            experiment_id=experiment_id,
            scheduler_profile=profile_path,
            expected_manifest_digest=digest(store.manifest()),
            context_refs=[],
            admission={"selected_todo": {"target_key": f"experiment:{experiment_id}"}},
            execute=True,
        )
    assert other.get_experiment(experiment_id)["status"] == "planned"


def test_scheduler_environment_does_not_inherit_provider_or_allocation_secrets(
    tmp_path, monkeypatch
):
    _, _, profile_path, _ = campaign(tmp_path)
    monkeypatch.setenv("EXAMPLE_API_KEY", "fixture-credential")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "fixture-tracker-credential")
    monkeypatch.setenv("SLURM_JOB_ID", "stale-allocation")
    monkeypatch.setenv("SBATCH_ACCOUNT", "unreviewed-account")
    environment = _environment(tmp_path, profile_path)
    assert {
        "EXAMPLE_API_KEY",
        "MLFLOW_TRACKING_TOKEN",
        "SLURM_JOB_ID",
        "SBATCH_ACCOUNT",
    }.isdisjoint(environment)
    assert environment["LOOPX_HPC_SCHEDULER_PROFILE"] == str(profile_path.resolve())


def governed_context(tmp_path, experiment_id, operations):
    from loopx.control_plane.testing.canary_harness import (
        run_json_cli,
        write_fixture_registry,
    )
    from loopx.extensions.capability_admission import bind_external_capability_to_goal
    from loopx.extensions.runtime import install_extension

    project = tmp_path / "project"
    project.mkdir()
    goal_id, agent_id, turn_id = "scheduler-acceptance", "site-host", "site-turn-1"
    document = project / ".codex" / "goals" / goal_id / "ACTIVE_GOAL_STATE.md"
    document.parent.mkdir(parents=True)
    document.write_text(
        "---\nstatus: active\n---\n\n# Active Goal State\n\n## Objective\n\nValidate one bounded scheduler experiment.\n\n## Agent Todo\n"
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

    added = cli(
        "todo",
        "add",
        "--goal-id",
        goal_id,
        "--role",
        "agent",
        "--text",
        "Execute and validate one scheduler fixture.",
        "--action-kind",
        "run_experiment",
        "--target-key",
        f"experiment:{experiment_id}",
    )
    todo_id = added["todo_id"]
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
    extensions = tmp_path / "extensions.json"
    installed = install_extension(
        PACKAGE / "extension.toml", state_file=extensions, execute=True
    )
    assert installed["doctor"]["verified"] is True
    bind_external_capability_to_goal(
        registry_path=registry,
        state_file=extensions,
        goal_id=goal_id,
        capability_id=provider.CAPABILITY_ID,
        operations=operations,
        execute=True,
    )
    arguments = {
        "state_file": extensions,
        "run_dir": tmp_path / "governed",
        "registry_path": registry,
        "goal_id": goal_id,
        "agent_id": agent_id,
        "todo_id": todo_id,
        "turn_instance_id": turn_id,
        "experiment_id": experiment_id,
        "context_refs": [
            {
                "kind": "experiment",
                "ref": f"experiment:{experiment_id}",
                "digest": "sha256:" + "b" * 64,
            }
        ],
        "admission": admission,
    }
    return arguments, cli, project


def fixture_native_clis(tmp_path, monkeypatch):
    directory = tmp_path / "native-bin"
    directory.mkdir()
    calls, native_state = tmp_path / "native-calls.jsonl", tmp_path / "native-state.txt"
    native_state.write_text("RUNNING")
    body = f"""#!{sys.executable}
import json,pathlib,sys
name = pathlib.Path(sys.argv[0]).name
with pathlib.Path({str(calls)!r}).open('a') as handle:
    handle.write(json.dumps({{'name': name}}) + '\\n')
if name == 'sbatch':
    print('417')
elif name == 'sacct':
    print('417|' + pathlib.Path({str(native_state)!r}).read_text() + '|0:0')
else:
    raise SystemExit(0)
"""
    for name in ("sbatch", "sacct", "scancel"):
        executable = directory / name
        executable.write_text(body)
        executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    return calls, native_state


def test_local_only_goal_binding_cannot_dispatch_scheduler_operation(
    tmp_path, monkeypatch
):
    store, experiment_id, profile_path, _ = campaign(tmp_path)
    calls, _ = fixture_native_clis(tmp_path, monkeypatch)
    arguments, _, _ = governed_context(
        tmp_path, experiment_id, ["observe", "run_local"]
    )
    with pytest.raises(ValueError, match="operation|binding|allowed"):
        start_scheduler_experiment(
            **arguments,
            root=store.root,
            scheduler_profile=profile_path,
            expected_manifest_digest=digest(store.manifest()),
            execute=True,
        )
    assert not calls.exists()
    assert store.get_experiment(experiment_id)["status"] == "planned"


def test_governed_scheduler_real_admission_worker_settlement_and_replay(
    tmp_path, monkeypatch
):
    from loopx.extensions.capability_admission import invoke_external_capability

    store, experiment_id, profile_path, _ = campaign(tmp_path)
    calls, native_state = fixture_native_clis(tmp_path, monkeypatch)
    arguments, cli, project = governed_context(
        tmp_path, experiment_id, ["observe", "run_scheduler"]
    )
    arguments.update(
        root=store.root,
        scheduler_profile=profile_path,
        expected_manifest_digest=digest(store.manifest()),
    )
    # A scheduler-scoped Goal binding still does not turn read-only invocation
    # into execution authority. Only the admitted governed route may dispatch.
    with pytest.raises(ValueError, match="governed Turn"):
        invoke_external_capability(
            state_file=arguments["state_file"],
            registry_path=arguments["registry_path"],
            goal_id=arguments["goal_id"],
            capability_id=provider.CAPABILITY_ID,
            operation="run_scheduler",
            provider_input={
                "context_refs": arguments["context_refs"],
                "input": {
                    "experiment_id": experiment_id,
                    "profile_digest": digest(
                        validate_profile(json.loads(profile_path.read_text()))
                    ),
                    "manifest_digest": digest(store.manifest()),
                },
            },
            execute=True,
            environment=_environment(store.root, profile_path),
        )
    assert start_scheduler_experiment(**arguments)["executed"] is False
    assert not calls.exists()
    started = start_scheduler_experiment(**arguments, execute=True)
    repeated = start_scheduler_experiment(**arguments, execute=True)
    assert repeated["invocation_id"] == started["invocation_id"]
    record = store.get_experiment(experiment_id)
    directory = store.root / "runs" / experiment_id / record["token"]
    worker = subprocess.run(
        [
            sys.executable,
            "-m",
            "loopx_hpc.batch_worker",
            str(directory / "packet.json"),
        ],
        env={**os.environ, "SLURM_JOB_ID": "417"},
        cwd=store.root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert worker.returncode == 0, worker.stderr
    native_state.write_text("COMPLETED")
    # Recovery uses the immutable admitted attempt; it must not depend on a
    # mutable staging profile still occupying its original path.
    profile_path.rename(profile_path.with_name("archived-profile.json"))
    callbacks = []
    goal_id, agent_id = arguments["goal_id"], arguments["agent_id"]
    todo_id, turn_id = arguments["todo_id"], arguments["turn_instance_id"]

    def writeback(context):
        callbacks.append("writeback")
        assert store.verify_record(store.get_experiment(experiment_id))
        receipt_digest = context["effect_receipt_digest"]
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
            "scheduler_experiment_validated",
            "--delivery-outcome",
            "outcome_progress",
            "--delivery-boundary",
            "in_flight_continuation",
            "--recommended-action",
            f"Validated experiment receipt {receipt_digest}",
            "--no-global-sync",
        )
        assert refreshed["settlement_identity"] == context["settlement_identity"]
        return {**refreshed, "effect_receipt_digest": receipt_digest}

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
        assert spent["settlement_identity"] == context["settlement_identity"]
        assert spent["quota_event"]["after"]["spent_slots"] == 1
        return spent

    reconciliation = {
        "run_dir": arguments["run_dir"],
        "invocation_id": started["invocation_id"],
        "root": store.root,
        "scheduler_profile": profile_path,
        "writeback": writeback,
        "spend": spend,
    }
    settled = reconcile_scheduler_experiment(**reconciliation)
    assert settled["status"] == "committed", settled
    assert callbacks == ["writeback", "spend"]
    assert reconcile_scheduler_experiment(**reconciliation)["status"] == "committed"
    assert callbacks == ["writeback", "spend"]
    native_calls = [json.loads(line)["name"] for line in calls.read_text().splitlines()]
    assert native_calls == ["sbatch", "sacct"]
    # Check the actual provider payload, not the private outer runtime journal paths.
    journal = json.loads(next(arguments["run_dir"].glob("*.json")).read_text())
    public_result = json.dumps(journal["provider_result"])
    assert journal["provider_result"]["observations"][0]["manifest_digest"] == (
        "sha256:" + digest(store.manifest())
    )
    for private in (
        str(tmp_path),
        "private-site-recipe",
        '"job_id"',
        '"profile"',
        sys.executable,
    ):
        assert private not in public_result
