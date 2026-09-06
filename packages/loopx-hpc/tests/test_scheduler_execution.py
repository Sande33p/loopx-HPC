"""Native command doubles with real batch scripts, workers, and result ingestion."""

import json
import os
import subprocess
import sys

import pytest

import loopx_hpc.scheduler_execution as execution
from loopx_hpc.campaign import CampaignStore
from loopx_hpc.cli import demo_spec, main
from loopx_hpc.local import LocalExecutor
from loopx_hpc.scheduler_execution import SchedulerExecutor


def profile(backend):
    return {
        "scheduler": backend,
        "resources": {"job_name": "test", "nodes": 1, "walltime": "00:01:00"},
        "environment": {"name": "test", "configured": True},
        "python": sys.executable,
        "launcher": [],
    }


class SchedulerDouble:
    def __init__(self, backend):
        self.backend = backend
        self.job_id = "123.server" if backend == "pbs" else "123"
        self.state = "queued"
        self.submissions = 0
        self.directory = None
        self.exit_code = 0

    def __call__(self, argv, cwd):
        if argv[0] in {"qsub", "sbatch"}:
            self.submissions += 1
            self.directory = cwd
            return subprocess.CompletedProcess(argv, 0, self.job_id + "\n", "")
        if argv[0] in {"qdel", "scancel"}:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if self.backend == "pbs":
            job = {
                "job_state": {"queued": "Q", "running": "R", "terminal": "F"}[
                    self.state
                ]
            }
            if self.state == "terminal":
                job["Exit_status"] = self.exit_code
            output = json.dumps({"Jobs": {self.job_id: job}})
        else:
            state = {
                "queued": "PENDING",
                "running": "RUNNING",
                "terminal": "COMPLETED",
            }[self.state]
            output = f"{self.job_id}|{state}|{self.exit_code}:0\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    def work(self):
        environment = {
            **os.environ,
            ("PBS_JOBID" if self.backend == "pbs" else "SLURM_JOB_ID"): self.job_id,
        }
        result = subprocess.run(
            ["/bin/bash", str(self.directory / "job.sh")],
            env=environment,
            cwd=self.directory,
            capture_output=True,
            timeout=20,
        )
        self.exit_code = result.returncode
        self.state = "terminal"
        return result


def test_native_cli_captures_bounded_stderr(tmp_path):
    result = execution._run_cli(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('accepted\\n'); "
            "sys.stderr.write('private diagnostic\\n')",
        ],
        tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout == "accepted\n"
    assert result.stderr == "private diagnostic\n"


def test_native_cli_rejects_oversized_stderr(tmp_path):
    with pytest.raises(ValueError, match="size limit"):
        execution._run_cli(
            [sys.executable, "-c", "import sys; sys.stderr.write('x' * 1_000_001)"],
            tmp_path,
        )


@pytest.mark.parametrize("backend", ["pbs", "slurm"])
def test_real_batch_worker_lifecycle_and_replay(tmp_path, monkeypatch, backend):
    store = CampaignStore(tmp_path / "study")
    store.initialize(demo_spec())
    experiment_id = store.add_experiment({"x": 2}, rationale="test midpoint")["id"]
    executor = SchedulerExecutor(store)
    fake = SchedulerDouble(backend)
    monkeypatch.setattr("loopx_hpc.scheduler_execution._run_cli", fake)
    before = store.status()
    assert executor.submit(experiment_id, profile(backend))["preview"]
    assert before == store.status()
    assert not (store.root / "runs").exists()
    submitted = executor.submit(experiment_id, profile(backend), execute=True)
    assert submitted["status"] == "submitting"
    assert fake.submissions == 1
    assert executor.reconcile(experiment_id)["status"] == "submitting"
    frozen = store.context()["context_digest"]
    assert executor.reconcile(experiment_id)["status"] == "submitting"
    assert (
        store.context()["context_digest"] == frozen
    )  # Quiet observations don't stale plans.
    fake.state = "running"
    assert executor.reconcile(experiment_id)["status"] == "running"
    with pytest.raises(ValueError, match="SchedulerExecutor"):
        LocalExecutor(store).reconcile(experiment_id)
    result = fake.work()
    assert result.returncode == 0, result.stderr.decode()
    # Compute side cannot mark SQLite succeeded before controller accounting.
    assert store.get_experiment(experiment_id)["status"] == "running"
    completed = executor.reconcile(experiment_id)
    assert completed["status"] == "succeeded"
    assert completed["metrics"] == {"loss": 0}
    assert store.verify_record(completed)
    assert executor.submit(experiment_id, profile(backend), execute=True) == completed
    assert executor.reconcile(experiment_id) == completed
    assert fake.submissions == 1
    receipt = json.loads((fake.directory / "receipt.json").read_text())
    assert receipt["job_id"] == fake.job_id
    assert receipt["started_at"] <= receipt["finished_at"]
    assert (
        fake.work().returncode == 75
    )  # A duplicate batch delivery cannot rerun payload.


@pytest.mark.parametrize("backend", ["pbs", "slurm"])
def test_worker_nonzero_exit_is_failed_not_scientific_success(
    tmp_path, monkeypatch, backend
):
    spec = demo_spec()
    spec["command"] = [
        sys.executable,
        "-c",
        "raise SystemExit(3)",
        "{config}",
        "{result}",
    ]
    store = CampaignStore(tmp_path)
    store.initialize(spec)
    experiment_id = store.add_experiment({"x": 0}, rationale="expected failure")["id"]
    fake = SchedulerDouble(backend)
    monkeypatch.setattr("loopx_hpc.scheduler_execution._run_cli", fake)
    executor = SchedulerExecutor(store)
    executor.submit(experiment_id, profile(backend), execute=True)
    assert fake.work().returncode == 1
    completed = executor.reconcile(experiment_id)
    assert completed["status"] == "failed"
    assert completed["metrics"] == {}
    assert not store.context()["eligible_evidence_ids"]


def test_cli_dispatches_scheduler_and_reconcile(tmp_path, monkeypatch, capsys):
    store = CampaignStore(tmp_path / "study")
    store.initialize(demo_spec())
    experiment_id = store.add_experiment({"x": 0}, rationale="CLI")["id"]
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile("pbs")))
    fake = SchedulerDouble("pbs")
    monkeypatch.setattr("loopx_hpc.scheduler_execution._run_cli", fake)
    base = ["--root", str(store.root)]
    assert (
        main(
            base
            + [
                "scheduler-submit",
                "--experiment",
                experiment_id,
                "--profile",
                str(profile_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["preview"]
    assert fake.submissions == 0
    assert (
        main(
            base
            + [
                "scheduler-submit",
                "--experiment",
                experiment_id,
                "--profile",
                str(profile_path),
                "--execute",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert fake.work().returncode == 0
    assert main(base + ["reconcile"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["experiments"][0]["status"] == "succeeded"
    )


def test_scheduler_script_disables_reruns(tmp_path):
    store = CampaignStore(tmp_path)
    store.initialize(demo_spec())
    experiment_id = store.add_experiment({"x": 0}, rationale="test")["id"]
    for backend, directive in (("pbs", "#PBS -r n"), ("slurm", "#SBATCH --no-requeue")):
        script = SchedulerExecutor(store).submit(experiment_id, profile(backend))[
            "script"
        ]
        assert directive in script
        assert script.index(directive) < script.index("set -euo pipefail")
        assert "Preview only" not in script
