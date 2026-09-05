"""Cancellation admission needs fresh native identity; all CLI calls are doubles."""

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import loopx_hpc.scheduler_execution as execution
from test_scheduler_safety import profile, study


class CancelCLI:
    def __init__(self, backend):
        self.backend = backend
        self.job_id = "417.server" if backend == "pbs" else "417"
        self.name = "unset"
        self.output = None
        self.code = 0
        self.cancel_code = 0
        self.calls = []
        self.before_proof = None

    def proof(self):
        if self.backend == "pbs":
            return json.dumps(
                {"Jobs": {self.job_id: {"Job_Name": self.name, "job_state": "R"}}}
            )
        return f"{self.job_id}|{self.name}\n"

    def __call__(self, argv, cwd):
        self.calls.append(list(argv))
        if argv[0] in {"qsub", "sbatch"}:
            return subprocess.CompletedProcess(argv, 0, self.job_id + "\n", "")
        if argv[0] in {"qdel", "scancel"}:
            return subprocess.CompletedProcess(argv, self.cancel_code, "", "")
        assert argv[0] in {"qstat", "squeue"}, argv
        if self.before_proof:
            self.before_proof()
        if isinstance(self.output, BaseException):
            raise self.output
        return subprocess.CompletedProcess(
            argv, self.code, self.proof() if self.output is None else self.output, ""
        )


def submitted(tmp_path, monkeypatch, backend):
    store, first, _ = study(tmp_path)
    fake = CancelCLI(backend)
    monkeypatch.setattr(execution, "_run_cli", fake)
    executor = execution.SchedulerExecutor(store)
    record = executor.submit(first["id"], profile(backend), execute=True)
    fake.name = "lx-" + record["token"][:12]
    return store, first["id"], executor, fake


def intents(store):
    with store.transaction() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM events WHERE kind='scheduler_cancel_requested'"
        ).fetchone()[0]


@pytest.mark.parametrize("backend", ["pbs", "slurm"])
def test_valid_exact_active_identity_allows_one_cancel_and_durable_replay(
    tmp_path, monkeypatch, backend
):
    store, experiment, executor, fake = submitted(tmp_path, monkeypatch, backend)
    before = store.context()["context_digest"]
    assert executor.cancel(experiment)["preview"]
    assert len(fake.calls) == 1
    assert store.context()["context_digest"] == before
    result = executor.cancel(experiment, execute=True)
    assert result["acknowledged"] and not result["terminal_confirmed"]
    expected_query = (
        ["qstat", "-f", "-F", "json", fake.job_id]
        if backend == "pbs"
        else ["squeue", "--noheader", "--jobs", fake.job_id, "--format=%i|%j"]
    )
    assert fake.calls[1] == expected_query
    assert fake.calls[2] == ["qdel" if backend == "pbs" else "scancel", fake.job_id]
    assert intents(store) == 1
    assert store.get_experiment(experiment)["status"] == "submitting"
    fake.output = "unavailable after first cancellation"
    assert executor.cancel(experiment, execute=True)["replayed"]
    assert len(fake.calls) == 3


@pytest.mark.parametrize("backend", ["pbs", "slurm"])
def test_recycled_id_wrong_name_refuses_without_intent_and_fresh_retry_is_allowed(
    tmp_path, monkeypatch, backend
):
    store, experiment, executor, fake = submitted(tmp_path, monkeypatch, backend)
    expected = fake.name
    fake.name = "some-other-study"
    before = store.context()["context_digest"]
    with pytest.raises(ValueError, match="identity unresolved"):
        executor.cancel(experiment, execute=True)
    assert intents(store) == 0
    assert store.context()["context_digest"] == before
    assert all(call[0] not in {"qdel", "scancel"} for call in fake.calls)
    fake.name = expected
    assert executor.cancel(experiment, execute=True)["acknowledged"]
    assert intents(store) == 1


@pytest.mark.parametrize("backend", ["pbs", "slurm"])
@pytest.mark.parametrize(
    "failure",
    [
        "empty",
        "wrong_id",
        "multiple",
        "missing_name",
        "query_failure",
        "exception",
        "malformed",
    ],
)
def test_unresolved_native_identity_never_cancels_or_records_intent(
    tmp_path, monkeypatch, backend, failure
):
    store, experiment, executor, fake = submitted(tmp_path, monkeypatch, backend)
    if failure == "empty":
        fake.output = ""
    elif failure == "wrong_id":
        fake.output = fake.proof().replace(
            fake.job_id, "999.server" if backend == "pbs" else "999"
        )
    elif failure == "multiple":
        fake.output = (
            json.dumps(
                {
                    "Jobs": {
                        fake.job_id: {"job_state": "R", "Job_Name": fake.name},
                        "999.server": {"job_state": "R", "Job_Name": fake.name},
                    }
                }
            )
            if backend == "pbs"
            else fake.proof() + fake.proof()
        )
    elif failure == "missing_name":
        fake.output = (
            json.dumps({"Jobs": {fake.job_id: {"job_state": "R"}}})
            if backend == "pbs"
            else fake.job_id + "|\n"
        )
    elif failure == "query_failure":
        fake.code = 1
    elif failure == "exception":
        fake.output = subprocess.TimeoutExpired(["query"], 30)
    elif failure == "malformed":
        fake.output = (
            '{"Jobs": [' if backend == "pbs" else fake.proof().strip() + "|extra\n"
        )
    before = store.context()["context_digest"]
    with pytest.raises(ValueError, match="identity unresolved"):
        executor.cancel(experiment, execute=True)
    assert intents(store) == 0
    assert store.context()["context_digest"] == before
    assert all(call[0] not in {"qdel", "scancel"} for call in fake.calls)


@pytest.mark.parametrize("state", ["F", "X", "M", "?", None])
def test_pbs_terminal_or_unknown_state_is_not_active_cancel_identity(
    tmp_path, monkeypatch, state
):
    store, experiment, executor, fake = submitted(tmp_path, monkeypatch, "pbs")
    fake.output = json.dumps(
        {
            "Jobs": {
                fake.job_id: {
                    "Job_Name": fake.name,
                    "job_state": state,
                    "Exit_status": 0,
                }
            }
        }
    )
    with pytest.raises(ValueError, match="identity unresolved"):
        executor.cancel(experiment, execute=True)
    assert intents(store) == 0
    assert all(call[0] != "qdel" for call in fake.calls)


def test_duplicate_pbs_name_keys_cannot_override_wrong_identity(tmp_path, monkeypatch):
    store, experiment, executor, fake = submitted(tmp_path, monkeypatch, "pbs")
    fake.output = (
        '{"Jobs":{"417.server":{"job_state":"R","Job_Name":"other","Job_Name":'
        + json.dumps(fake.name)
        + "}}}"
    )
    with pytest.raises(ValueError, match="identity unresolved"):
        executor.cancel(experiment, execute=True)
    assert intents(store) == 0
    assert all(call[0] != "qdel" for call in fake.calls)


def test_cancel_command_failure_is_not_retried_after_admitted_intent(
    tmp_path, monkeypatch
):
    store, experiment, executor, fake = submitted(tmp_path, monkeypatch, "slurm")
    fake.cancel_code = 1
    result = executor.cancel(experiment, execute=True)
    assert not result["acknowledged"]
    assert result["cancel_requested"]
    assert executor.cancel(experiment, execute=True)["replayed"]
    assert len(fake.calls) == 3
    assert intents(store) == 1


def test_concurrent_cancel_callers_have_one_native_proof_and_one_intent(
    tmp_path, monkeypatch
):
    store, experiment, executor, fake = submitted(tmp_path, monkeypatch, "slurm")
    entered = threading.Event()
    release = threading.Event()

    def pause():
        entered.set()
        assert release.wait(5)

    fake.before_proof = pause
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.cancel, experiment, execute=True)
        assert entered.wait(5)
        second = pool.submit(executor.cancel, experiment, execute=True)
        release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert sum(bool(result.get("replayed")) for result in results) == 1
    assert [call[0] for call in fake.calls] == ["sbatch", "squeue", "scancel"]
    assert intents(store) == 1
