"""Real local-process regression checks; never launch remote jobs.

Child fixtures self-expire as a second bound and are cleaned up in finally. These
tests establish POSIX group hygiene, not containment against setsid escapes.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time

import pytest

from loopx_hpc.campaign import CampaignStore
from loopx_hpc.cli import demo_spec
from loopx_hpc.local import LocalExecutor

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="local worker uses POSIX process groups"
)


def make_store(tmp_path, body, timeout=3):
    spec = demo_spec()
    spec["command"] = [sys.executable, "-c", body, "{config}", "{result}"]
    spec["limits"]["timeout_seconds"] = timeout
    store = CampaignStore(tmp_path)
    store.initialize(spec)
    record = store.add_experiment({"x": 1}, rationale="Worker safety fixture")
    return store, record


def wait_result(store, experiment_id):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        record = LocalExecutor(store).reconcile(experiment_id)
        if record["status"] in {"succeeded", "failed", "unknown"}:
            return record
        time.sleep(0.025)
    pytest.fail("local safety fixture exceeded its bounded wait")


def cleanup_test_child(path):
    if path.is_file():
        pid = int(path.read_text())
        try:
            # Restrict cleanup to the recorded fixture group, rather than a
            # recycled unrelated PID.
            if os.getpgid(pid) == int(path.with_name("child.pgid").read_text()):
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("parent_exits", [True, False])
def test_exit_and_timeout_drain_term_ignoring_child(tmp_path, parent_exits):
    child = """
import os,signal,time,pathlib
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path('child.pgid').write_text(str(os.getpgrp()))
pathlib.Path('child.pid').write_text(str(os.getpid()))
deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    pathlib.Path('child.tick').write_text(str(time.monotonic_ns()))
    time.sleep(0.025)
"""
    body = f"""
import pathlib,subprocess,sys,time
subprocess.Popen([sys.executable, '-c', {child!r}])
deadline = time.monotonic() + 2
while not pathlib.Path('child.pid').is_file():
    if time.monotonic() > deadline:
        raise RuntimeError('child startup failed')
    time.sleep(0.01)
pathlib.Path(sys.argv[2]).write_text('{{"metrics":{{"loss":0}}}}')
if not {parent_exits!r}:
    time.sleep(8)
"""
    store, record = make_store(tmp_path, body, timeout=1)
    submitted = LocalExecutor(store).submit(record["id"], execute=True)
    child_path = store.root / "runs" / record["id"] / submitted["token"] / "child.pid"
    try:
        finished = wait_result(store, record["id"])
        assert finished["status"] in {"failed", "unknown"}
        assert (
            "descendants_after_command_exit" if parent_exits else "timeout"
        ) in finished["failure"]
        # Inspect only our fixture heartbeat, not the host process list. A dead
        # zombie waiting for host init is not executing this heartbeat either.
        heartbeat = child_path.with_name("child.tick")
        stopped_tick = heartbeat.read_text()
        time.sleep(0.15)
        assert heartbeat.read_text() == stopped_tick
        assert finished["metrics"] == {}
        assert finished["artifacts"] == []
        assert store.context()["eligible_evidence_ids"] == []
        replay = LocalExecutor(store).submit(record["id"], execute=True)
        assert replay["token"] == submitted["token"]
    finally:
        cleanup_test_child(child_path)


def test_plain_timeout_is_failed_and_does_not_retry(tmp_path):
    store, record = make_store(tmp_path, "import time; time.sleep(8)", timeout=1)
    submitted = LocalExecutor(store).submit(record["id"], execute=True)
    finished = wait_result(store, record["id"])
    assert finished["status"] == "failed"
    assert finished["failure"] == "timeout"
    assert (
        LocalExecutor(store).submit(record["id"], execute=True)["token"]
        == submitted["token"]
    )


@pytest.mark.parametrize(
    "result",
    ['{"metrics":{"loss":1,"loss":0}}', '{"metrics":{"loss":1},"metrics":{"loss":0}}'],
)
def test_duplicate_result_keys_are_not_scientific_evidence(tmp_path, result):
    body = f"import pathlib,sys; pathlib.Path(sys.argv[2]).write_text({result!r})"
    store, record = make_store(tmp_path, body)
    LocalExecutor(store).submit(record["id"], execute=True)
    finished = wait_result(store, record["id"])
    assert finished["status"] == "failed"
    assert store.context()["eligible_evidence_ids"] == []


@pytest.mark.parametrize("config", ['{"x":true}', '{"x":1.0}', '{"x":0,"x":1}'])
def test_exact_config_type_and_duplicate_keys_are_rejected(tmp_path, config):
    body = f'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text({config!r}); pathlib.Path(sys.argv[2]).write_text(\'{{"metrics":{{"loss":0}}}}\')'
    store, record = make_store(tmp_path, body)
    LocalExecutor(store).submit(record["id"], execute=True)
    assert wait_result(store, record["id"])["status"] == "failed"
    assert store.context()["eligible_evidence_ids"] == []


def test_valid_artifacts_are_hashed_from_the_validated_bytes(tmp_path):
    body = 'import pathlib,sys; pathlib.Path(sys.argv[2]).write_text(\'{"metrics": {"loss": 0.25}}\\n\')'
    store, record = make_store(tmp_path, body)
    LocalExecutor(store).submit(record["id"], execute=True)
    finished = wait_result(store, record["id"])
    assert finished["status"] == "succeeded"
    assert store.verify_record(finished)
    for artifact in finished["artifacts"]:
        data = (store.root / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        if artifact["path"].endswith("result.json"):
            assert json.loads(data)["metrics"] == finished["metrics"]


def test_concurrent_duplicate_submit_launches_one_command(tmp_path):
    body = "import pathlib,sys,time; f=pathlib.Path('launches.txt'); f.open('a').write('launch\\n'); time.sleep(0.2); pathlib.Path(sys.argv[2]).write_text('{\"metrics\": {\"loss\": 1}}')"
    store, record = make_store(tmp_path, body)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                LocalExecutor(CampaignStore(store.root)).submit,
                record["id"],
                execute=True,
            )
            for _ in range(2)
        ]
        receipts = [future.result(timeout=5) for future in futures]
    assert receipts[0]["token"] == receipts[1]["token"]
    finished = wait_result(store, record["id"])
    assert finished["status"] == "succeeded"
    assert (
        store.root / "runs" / record["id"] / finished["token"] / "launches.txt"
    ).read_text().splitlines() == ["launch"]
    with sqlite3.connect(store.database) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM events WHERE kind='attempt_reserved'"
            ).fetchone()[0]
            == 1
        )
