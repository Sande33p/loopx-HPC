"""Local-only recipe/rank tests. No MPI, GPU, qsub, sbatch, SSH or site access."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from loopx_hpc import aurora
from loopx_hpc.aurora import (
    aurora_profile,
    launch_argv,
    rank_environment,
    slurm_profile,
)
from loopx_hpc.scheduler_execution import validate_profile
from loopx_hpc.schedulers import render_script


def configured(**changes):
    values = {
        "account": "example-project",
        "queue": "debug",
        "python": "/shared/venv/bin/python",
        "working_directory": "/shared/project",
        "modules": ["frameworks/2025.3.1"],
        "filesystems": ["home", "flare"],
        "configured": True,
    }
    values.update(changes)
    return aurora_profile(**values)


def test_template_is_inert_public_input_and_exact_executor_schema():
    path = Path(__file__).parents[1] / "examples" / "aurora-profile-input.json"
    template = json.loads(path.read_text())
    profile = aurora_profile(**template)
    assert profile == validate_profile(profile)
    assert profile == aurora_profile()
    assert not profile["environment"]["configured"]
    with pytest.raises(ValueError, match="unconfigured"):
        render_script(
            "pbs",
            [profile["python"], "-m", "loopx_hpc.batch_worker"],
            profile["resources"],
            profile["environment"],
        )
    with pytest.raises(ValueError, match="configured"):
        aurora_profile(**{**template, "configured": True})


def test_aurora_profile_launches_only_workload_ranks_and_matches_cpu_gpu_topology():
    profile = configured(nodes=2)
    assert profile == validate_profile(profile)
    assert "system" not in profile["resources"]
    assert profile["resources"]["nodes"] == 2
    assert profile["resources"]["place"] == "scatter"
    script = render_script(
        "pbs", ["scientific-program"], profile["resources"], profile["environment"]
    )
    assert "#PBS -l select=2\n" in script
    assert ":system=" not in script
    assert profile["environment"]["login_shell"]
    assert profile["launcher"][:4] == [
        profile["python"],
        "-m",
        "loopx_hpc.aurora",
        "--launch",
    ]
    assert profile["launcher"][-1] == "--"
    argv = launch_argv(
        ["scientific-program"],
        {"PBS_JOBID": "123.aurora-pbs"},
        "n01\nn02\n",
        python=profile["python"],
        nodes=2,
    )
    assert argv[:6] == ["mpiexec", "--envall", "-n", "24", "-ppn", "12"]
    assert (
        argv[9]
        == "list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
    )
    assert "loopx_hpc.batch_worker" not in argv
    assert argv[-2:] == ["--", "scientific-program"]
    assert argv[argv.index("--master-host") + 1] == "n01.hsn.cm.aurora.alcf.anl.gov"
    assert argv[argv.index("--job-id") + 1] == "123.aurora-pbs"
    assert profile["environment"]["variables"]["ZE_FLAT_DEVICE_HIERARCHY"] == "FLAT"
    device = configured(gpu_mode="device", cpus_per_rank=16)
    device_argv = launch_argv(
        ["scientific-program"],
        {"PBS_JOBID": "123.aurora-pbs"},
        "n01\n",
        python=profile["python"],
        nodes=1,
        gpu_mode="device",
        cpus_per_rank=16,
    )
    assert device_argv[:6] == ["mpiexec", "--envall", "-n", "6", "-ppn", "6"]
    assert device_argv[9] == "list:1-16:17-32:33-48:53-68:69-84:85-100"
    assert device["environment"]["variables"]["ZE_FLAT_DEVICE_HIERARCHY"] == "COMPOSITE"


@pytest.mark.parametrize(
    "changes",
    [
        {"nodes": True},
        {"nodes": 0},
        {"nodes": 10625},
        {"configured": "true"},
        {"gpu_mode": "slurm"},
        {"cpus_per_rank": 9},
        {"cpus_per_rank": True},
        {"gpu_mode": "device", "cpus_per_rank": 17},
        {"master_port": 65536},
        {"master_port": True},
        {"rendezvous_network": "arbitrary-domain"},
        {"modules": ["frameworks"]},
        {"modules": []},
        {"filesystems": ["home", "flare", "daos_user_fs"]},
        {"account": "project\n#PBS -q prod"},
        {"queue": "--unsafe"},
        {"python": "/shared/$RUNTIME/bin/python"},
        {"python": "/shared/../python"},
        {"working_directory": "/shared/$(touch marker)"},
        {"variables": {"RANK": "0"}},
        {"variables": {"PBS_NODEFILE": "/other/job"}},
        {"variables": {"ZE_AFFINITY_MASK": "0"}},
        {"variables": {"BASH_ENV": "/bad"}},
    ],
)
def test_profile_refuses_injection_invalid_resources_and_rank_overrides(changes):
    with pytest.raises(ValueError):
        configured(**changes)


def test_builder_does_not_mutate_owner_environment_or_resolve_site():
    variables = {"CCL_PROCESS_LAUNCHER": "none", "CCL_ATL_TRANSPORT": "ofi"}
    modules = ["frameworks/2025.3.1"]
    before = copy.deepcopy((variables, modules))
    profile = configured(variables=variables, modules=modules)
    profile["environment"]["modules"].append("other/1")
    assert (variables, modules) == before
    assert profile["python"] == "/shared/venv/bin/python"


def mpi_env(rank=13, local=1):
    return {
        "PBS_JOBID": "123.aurora-pbs",
        "PALS_RANKID": str(rank),
        "PALS_LOCAL_RANKID": str(local),
    }


def test_rank_mapping_uses_explicit_world_when_pals_size_is_absent():
    values = mpi_env()
    before = values.copy()
    mapped = rank_environment(values, "n01.example\nn02.example\n", nodes=2)
    assert mapped["RANK"] == "13"
    assert mapped["WORLD_SIZE"] == "24"
    assert mapped["LOCAL_RANK"] == "1"
    assert mapped["ZE_AFFINITY_MASK"] == "1"
    assert mapped["MASTER_ADDR"] == "n01.hsn.cm.aurora.alcf.anl.gov"
    assert mapped["MASTER_PORT"] == "29500"
    assert mapped["NODE_RANK"] == "1"
    assert values == before
    device = rank_environment(mpi_env(11, 5), "n01\nn02\n", nodes=2, gpu_mode="device")
    assert device["WORLD_SIZE"] == "12"
    assert device["ZE_FLAT_DEVICE_HIERARCHY"] == "COMPOSITE"
    assert device["ZE_AFFINITY_MASK"] == "5"


def test_hsn_suffix_is_not_duplicated_and_pbs_network_is_explicit():
    hsn = "n01.hsn.cm.aurora.alcf.anl.gov"
    mapped = rank_environment(mpi_env(), hsn + "\nn02\n", nodes=2)
    assert mapped["MASTER_ADDR"] == hsn
    preserved = rank_environment(
        mpi_env(), "n01.example\nn02.example\n", nodes=2, rendezvous_network="pbs"
    )
    assert preserved["MASTER_ADDR"] == "n01.example"


@pytest.mark.parametrize(
    "changes",
    [
        {"PBS_JOBID": ""},
        {"PALS_RANKID": ""},
        {"PALS_LOCAL_RANKID": ""},
        {"PALS_RANKID": "-1"},
        {"PALS_RANKID": "1.0"},
        {"PALS_LOCAL_RANKID": "12"},
        {"PALS_RANKID": "24"},
        {"PALS_LOCAL_RANKID": "2"},
        {"PMI_RANK": "12"},
        {"PALS_SIZE": "12"},
        {"PALS_LOCAL_SIZE": "6"},
        {"RANK": "0"},
        {"WORLD_SIZE": "12"},
        {"MASTER_ADDR": "unrelated.example"},
        {"MASTER_PORT": "12345"},
        {"ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE"},
        {"ZE_AFFINITY_MASK": "0"},
    ],
)
def test_rank_wrapper_rejects_missing_conflicting_or_out_of_range_identity(changes):
    with pytest.raises(ValueError):
        rank_environment({**mpi_env(), **changes}, "n01\nn02\n", nodes=2)


@pytest.mark.parametrize(
    "nodefile",
    [
        "",
        "n01\n",
        "n01\nn02\nn03\n",
        "n01\n$(bad)\n",
        "n01\n\nn02\n",
        "n01\n-bad\n",
        "n01\nn02\x00\n",
    ],
)
def test_rank_wrapper_rejects_wrong_allocation_or_malformed_nodefile(nodefile):
    with pytest.raises(ValueError):
        rank_environment(mpi_env(), nodefile, nodes=2)


def test_real_local_wrapper_preserves_literal_argv_and_child_exit_code(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PALS_", "PMI_", "PMIX_", "PBS_", "SLURM_", "ZE_"))
        and key
        not in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "NODE_RANK",
            "NUM_NODES",
        }
    }
    # The batch nodefile is deliberately not available to this remote-rank
    # simulation. Its master host comes from the one-shot launch bootstrap.
    env.update({**mpi_env(), "PBS_NODEFILE": str(tmp_path / "unavailable-nodefile")})
    literal = "$(touch NOT_EXECUTED); argument with space"
    command = [
        sys.executable,
        "-m",
        "loopx_hpc.aurora",
        "--nodes",
        "2",
        "--master-host",
        "n01",
        "--job-id",
        "123.aurora-pbs",
        "--",
        sys.executable,
        "-c",
        "import os,sys,json; print(json.dumps([os.environ['RANK'],os.environ['WORLD_SIZE'],os.environ['ZE_AFFINITY_MASK'],sys.argv[1]])); sys.exit(7)",
        literal,
    ]
    result = subprocess.run(
        command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == ["13", "24", "1", literal]
    assert not (tmp_path / "NOT_EXECUTED").exists()
    missing = {**env, "PALS_RANKID": ""}
    refused = subprocess.run(
        command, env=missing, cwd=tmp_path, capture_output=True, text=True, timeout=5
    )
    assert refused.returncode == 2
    assert refused.stdout == ""
    wrong_job = subprocess.run(
        command,
        env={**env, "PBS_JOBID": "456.aurora-pbs"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert wrong_job.returncode == 2
    assert wrong_job.stdout == ""


def test_launch_bootstrap_reads_one_nodefile_and_builds_literal_mpi_not_controller(
    tmp_path, monkeypatch
):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("n01\nn02\n")
    monkeypatch.setattr(
        aurora.os,
        "environ",
        {"PBS_JOBID": "123.aurora-pbs", "PBS_NODEFILE": str(nodefile)},
    )
    read = aurora._read_nodefile
    reads = []
    calls = []

    def counted(path):
        reads.append(path)
        return read(path)

    def capture(executable, argv, environment):
        calls.append((executable, argv, environment))
        raise SystemExit(0)

    monkeypatch.setattr(aurora, "_read_nodefile", counted)
    monkeypatch.setattr(aurora.os, "execvpe", capture)
    with pytest.raises(SystemExit) as exit_status:
        aurora.main(
            [
                "--launch",
                "--nodes",
                "2",
                "--",
                "/shared/scientific",
                "literal; not-shell",
            ]
        )
    assert exit_status.value.code == 0
    assert reads == [str(nodefile)]
    assert len(calls) == 1
    executable, argv, _ = calls[0]
    assert executable == "mpiexec"
    assert argv[-2:] == ["/shared/scientific", "literal; not-shell"]
    assert "loopx_hpc.batch_worker" not in argv
    assert "--launch" not in argv


def test_launch_refuses_nested_mpi_wrong_nodes_and_missing_allocation():
    for env, nodefile in [
        (mpi_env(), "n01\nn02\n"),
        ({}, "n01\nn02\n"),
        ({"PBS_JOBID": "123.aurora-pbs"}, "n01\n"),
    ]:
        with pytest.raises(ValueError):
            launch_argv(["program"], env, nodefile, python="/shared/python", nodes=2)


def test_rank_requires_one_master_source_and_rejects_untrusted_hostname_syntax():
    for text, master in [(None, None), ("n01\nn02\n", "n01"), (None, "n01;execute")]:
        with pytest.raises(ValueError):
            rank_environment(mpi_env(), text, nodes=2, master_host=master)


def test_slurm_builder_is_generic_srun_with_no_aurora_topology():
    profile = slurm_profile(
        resources={
            "job_name": "example",
            "nodes": 2,
            "walltime": "00:10:00",
            "ntasks_per_node": 4,
            "cpus_per_task": 2,
        },
        environment={"configured": False},
        python="/shared/venv/bin/python",
    )
    assert profile == validate_profile(profile)
    assert profile["launcher"] == [
        "srun",
        "--kill-on-bad-exit=1",
        "--nodes=2",
        "--ntasks=8",
        "--ntasks-per-node=4",
        "--cpus-per-task=2",
    ]
    assert profile["environment"]["variables"] == {}
    assert not profile["environment"]["login_shell"]
    with pytest.raises(ValueError, match="requires"):
        slurm_profile(
            resources={"job_name": "bad", "nodes": 1, "walltime": "00:01:00"},
            environment={},
            python="/shared/python",
        )
