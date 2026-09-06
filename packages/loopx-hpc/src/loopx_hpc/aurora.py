"""Inert site recipes and an allocation-only Aurora scientific-rank wrapper.

``aurora_profile`` returns the executor contract ``scheduler, resources,
environment, python, launcher``. Only the scientific workload gets ``launcher``;
the batch worker/controller MUST run once, outside mpiexec. Recipes do not probe
a cluster, certify installed modules, authorize submission, or stage data.

The wrapper is intentionally narrower than the PRISM launchers: shared storage,
an already installed shared Python, 12 FLAT tiles or 6 COMPOSITE devices per node.
The batch script uses a trusted site/user login shell for module initialization.
The scientific launch wrapper reads PBS_NODEFILE once on the batch host, then
inherits that module environment via mpiexec --envall without per-rank login
shells that can reset it. No DAOS, node-local staging, restart, SSH or installation.
Masked workloads must select xpu:0; LOCAL_RANK remains the actual local MPI rank.
Only global rank zero may publish the experiment's authoritative result file.

Syntax/topology checked 2026-09-04 against:
https://docs.alcf.anl.gov/aurora/running-jobs-aurora/
https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/
https://slurm.schedmd.com/srun.html
PRISM's web/DAOS launchers informed module-before-runtime ordering and PALS rank
mapping, not accounts, paths, model settings, or claims of hardware acceptance.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from pathlib import PurePosixPath
from typing import Mapping

from .environments import validate_environment
from .schedulers import validate_resources

_UNCONFIGURED = "UNCONFIGURED"
_MAX_NODES = 10_624
_MAX_NODEFILE = 4 * 1024 * 1024
_JOB_ID = re.compile(r"[0-9]+(?:\.[A-Za-z0-9][A-Za-z0-9_.-]*)?\Z")
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")
_RANK_KEYS = ("PALS_RANKID", "PMI_RANK", "PMIX_RANK")
_LOCAL_KEYS = ("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "PMIX_LOCAL_RANK")
_SIZE_KEYS = ("PALS_SIZE", "PMI_SIZE", "PMIX_SIZE")
_LOCAL_SIZE_KEYS = ("PALS_LOCAL_SIZE", "PMI_LOCAL_SIZE", "PMIX_LOCAL_SIZE")
_RUNTIME_KEYS = {
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NUM_NODES",
    "ZE_AFFINITY_MASK",
    "ZE_FLAT_DEVICE_HIERARCHY",
    "ZE_ENABLE_PCI_ID_DEVICE_ORDER",
}


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _path(value: object, name: str, configured: bool) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            ord(char) < 32 or ord(char) == 127 or char in "$`{}<>\\" for char in value
        )
    ):
        raise ValueError(f"{name} must be a literal absolute POSIX path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or value.startswith("//")
        or ".." in path.parts
        or str(path) != value
        or value == "/"
        or (configured and _UNCONFIGURED in path.parts)
    ):
        raise ValueError(f"{name} must be an explicitly configured absolute POSIX path")
    return value


def _rank_shape(gpu_mode: str, cpus_per_rank: int) -> tuple[int, str]:
    if gpu_mode not in ("tile", "device"):
        raise ValueError("gpu_mode must be tile or device")
    ppn = 12 if gpu_mode == "tile" else 6
    depth = _integer(cpus_per_rank, "cpus_per_rank", 1, 96 // ppn)
    # Reserve the documented service cores and keep each rank on one socket.
    half = ppn // 2
    ranges = []
    for local_rank in range(ppn):
        start = (1 if local_rank < half else 53) + (local_rank % half) * depth
        ranges.append(str(start) if depth == 1 else f"{start}-{start + depth - 1}")
    return ppn, "list:" + ":".join(ranges)


def aurora_profile(
    *,
    account: str = _UNCONFIGURED,
    queue: str = _UNCONFIGURED,
    python: str = "/UNCONFIGURED/venv/bin/python",
    working_directory: str = "/UNCONFIGURED/shared-project",
    modules: list[str] | None = None,
    filesystems: list[str] | None = None,
    nodes: int = 1,
    walltime: str = "00:10:00",
    gpu_mode: str = "tile",
    cpus_per_rank: int = 8,
    configured: bool = False,
    variables: dict[str, str] | None = None,
    master_port: int = 29500,
    rendezvous_network: str = "hsn",
    job_name: str = "loopx-aurora",
) -> dict:
    """Build a detached PBS recipe, disabled unless explicitly configured.

    Configured recipes require an account, queue, pinned module identifiers,
    absolute shared runtime/work directory and explicit home/flare filesystems.
    Queue syntax is validated, not current queue limits, permission or capacity.
    ``configured=True`` is an owner declaration, NOT site/hardware verification.
    All ranks use the same explicit rendezvous port inside this allocation.
    ``rendezvous_network='hsn'`` maps the allocation's first short hostname to
    Aurora's documented HSN suffix; 'pbs' preserves the nodefile hostname.
    """
    if type(configured) is not bool:
        raise ValueError("configured must be a boolean")
    nodes = _integer(nodes, "nodes", 1, _MAX_NODES)
    master_port = _integer(master_port, "master_port", 1024, 65535)
    _rendezvous("validation-only", rendezvous_network)
    _rank_shape(gpu_mode, cpus_per_rank)
    python = _path(python, "python", configured)
    directory = _path(working_directory, "working_directory", configured)
    if configured and (account == _UNCONFIGURED or queue == _UNCONFIGURED):
        raise ValueError("account and queue must be explicitly configured")
    if configured and (not modules or filesystems is None):
        raise ValueError("configured Aurora profiles require modules and filesystems")
    resource = validate_resources(
        "pbs",
        {
            "job_name": job_name,
            "nodes": nodes,
            "walltime": walltime,
            "account": account,
            "queue": queue,
            "place": "scatter",
            "filesystems": ["home", "flare"] if filesystems is None else filesystems,
        },
    )
    if set(resource["filesystems"]) - {"home", "flare"}:
        raise ValueError(
            "this shared-filesystem profile supports home/flare only; DAOS needs a separate adapter"
        )
    environment = validate_environment(
        {
            "name": "aurora",
            "configured": configured,
            "modules": [] if modules is None else modules,
            "variables": {} if variables is None else variables,
            "working_directory": directory,
            "login_shell": True,
        }
    )
    if configured and any("/" not in module for module in environment["modules"]):
        raise ValueError("configured Aurora module versions must be explicitly pinned")
    for key in environment["variables"]:
        if key in _RUNTIME_KEYS or key.startswith(
            ("PBS_", "PALS_", "PMI_", "PMIX_", "OMPI_", "SLURM_")
        ):
            raise ValueError(f"allocation/rank variable must not be preset: {key}")
    environment["variables"].update(
        {
            "ZE_FLAT_DEVICE_HIERARCHY": "FLAT" if gpu_mode == "tile" else "COMPOSITE",
            "ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1",
        }
    )
    environment = validate_environment(environment)
    return {
        "scheduler": "pbs",
        "resources": resource,
        "environment": environment,
        "python": python,
        "launcher": [
            python,
            "-m",
            "loopx_hpc.aurora",
            "--launch",
            "--nodes",
            str(nodes),
            "--gpu-mode",
            gpu_mode,
            "--master-port",
            str(master_port),
            "--cpus-per-rank",
            str(cpus_per_rank),
            "--rendezvous-network",
            rendezvous_network,
            "--",
        ],
    }


def slurm_profile(*, resources: dict, environment: dict, python: str) -> dict:
    """Generic srun recipe, NOT an Aurora profile or generic GPU-binding policy.

    Site MPI plugins, accelerator binding and framework rank initialization stay
    explicit workload responsibilities. environment.login_shell defaults false;
    enable explicitly only for trusted site/user startup files. This function
    never calls srun/sbatch.
    """
    checked = validate_resources("slurm", resources)
    env = validate_environment(environment)
    runtime = _path(python, "python", env["configured"])
    ppn = checked.get("ntasks_per_node")
    cpus = checked.get("cpus_per_task")
    if ppn is None or cpus is None:
        raise ValueError("srun profile requires ntasks_per_node and cpus_per_task")
    return {
        "scheduler": "slurm",
        "resources": checked,
        "environment": env,
        "python": runtime,
        "launcher": [
            "srun",
            "--kill-on-bad-exit=1",
            f"--nodes={checked['nodes']}",
            f"--ntasks={checked['nodes'] * ppn}",
            f"--ntasks-per-node={ppn}",
            f"--cpus-per-task={cpus}",
        ],
    }


def _mpi_integer(
    env: Mapping[str, str], keys: tuple[str, ...], *, required: bool
) -> int | None:
    values = []
    for key in keys:
        value = env.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,9}", value):
            raise ValueError(f"malformed MPI variable: {key}")
        values.append(int(value))
    if not values:
        if required:
            raise ValueError(
                "MPI rank identity is missing; refusing a rank-zero fallback"
            )
        return None
    if len(set(values)) != 1:
        raise ValueError("conflicting MPI rank/size variables")
    return values[0]


def _hostname(host: str) -> str:
    if (
        not isinstance(host, str)
        or not host
        or len(host) > 253
        or any(not _HOST_LABEL.fullmatch(label) for label in host.split("."))
    ):
        raise ValueError("PBS allocation contains an invalid hostname")
    return host


def _rendezvous(host: str, network: str) -> str:
    _hostname(host)
    if network not in ("hsn", "pbs"):
        raise ValueError("rendezvous_network must be hsn or pbs")
    # The official Aurora PyTorch DDP example uses this HSN domain. Strip any
    # nodefile suffix first as PRISM does; never append the suffix twice.
    return (
        host.split(".", 1)[0] + ".hsn.cm.aurora.alcf.anl.gov"
        if network == "hsn"
        else host
    )


def _allocation_master(nodefile_text: str, nodes: int) -> str:
    if (
        not isinstance(nodefile_text, str)
        or len(nodefile_text.encode()) > _MAX_NODEFILE
    ):
        raise ValueError("PBS nodefile is malformed or too large")
    hosts = []
    seen = set()
    for host in nodefile_text.splitlines():
        _hostname(host)
        if host not in seen:
            hosts.append(host)
            seen.add(host)
    if len(hosts) != nodes:
        raise ValueError("PBS allocation node count differs from the explicit profile")
    return hosts[0]


def rank_environment(
    inherited: Mapping[str, str],
    nodefile_text: str | None = None,
    *,
    nodes: int,
    gpu_mode: str = "tile",
    master_port: int = 29500,
    master_host: str | None = None,
    rendezvous_network: str = "hsn",
) -> dict[str, str]:
    """Return validated allocation/rank env without mutation or hardware access.

    PBS/PALS environment and the launch-wrapper-provided master_host are trusted
    scheduler inputs, not authentication. Accept either nodefile text (for pure
    preview/tests) or the validated launch host, never both. Remote ranks need
    not have access to the batch host's PBS_NODEFILE.
    """
    nodes = _integer(nodes, "nodes", 1, _MAX_NODES)
    port = _integer(master_port, "master_port", 1024, 65535)
    ppn, _ = _rank_shape(gpu_mode, 1)
    job = inherited.get("PBS_JOBID", "")
    if not isinstance(job, str) or not _JOB_ID.fullmatch(job):
        raise ValueError("Aurora rank wrapper requires a PBS batch allocation")
    if (nodefile_text is None) == (master_host is None):
        raise ValueError(
            "supply either allocation nodefile text or launch-wrapper master host"
        )
    master = (
        _rendezvous(_allocation_master(nodefile_text, nodes), rendezvous_network)
        if nodefile_text is not None
        else _hostname(master_host)
    )
    rank = _mpi_integer(inherited, _RANK_KEYS, required=True)
    local = _mpi_integer(inherited, _LOCAL_KEYS, required=True)
    world = _mpi_integer(inherited, _SIZE_KEYS, required=False)
    local_size = _mpi_integer(inherited, _LOCAL_SIZE_KEYS, required=False)
    if (
        rank is None
        or local is None
        or rank >= nodes * ppn
        or local >= ppn
        or rank % ppn != local
    ):
        raise ValueError(
            "MPI rank is out of bounds or inconsistent with block placement"
        )
    if (world is not None and world != nodes * ppn) or (
        local_size is not None and local_size != ppn
    ):
        raise ValueError("MPI process count differs from the explicit profile")
    expected = {
        "RANK": str(rank),
        "LOCAL_RANK": str(local),
        "WORLD_SIZE": str(nodes * ppn),
        "LOCAL_WORLD_SIZE": str(ppn),
        "NODE_RANK": str(rank // ppn),
        "NUM_NODES": str(nodes),
        "MASTER_ADDR": master,
        "MASTER_PORT": str(port),
        "ZE_FLAT_DEVICE_HIERARCHY": "FLAT" if gpu_mode == "tile" else "COMPOSITE",
        "ZE_AFFINITY_MASK": str(local),
        "ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1",
    }
    for key, value in expected.items():
        if inherited.get(key) not in (None, "", value):
            raise ValueError(
                f"inherited {key} conflicts with the allocation/rank mapping"
            )
    return expected


def _command(command: list[str]) -> list[str]:
    if (
        not isinstance(command, list)
        or not command
        or len(command) > 4096
        or any(
            not isinstance(arg, str)
            or len(arg) > 65536
            or any(ord(char) < 32 or ord(char) == 127 for char in arg)
            for arg in command
        )
        or not command[0]
        or command[0].startswith("-")
    ):
        raise ValueError("scientific command must be a bounded literal argv")
    return list(command)


def launch_argv(
    command: list[str],
    inherited: Mapping[str, str],
    nodefile_text: str,
    *,
    python: str,
    nodes: int,
    gpu_mode: str = "tile",
    cpus_per_rank: int = 8,
    master_port: int = 29500,
    rendezvous_network: str = "hsn",
) -> list[str]:
    """Pure one-allocation MPI launch plan; rank bootstrap never reads nodefile.

    Called once by the scientific launch wrapper, not by the controller on a
    login node and not once per rank. Does not execute the returned argv.
    """
    command = _command(command)
    nodes = _integer(nodes, "nodes", 1, _MAX_NODES)
    port = _integer(master_port, "master_port", 1024, 65535)
    ppn, binding = _rank_shape(gpu_mode, cpus_per_rank)
    python = _path(python, "python", True)
    job = inherited.get("PBS_JOBID", "")
    if not isinstance(job, str) or not _JOB_ID.fullmatch(job):
        raise ValueError("Aurora launch wrapper requires a PBS batch allocation")
    if any(
        inherited.get(key) not in (None, "")
        for key in (*_RANK_KEYS, *_LOCAL_KEYS, "RANK", "LOCAL_RANK")
    ):
        raise ValueError("refusing a nested MPI launch from an existing rank")
    master = _rendezvous(_allocation_master(nodefile_text, nodes), rendezvous_network)
    return [
        "mpiexec",
        "--envall",
        "-n",
        str(nodes * ppn),
        "-ppn",
        str(ppn),
        "--depth",
        str(cpus_per_rank),
        "--cpu-bind",
        binding,
        python,
        "-m",
        "loopx_hpc.aurora",
        "--nodes",
        str(nodes),
        "--gpu-mode",
        gpu_mode,
        "--master-port",
        str(port),
        "--master-host",
        master,
        "--job-id",
        job,
        "--",
        *command,
    ]


def _read_nodefile(path: str) -> str:
    path = _path(path, "PBS_NODEFILE", True)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_NODEFILE:
            raise ValueError("PBS nodefile must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(_MAX_NODEFILE + 1)
        if len(data) > _MAX_NODEFILE:
            raise ValueError("PBS nodefile exceeds the size limit")
        return data.decode("ascii")
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    """Bootstrap one scientific MPI launch or exec one validated scientific rank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpu-mode", choices=("tile", "device"), default="tile")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--cpus-per-rank", type=int, default=8)
    parser.add_argument("--master-host")
    parser.add_argument("--job-id")
    parser.add_argument("--rendezvous-network", choices=("hsn", "pbs"), default="hsn")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        command = _command(command)
        if not _JOB_ID.fullmatch(os.environ.get("PBS_JOBID", "")):
            raise ValueError("Aurora rank wrapper requires a PBS batch allocation")
        if args.launch:
            if args.master_host is not None or args.job_id is not None:
                raise ValueError(
                    "launch mode derives identity only from its PBS allocation"
                )
            text = _read_nodefile(os.environ.get("PBS_NODEFILE", ""))
            mpi = launch_argv(
                command,
                os.environ,
                text,
                python=sys.executable,
                nodes=args.nodes,
                gpu_mode=args.gpu_mode,
                cpus_per_rank=args.cpus_per_rank,
                master_port=args.master_port,
                rendezvous_network=args.rendezvous_network,
            )
            os.execvpe(mpi[0], mpi, dict(os.environ))
        else:
            if args.master_host is None or args.job_id != os.environ["PBS_JOBID"]:
                raise ValueError(
                    "rank mode requires the launch wrapper's matching PBS job identity and master host"
                )
            updates = rank_environment(
                os.environ,
                nodes=args.nodes,
                gpu_mode=args.gpu_mode,
                master_port=args.master_port,
                master_host=args.master_host,
            )
            os.execvpe(command[0], command, {**os.environ, **updates})
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Aurora scientific rank refused: {exc}", file=sys.stderr)
        return 2
    return 0  # execvpe does not return on success


if __name__ == "__main__":
    raise SystemExit(main())
