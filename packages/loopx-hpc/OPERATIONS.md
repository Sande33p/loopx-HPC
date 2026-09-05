# Local HPC integration: operator guide

This is an experimental downstream integration, not an official upstream LoopX
release or a production HPC controller. It adds an optional package without
changing the upstream kernel or the default LoopX installation.

## Install and run the local acceptance study

From the fork checkout, using Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e . -e 'packages/loopx-hpc[test,mlflow,wandb]'
.venv/bin/loopx-hpc --root .local/hpc-pilot demo
.venv/bin/loopx-hpc --root .local/hpc-pilot demo --execute
.venv/bin/loopx-hpc --root .local/hpc-pilot context
```

The first demo command only previews. The explicit execution command runs two
tiny local experiments, reads their actual results, proposes a midpoint, and
runs one follow-up. The objective is a synthetic quadratic, not a scientific
benchmark. The planner is deterministic: no model account, agent login, remote
job, or paid inference is used. Repeating the command reuses the same records
and attempts. It does not spend another experiment budget.

The controller need not remain alive while the detached worker executes. This
does not survive machine shutdown, sleep, lost filesystems, or deliberate process
escape. After interruption, inspect the original directory:

```sh
.venv/bin/loopx-hpc --root .local/hpc-pilot reconcile
.venv/bin/loopx-hpc --root .local/hpc-pilot status
```

An unresolved attempt becomes `unknown`, continues reserving capacity, and is
never automatically retried. There is intentionally no reset/retry/cancel
command in this milestone. Do not edit the SQLite database to force a retry;
inspect the exact worker/process evidence first. A fresh study directory is not
permission to duplicate an unresolved job.

## What is saved

Each campaign directory contains one immutable study definition in
`campaign.sqlite3`, experiment configurations and SHA-256 hashes, parent
experiment references, proposal context digests, runtime labels, rationale,
attempt tokens, observations, and an append-only event table. Each attempt keeps
its exact input, result, and local stdout/stderr. A successful exit without a
valid primary-metric artifact is a failed experiment. Changed artifacts are
excluded from planning evidence.

The command is a trusted owner-supplied argv list with separate `{config}` and
`{result}` arguments; `{python}` resolves to the worker interpreter. The workload
must write exactly `{"metrics": {"primary_metric_name": 1.0}}` to the result path.
All metric values must be finite numbers. Commands are not security-sandboxed.
Keep real campaigns and credentials outside tracked source, preferably under
ignored `.local/`. Local tracking exports are private copies, not redacted
public projections. Source files and runtime dependencies must be pinned by the
study owner; recording a revision string does not enforce a clean checkout.

## Runtime-independent proposals

`context` exports an exact JSON decision packet. Any agent harness may consume
it and produce this contract:

```json
{
  "id": "next-batch-1",
  "context_digest": "exact digest from the current context packet",
  "runtime": "your-agent-runtime",
  "rationale": "Why these configurations discriminate between hypotheses",
  "evidence_ids": ["exact-successful-experiment-id"],
  "configs": [{"x": 2}]
}
```

```sh
.venv/bin/loopx-hpc --root .local/hpc-pilot propose --input proposal.json
.venv/bin/loopx-hpc --root .local/hpc-pilot submit --experiment EXACT_ID
.venv/bin/loopx-hpc --root .local/hpc-pilot submit --experiment EXACT_ID --execute
```

Proposals cannot change the command, study metric, search space, or resource
limits. Stale context, invalid evidence, conflicting proposal IDs and batches
over the experiment budget are rejected atomically. A proposal never launches
work. This JSON boundary is usable from Codex, Claude Code or Pi, but their
native agent-spawning adapters and live model campaigns are not implemented or
validated by this milestone. Runtime labels are provenance, not credentials.

## Local tracking

For explicit predictions, evidence-linked assessments, the evolution graph,
and conditional cross-project lessons, see [RESEARCH.md](RESEARCH.md). The
`learning-demo` command exercises this path using four actual local trials.

```sh
.venv/bin/loopx-hpc --root .local/hpc-pilot track --backend mlflow --destination .local/hpc-tracking
.venv/bin/loopx-hpc --root .local/hpc-pilot track --backend wandb-offline --destination .local/hpc-tracking
```

JSON tracking is dependency-free; MLflow uses an explicitly local SQLite store
and local artifacts. W&B is forced offline with source, Git, host metadata and
system-stat collection disabled. Neither path logs in or syncs to a remote
service. Re-exporting the same snapshot reuses its receipt. Tracker SDK runs are
immutable *projection revisions* linked by a stable experiment key, not new
scientific attempts. Artifact inventories are copied as references; source
artifacts are not automatically copied or uploaded. W&B cannot resume an
interrupted offline export: an unreceipted spool fails closed for inspection.

## Scheduler and environment previews

`render-job --scheduler pbs|slurm --input job.json` prints a script only. Its JSON
input contains `command` (argv), `resources`, and an optional `environment`.
Required resources are `job_name`, positive `nodes`, and `walltime` (`HH:MM:SS`).
Supply site-approved `account` and `queue` values when needed; no site values are
guessed. The Python `preview_plan` API returns inert submit/status/cancel argv,
and `parse_status` interprets one exact scheduler job record. No API in this
milestone executes those remote commands or connects over SSH.

Environment recipes support literal module names, literal variable exports,
an optional absolute working directory, and a recipe digest. They do not build
containers, solve Conda environments, attest installed software, or configure
MPI. `aurora_environment()` is deliberately unconfigured and cannot render.
Actual PBS/Slurm and Aurora acceptance must happen separately against a reviewed
site profile, allocation, storage location, and Intel GPU launch environment.

## Optional LoopX connection

The source-distributed manifest is `packages/loopx-hpc/extension.toml`.
Install and enable this provider only in an explicitly selected LoopX runtime;
the local demo does not modify existing global LoopX state. The provider exposes
`experiment-execution.observe` and `experiment-execution.run_local`.

`observe` uses the existing read-only capability binding. `run_local` must use
`bridge.start_local_experiment` with an actual admitted goal/agent/todo/turn;
the selected Todo must target the exact experiment. Reconciliation requires
real durable writeback and spend callbacks supplied by the host. There is no
synthetic authority packet or automatic Todo completion. Direct `extension run`
and read-only `capability invoke` do not grant material execution permission.

Protocol `status=succeeded` means a terminal execution outcome was recorded;
the experiment itself may have failed. The host must read
`observations[].experiment_status` before accepting progress. Exact configs and
logs stay local; the public provider returns opaque references and digests.
The direct local CLI is an explicitly invoked operator tool, not a substitute
for LoopX admission in unattended use.

## Validation and removal

```sh
.venv/bin/python -m pytest packages/loopx-hpc/tests -q
LOOPX_HPC_TEST_WANDB=1 .venv/bin/python -m pytest packages/loopx-hpc/tests/test_tracking.py -q
.venv/bin/python -m ruff check packages/loopx-hpc
.venv/bin/python -m pip check
```

The upstream Effect runtime needs Node 22.6+ and permission to open local IPC.
Optional SDK tests require their respective extras. Preserve campaign data when
removing the package: `.venv/bin/python -m pip uninstall loopx-hpc` removes the
installed package, not the scientific records. No launchd service, heartbeat,
shell-profile change, or global installation is created by these commands.
