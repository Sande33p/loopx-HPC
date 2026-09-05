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
never automatically retried. There is intentionally no reset/retry or local
process-cancel command. Scheduler cancellation is a separate explicit request.
Do not edit the SQLite database to force a retry;
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
work. This JSON boundary is usable from Codex, Claude Code or Pi through their
existing upstream LoopX host integrations. The optional HPC package does not
add a competing agent spawner. Live model-driven HPC campaigns have not yet
been qualified by this milestone. Runtime labels are provenance, not credentials.

### Reuse the existing LoopX host runtimes

Host support already ships upstream; it is not a planned HPC feature:

| Host | Existing integration | Reuse boundary |
| --- | --- | --- |
| Codex CLI | `loopx turn run-once --host codex-cli` invokes `codex exec` or resumes a bound session. | One governed, isolated-headless Turn with independent postcondition validation; repeated Turns need the selected host/controller wake path. |
| Claude Code | The opt-in [Claude adapter](../../loopx/claude_goal_mode/README.md) sets up `/loopx` and LoopX MCP tools; native `/loop` drives continuation. | Reuse that native runtime and LoopX's per-tick gate; do not start a second outer loop for the same goal. |
| Pi | The opt-in [Pi adapter](../../loopx/pi_goal_mode/README.md) binds a session and triggers quota-gated continuation after `agent_settled`. | Reuse its follow-up, backoff and pause/resume lifecycle; Pi is not a separate headless `turn run-once --host pi` option. |

The concrete Codex launch owner is
[`run_codex_cli_host`](../../loopx/control_plane/turn_driver/codex_cli.py);
[`turn run-once`](../../loopx/cli_commands/turn_registration.py) exposes the
built-in Codex/dsh and typed generic-host routes. Upstream also owns the
[Turn executor](../../loopx/control_plane/turn_driver/executor.py) and
[loop disposition](../../loopx/control_plane/turn_driver/loop_controller.py)
rules. Host availability, installation, authentication and selected lifecycle
must still be verified on the actual machine. A host terminal event alone is
not a scientific result or permission to accept LoopX progress.

For child-agent orchestration, the existing
[host child-context adapter](../../loopx/control_plane/turn_driver/subagent_host_adapter.py)
maps Codex to native `spawn_agent` (fresh or forked context) and Claude to native
`Task` (fresh context). LoopX projects scoped child briefs and reconciles typed
evidence; the host performs the native operation. This mapping does not imply
an equivalent Pi child-spawning adapter or universal agent-tool interception.

The remaining HPC work is composition and acceptance: give the existing host
the study context, validate its bounded proposal, execute through the admitted
capability, then settle from verified experiment evidence. This does not require
reimplementing Codex, Claude or Pi launch/continuation adapters.

## Experiment tracking

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

For one MLflow run that follows an actual attempt through its lifecycle, use
the separate opt-in sync (the older `track` export is unchanged):

```sh
.venv/bin/loopx-hpc --root .local/hpc-pilot tracking-sync --destination .local/hpc-tracking
.venv/bin/loopx-hpc --root .local/hpc-pilot tracking-sync --destination .local/hpc-tracking --execute
```

The default is a local SQLite store under `mlflow-lifecycle/`, with local artifact
storage. The run key combines immutable study identity, experiment and attempt;
planned configurations do not create runs. Submission/running/unknown mirror as
`RUNNING`, verified success as `FINISHED`, and terminal failure as `FAILED`.
Parameters, revision/environment provenance, native job identity, recipe digest,
and verified config/result bytes accompany original event/worker timestamps.
Metric time is the actual receipt time when available, otherwise the original
terminal observation, never export time or epoch zero. Raw logs and environment
variables are not uploaded by this adapter. Sync never launches or reconciles a
job and does not change the campaign ledger.

An explicit `--tracking-uri https://YOUR_APPROVED_SERVER --allow-external
--execute` opts into sending those records and selected artifacts to the server
and its configured artifact repository. Existing SDK authentication applies;
there is no credential embedded in the profile and no login operation. This is
not a hostname/redirect egress restriction. External delivery has not been live
qualified. Use one persistent destination for replay bookkeeping: ambiguous
run creation is held for inspection, and independent remote writers do not have
an atomic global uniqueness guarantee. For offline compute nodes, perform sync
from the controller after reconciliation; no MLflow dependency is required on
compute ranks. There is no background tracker daemon.

## PBS, Slurm and Aurora execution

`render-job --scheduler pbs|slurm --input job.json` prints a script only. Its JSON
input contains `command` (argv), `resources`, and an optional `environment`.
Required resources are `job_name`, positive `nodes`, and `walltime` (`HH:MM:SS`).
Supply site-approved `account` and `queue` values when needed; no site values are
guessed. The Python `preview_plan` API returns inert submit/status/cancel argv,
and `parse_status` interprets one exact scheduler job record. Those pure APIs
remain nonexecuting. The separate `scheduler-submit` / `scheduler-cancel`
commands and `SchedulerExecutor` now perform explicitly opted-in native CLI
effects. `reconcile` dispatches to the original local or scheduler backend.
See [SCHEDULERS.md](SCHEDULERS.md) for the site setup, shared-storage contract,
commands, recovery semantics and acceptance checklist.

Environment recipes support pinned module names, literal variable exports,
an optional absolute working directory and explicit login-shell initialization.
They do not build containers, solve Conda environments or attest installed
software. `aurora_environment()` remains an unconfigured generic placeholder;
the new `aurora_profile()` builds an explicit PBS/MPI recipe and allocation-only
rank wrapper. Aurora is PBS, not Slurm. The separate Slurm profile uses `srun`
and leaves site MPI/GPU choices to the workload. Native CLI tests use scheduler
doubles with real workers; live scheduler, Intel GPU and site acceptance remain
separate gates. No SSH connection or real allocation is made by local tests.

## Optional LoopX connection

The source-distributed manifest is `packages/loopx-hpc/extension.toml`.
Install and enable this provider only in an explicitly selected LoopX runtime;
the local demo does not modify existing global LoopX state. The provider exposes
`experiment-execution.observe`, `experiment-execution.run_local` and
`experiment-execution.run_scheduler`.

`observe` uses the existing read-only capability binding. `run_local` must use
`bridge.start_local_experiment` with an actual admitted goal/agent/todo/turn;
the selected Todo must target the exact experiment. Reconciliation requires
real durable writeback and spend callbacks supplied by the host. There is no
synthetic authority packet or automatic Todo completion. Direct `extension run`
and read-only `capability invoke` do not grant material execution permission.

`bridge.start_scheduler_experiment` uses that same upstream governed machinery,
requiring the separate `experiment.scheduler.execute` permission and an
operator-owned profile. The caller must pass `expected_manifest_digest` from the
selected study context. The new scheduler request contains exactly
`experiment_id`, `profile_digest` and `manifest_digest` (full SHA-256 hex values),
not private paths or resource/environment contents. Both start and reconciliation
reject a different study even when its experiment ID collides; scheduler outcome
evidence includes the source manifest digest. Existing `run_local` and `observe`
input contracts remain unchanged for compatibility. Start rejects a changed
profile; reconciliation uses the frozen attempt binding and does not require
the original profile file.
`bridge.reconcile_scheduler_experiment` requires the same real writeback and
spend callbacks. Cancellation is currently an explicit operator CLI action,
not a new automatic Goal transition.

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
