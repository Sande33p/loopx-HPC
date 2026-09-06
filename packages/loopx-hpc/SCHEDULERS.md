# Scheduler execution and site acceptance

Aurora uses **PBS**. This package also supports native **Slurm** commands for
other sites; it does not describe Slurm as Aurora's scheduler. The implementations
are opt-in extensions beneath existing LoopX host adapters, not agent launchers.

## Deployment contract

- Run the controller on a login node with the selected site's `qsub/qstat/qdel`
  or `sbatch/sacct/scancel` available. It submits a batch allocation; it never
  runs the scientific payload on the login node.
- Keep one authoritative controller and persistent campaign directory. Attempt
  directories must be accessible at the same absolute path on the batch host
  and all workload nodes. The present adapter has no SSH, rsync, object-store
  stage-out or remote filesystem translation.
- Only the controller accesses SQLite. Because the ledger and attempt files
  currently share a root, validate locking, crash durability and persistence on
  the selected filesystem before cluster use. Do not assume an arbitrary
  parallel filesystem is a supported SQLite deployment.
- Install `loopx-hpc` in the explicit shared Python named by the profile. Pin
  workload code, dataset and environment; declared hashes/revisions are not
  installed-software attestations. MLflow is needed only on the sync controller.
- Profiles, workloads, startup scripts and shared storage are owner-controlled
  trusted inputs. This is not an adversarial sandbox or signed-provenance system.
  Keep private accounts, directories and raw logs out of tracked source.

## Configure Aurora

Copy [the sanitized input template](examples/aurora-profile-input.json) into an
ignored local configuration and fill in the approved account, queue, absolute
runtime/work directory, versioned modules, filesystem and resource limits.
Set `configured` to true only after review. A generated recipe is not proof that
the account, modules or allocation work. Then build the execution profile:

```sh
.venv/bin/loopx-hpc site-profile --site aurora --input .local/site-input.json > .local/site-profile.json
```

Aurora's recipe uses a PBS login shell for site initialization, scoped nounset
relaxation around module loading, then literal variable exports and the selected
working directory. A login shell executes trusted user/site startup files; review
those along with the profile. The outer batch worker runs once. Its allocation
bootstrap reads `PBS_NODEFILE` once before `mpiexec`, derives the rendezvous host,
and starts one scientific process per selected tile/device. Each rank validates
its native rank identity and sets explicit distributed environment variables.
Aurora's execution queue supplies its `system` chunk default; the site profile
therefore emits `select=N` without an explicit `system=aurora` chunk. Current
Aurora PBS marks that resource host-owned and rejects it when a job selects it
explicitly.

The default tile layout is 12 ranks/node (6 physical GPUs with 2 tiles each);
device mode is 6 ranks/node. CPU binding uses documented compact core ranges.
Rendezvous defaults to the documented Aurora HSN hostname; the explicit `pbs`
network option preserves the allocation nodefile hostname instead.
The rank wrapper applies an affinity mask, so the workload must select `xpu:0`
while preserving the true global/local rank for communication. Frameworks that
instead interpret `LOCAL_RANK` as an unmasked device index need an explicit
workload adaptation; no generic Accelerate compatibility is claimed.

This follows the reusable PBS, modules-before-runtime, allocation/rank and
rendezvous patterns in PRISM's Aurora launchers. It deliberately does not copy
project accounts, private paths, broad process-kill commands, data staging,
training arguments or DAOS setup. The shared-storage recipe supports home/flare;
DAOS and node-local environment unpacking require separate workload adapters.
For PRISM-style PyTorch jobs, review explicit post-module variables such as
`CCL_PROCESS_LAUNCHER=none` and `CCL_ATL_TRANSPORT=ofi` against the pinned framework
stack; collective-algorithm and model knobs are not silently set by the profile.
Do not put a launcher's own `--batch`/`qsub` action inside the scientific command:
that would submit a nested job. Supply the in-allocation payload instead.

## Configure Slurm

`site-profile --site slurm` accepts `{resources, environment, python}` using
[this sanitized template](examples/slurm-profile-input.json) and builds
an `srun` prefix. Resources must include explicit `ntasks_per_node` and
`cpus_per_task` as well as job name, nodes and walltime. Select account, partition,
GPU resources, module environment and framework rank initialization for that
site. No Intel/Aurora assumptions are applied. Federated-cluster submission
acknowledgements are held unresolved: cross-cluster routing is not implemented.

## Study and execution

Initialize a reviewed study manifest and register a candidate using `init` and
`add` as described in [OPERATIONS.md](OPERATIONS.md). The frozen scientific command
is an argv list containing separate `{config}` and `{result}` arguments; it must
write exactly `{"metrics":{"YOUR_PRIMARY_METRIC":1.0}}` to the supplied result
path. For MPI, only global rank zero writes that authoritative result. Config and
result arguments are absolute paths; payload cwd is the profile's working
directory, or the attempt directory when none is selected.

```sh
.venv/bin/loopx-hpc --root .local/study scheduler-submit --experiment EXACT_ID --profile .local/site-profile.json
.venv/bin/loopx-hpc --root .local/study scheduler-submit --experiment EXACT_ID --profile .local/site-profile.json --execute
.venv/bin/loopx-hpc --root .local/study reconcile
.venv/bin/loopx-hpc --root .local/study context
.venv/bin/loopx-hpc --root .local/study tracking-sync --destination .local/tracker --execute
```

Preview creates neither run files nor a submission. Execution durably reserves
the attempt and concurrency slot before contacting the scheduler, writes a
frozen packet/config and checked script, and records the returned job identity.
Worker output remains private under `runs/EXPERIMENT/TOKEN/`; the compute worker
writes a bounded receipt and never opens the campaign database. `reconcile`
requires both exact terminal accounting and verified worker/config/result bytes
before success can enter planning evidence. MLflow sync is a separate explicit
effect and does not drive execution or accept scientific conclusions.

## Restart, cancellation and failures

- Repeating submission never launches a second attempt. A changed profile is
  rejected. Unknown attempts retain their capacity reservation.
- If submission acknowledgement is lost, reconciliation can recover the job
  ID from a matching completed worker receipt and independently query accounting.
  Before that proof, the attempt remains unknown; there is no manual job-ID
  override or automatic resubmission.
- When a returned submission acknowledgement is unresolved, bounded native
  stderr is retained only in the mode-`0600` private attempt artifact
  `runs/EXPERIMENT/TOKEN/scheduler-submit.stderr`. It is never copied into the
  SQLite journal, status/context output, or tracking projections. A missing
  artifact means stderr was empty or its private persistence also failed.
- PBS jobs are marked non-rerunnable; Slurm uses `--no-requeue`. The worker also
  exclusively claims a durable `started.json` before execution. Scheduler
  redelivery cannot silently rerun the same payload.
- Missing/accounting-lagged/ambiguous jobs remain unknown. Scheduler success
  with a missing, malformed or changed receipt also remains unknown. Terminal
  scheduler failure is a failed attempt, never successful scientific evidence.
- `scheduler-cancel --experiment EXACT_ID` previews. Adding `--execute` records
  intent and calls `qdel`/`scancel` once, only after a fresh native query proves
  the exact active job ID and generated attempt name. Unresolved or recycled
  IDs produce no cancellation. Query and cancel are not an atomic scheduler
  compare-and-cancel operation. Acknowledgement is not termination;
  reconcile until accounting proves it. Replayed cancel requests do not repeat
  the external effect. Ambiguous cancellation needs operator inspection.
- Payload timeout and POSIX process-group cleanup are bounded local safeguards;
  the native scheduler owns allocation termination. There is no checkpoint
  resume, retry policy, batch-array manager, or automatic allocation migration.

## Qualification checklist

Local acceptance uses native-command doubles but real generated scripts, worker
processes, JSON result validation and the real local MLflow SDK. It is not an
Aurora or Slurm cluster benchmark. Before a bounded live pilot, independently
verify the account/queue, scheduler config and accounting retention, shared path
and SQLite semantics, pinned modules/Python, MPI/XPU rank binding, representative
rank-zero result writer, cancellation and controller restart. Obtain an approved
small resource budget. No background polling service or host loop is installed
by these commands; compose the existing LoopX continuation and governed
`run_scheduler` bridge with explicit monitor/writeback/settlement policy.

Syntax and layout references: [ALCF Aurora jobs](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/),
[Slurm sbatch](https://slurm.schedmd.com/sbatch.html),
[Slurm sacct](https://slurm.schedmd.com/sacct.html).
