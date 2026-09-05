# HPC integration boundaries and next acceptance gates

## Placement decision

The provider id is `loopx-hpc`; the caller-facing capability is
`experiment-execution`. This independently installed package owns scientific
execution and records. Existing ML-domain helpers are advisory, while the
benchmark toolkit intentionally delegates runner ownership. Neither requires
putting scheduler-specific state into LoopX's generic kernel. The fork keeps
upstream code, licensing notices and entry points intact; this is an independent
downstream integration, not an official LoopX release.

LoopX remains authoritative for goal, Todo, gates, accepted progress and Turn
settlement. The campaign database owns exact scientific definitions, attempts,
results and decision lineage. Tracking is a projection. Future semantic memory
must remain advisory, linked to exact source records rather than granting
execution authority or replacing them.

## Implemented integration slice

- Immutable finite study definitions, exact typed configs, provenance fields,
  local event history and bounded proposal admission.
- Explicit local process execution with detached workers, durable attempt
  reservation, completion validation, timeout handling and unresolved-state
  duplicate suppression. Unknown attempts continue to reserve capacity.
- A model-free adaptive demonstration and a harness-independent JSON proposal
  boundary, fenced against stale context and changed evidence.
- Local JSON, SQLite-backed MLflow and offline W&B snapshot projections, plus
  stable per-attempt MLflow lifecycle sync with explicit external-delivery opt-in.
- PBS/Slurm native CLI submission, reconciliation and explicit cancellation,
  exact scheduler/receipt identity binding, and durable lost-ack suppression.
  Compute workers publish shared-file receipts without accessing SQLite.
- A PRISM-informed Aurora PBS/MPI profile and allocation-only rank wrapper,
  separate generic Slurm/srun profile, and declarative environment recipes.
- Optional read-only and governed material LoopX provider integration. Host
  writeback must distinguish scientific failure from protocol settlement.
- Optional structured predictions and alternatives, source-bound immutable
  assessments and a read-only experiment evolution graph with exact lineage.
- Explicit local cross-project lessons with applicability limits, evidence/event
  hashes, corrections, withdrawal and reviewed adoption. Lesson-informed
  proposals recheck the same library without importing source results as target
  evidence. A four-trial synthetic source-to-target acceptance demo is available.

No upstream default behavior changes. No general scheduler service, live model
campaign, Aurora execution, container builder, FALDA connector, or statistical
optimizer is claimed.

## Ordered follow-up work

1. Use a representative local scientific workload, not the synthetic demo.
   Specify baseline, data splits, metric, replicates and evaluation rules.
   Qualify two existing upstream agent-host integrations consuming identical
   context and returning bounded proposals without inheriting hidden conversation
   state; do not build a duplicate agent spawner.
2. Compose and qualify the existing LoopX host continuation/Turn machinery with
   the HPC provider's admission, writeback, monitor and quota contracts. Codex,
   Claude Code and Pi host support already exists upstream; the remaining gap is
   an end-to-end model-driven HPC campaign, not a missing generic host runtime.
   Keep one outer lifecycle owner. Measure CPU/GPU/node-time separately from
   agent compute; qualify cancellation and explicit attempt-recovery policy.
3. Qualify the implemented PBS/Slurm adapters against a real scheduler. Native
   command doubles and real local worker tests already cover separate experiment,
   attempt and scheduler-job identity, receipt-based lost-ack recovery, queued
   versus running state, cancellation, result ingestion and duplicate suppression.
   Remote artifact transfer, checkpoint continuation and automatic retry are not
   implemented; the current transport is a shared filesystem.
4. Qualify a real site environment and credential boundary. For an Aurora pilot,
   obtain an approved allocation/queue, filesystem, modules/container approach,
   Intel GPU/MPI launch recipe and small smoke budget. Preview/review first;
   do not infer these values from the local machine.
5. Run crash/restart, repeated delivery, interrupted submission, failed
   execution, partial-artifact and requeue tests on that site. Only then enable
   a bounded real scientific campaign and optional remote tracker sync.
6. Extend scientific selection with comparability, replicates/uncertainty,
   holdout protection and explicit promotion/stopping policy. Add evidence-linked
   semantic retrieval only after exact-record acceptance is stable.

## Known limits

The local worker is not an adversarial sandbox. Arbitrary owner-approved
commands run with local process permissions. POSIX process groups cannot contain
deliberate session escapes; local execution is unsuitable for untrusted code.
The first store supports one attempt per config with no automatic retries.
Replicates must include seed/replicate identity in the frozen configuration.
SQLite state belongs on a local filesystem; do not put this ledger on a shared
HPC filesystem without validating its locking and durability semantics.
Environment digests describe declared recipes, not actual resolved software.
Lessons are advisory local snapshots, not semantic model memory or signed
scientific truth. Automatic correction propagation across copied libraries,
statistical validity, source authorship and remote provenance attestations are
not implemented. See [RESEARCH.md](RESEARCH.md) for the exact learning contracts.

The future-facing review kept scheduler recipes, tracking projections and
scientific records in the optional package instead of introducing a parallel
generic control plane. Reuse upstream's existing host adapters and continuation
machinery; see the [host reuse map](OPERATIONS.md#reuse-the-existing-loopx-host-runtimes).
Production HPC campaign and live scheduler qualification remain explicit
follow-up work rather than a new generic agent-launch framework in core.
