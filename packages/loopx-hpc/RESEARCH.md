# Research evolution and cross-project lessons

This optional local layer preserves how an experiment campaign evolves. It does
not choose a model, spawn agents, run an unattended service, or grant permission
to execute. Any harness can supply the same JSON contracts. “Reasoning” means
explicit authored hypotheses, predictions, alternatives, selection rationales
and uncertainty—not access to a model's private chain-of-thought.

## Run the transfer acceptance study

After installing the optional package as described in [OPERATIONS.md](OPERATIONS.md):

```sh
.venv/bin/loopx-hpc --root .local/research-pilot learning-demo
.venv/bin/loopx-hpc --root .local/research-pilot learning-demo --execute
.venv/bin/loopx-hpc --root .local/research-pilot/source evolution --format markdown
.venv/bin/loopx-hpc --root .local/research-pilot/target evolution
.venv/bin/loopx-hpc lesson-search --library .local/research-pilot/library --tag synthetic
```

Preview creates no study or library. Explicit execution runs three tiny source
experiments and one independent target experiment. The source measures losses
4, 4 and 0; it records an assessment and publishes a conditional lesson to the
specified local library. The target records why the lesson applies, proposes
one candidate, executes it locally, and assesses its own result. Repeating the
command preserves attempts, decisions, assessments and publication identities.
There are zero model calls or remote jobs. This proves the record/reuse path on
an identical deterministic workload, not scientific generalization, autonomous
discovery, or performance of an optimizer.

## What the evolution projection contains

`evolution` returns an exact JSON graph with a Markdown reading view. It uses
stored parent links and the actual accepted outputs of each proposal, never
guesses causality from time ordering. It includes configuration differences,
original event times and digests, attempted execution, validated metrics,
authored assessments, corrections and explicit lesson-use edges. Numerical
improvements are not statistical significance or a scientific verdict.

The projection identifies the frozen study by both campaign ID and manifest
digest. Experiment IDs alone are insufficient across studies. Attempt tokens,
config/result digests and event identities qualify captured evidence. Research
records hash the entire immutable envelope, including its actual append time;
correction creates a successor record rather than rewriting history.

Missing old reasoning is shown as `not_recorded`; no retrospective rationale is
invented. The original execution timeline remains distinct from assessment,
publication and reuse times. The evolution view shows historical lesson state
at use, not live status in every external library. Local execution history is
the timing source; MLflow's replay-stable metric timestamp is not used.

## Record a prediction before execution

Existing v0 proposals still work. An optional `reasoning` object strengthens the
record. Obtain a fresh `context` first, then submit a proposal such as:

```json
{
  "id": "next-trial",
  "context_digest": "COPY_CURRENT_CONTEXT_DIGEST",
  "runtime": "your-runtime",
  "rationale": "Test the center after measuring both endpoints",
  "evidence_ids": ["EXACT_LOCAL_SUCCESSFUL_ID"],
  "configs": [{"x": 2}],
  "reasoning": {
    "hypothesis": "The bracket midpoint reduces loss",
    "prediction": "Loss below both measured endpoints",
    "selection_basis": "Explain the discriminating value of this trial",
    "alternatives": [
      {"config": {"x": 1}, "reason_not_selected": "Less discriminating under this bounded design"}
    ],
    "uncertainty": "One synthetic trial does not estimate variance",
    "lesson_use_ids": [],
    "agent": {"runtime": "your-runtime", "model": "your-model-or-none"}
  }
}
```

```sh
.venv/bin/loopx-hpc --root STUDY propose --input proposal.json
```

This records planned experiments only. Normal explicit execution or governed
LoopX admission remains necessary. Models/runtime labels are self-reported
provenance, not authenticated identities. The frozen command, search space,
metric and budgets cannot be changed by reasoning or a lesson.

## Assess observed results

An assessment request includes `id`, `source` (copy the complete `source` object
from `context`), `hypothesis`, `outcome`, `interpretation`, `limitations` (a
nonempty list), `next_action`, `experiment_ids`, and `agent` with `runtime` and
`model`. Optional `supersedes` names an existing local assessment being corrected.
The expected source is checked before looking up IDs, preventing accidental
assignment to a similarly named but different study.

```sh
.venv/bin/loopx-hpc --root STUDY assess --input assessment.json
```

`supports`, `refutes`, and `inconclusive` require successful executions with
currently verified artifacts. `infrastructure_failure` requires a failed
attempt and remains an authored diagnostic category, not a conclusion about
the hypothesis. Planned, running and unknown attempts cannot support scientific
assessments. Interpreting valid measurements still requires scientific review:
the package does not validate causal inference, sample size or honesty of text.

Exact replay returns the original record and timestamp. Changing content under
the same ID fails. A new record with `supersedes` preserves the original and
makes it ineligible for new lesson publication. Adding research records changes
the proposal context fence, so an agent must refresh before its next proposal.

## Share a conditional lesson and consider it elsewhere

A lesson publication request contains `title`, `claim`, `tags`, `applicability`
and `limitations` (nonempty lists), and `assessment_ids`. Optional `replaces`
names an active predecessor in the same frozen source study. The builder
revalidates source artifacts, captures the selected assessments, source revision
labels, evidence digests and original execution-event identities/times, and
computes a content-addressed `lesson-...` ID.

```sh
.venv/bin/loopx-hpc --root SOURCE lesson-publish --input lesson.json --library SHARED_LOCAL_LIBRARY
.venv/bin/loopx-hpc --root SOURCE lesson-publish --input lesson.json --library SHARED_LOCAL_LIBRARY --execute
.venv/bin/loopx-hpc lesson-search --library SHARED_LOCAL_LIBRARY --tag METHOD_TAG --tag DATA_TAG
```

The first publication previews only; the second explicitly writes. Search uses
case-sensitive exact AND tags and excludes withdrawn/superseded lessons. Empty
tags list active lessons. There is no global library default, network fetch,
embedding model, FALDA connector or automatic cross-repository scan.

In a target study, a `lesson-use` request contains `id`, target `source` from its
fresh context, `lesson_id`, `action` (`adopt`, `adapt`, or `reject`), `rationale`,
`adaptation` (state explicitly if none), `checks` (nonempty applicability-check
statements), and `agent`. The journal preserves the exact lesson snapshot and
library identity, including why it was rejected if inappropriate.

```sh
.venv/bin/loopx-hpc --root TARGET lesson-use --input use.json --library SHARED_LOCAL_LIBRARY
.venv/bin/loopx-hpc --root TARGET context
.venv/bin/loopx-hpc --root TARGET propose --input proposal.json --library SHARED_LOCAL_LIBRARY
```

The new proposal cites local adopted/adapted record IDs in
`reasoning.lesson_use_ids`. Source experiments are **not** target evidence IDs;
target metrics and incumbent stay empty until target experiments actually run.
Admission checks the same logical library's current active lesson under a read lock
held through campaign commit. Changing libraries requires a new explicit use
record. Withdrawal/supersession blocks new lesson-informed proposals but does
not rewrite historical decisions or retroactively cancel already planned jobs.

```sh
.venv/bin/loopx-hpc lesson-withdraw --library SHARED_LOCAL_LIBRARY --lesson EXACT_LESSON_ID --reason "Explain the correction"
.venv/bin/loopx-hpc lesson-withdraw --library SHARED_LOCAL_LIBRARY --lesson EXACT_LESSON_ID --reason "Explain the correction" --execute
```

Publication and withdrawal replay are idempotent. Replacement preserves old
content and lifecycle events and cannot silently change the frozen source.
There is no automatic propagation from later source-file corruption to an
already exported lesson: withdraw it explicitly. Offline copies are snapshots,
not a distributed consistency service or an assertion of current source truth.
A copy/restore of a whole library preserves its logical ID; locking protects
the selected SQLite file, not divergent copies on different machines.

## Privacy, trust and boundaries

The local shared library is private operator-selected storage, **not a
public-safe export**. Raw log files, artifact paths, commands, process IDs and
source artifact bodies are not copied into lessons. Only allowlisted source
metadata is captured automatically; free-text claims, rationales and revision
labels still need review for secrets before sharing. A full local graph or
tracking projection may contain study details. Do not upload it blindly.

Hashes prove content identity, not authorship or scientific truth. Imported
text is advisory data, not instructions or permission. SQLite triggers guard
ordinary API writes; the local owner can alter files and databases. This is
neither signed provenance nor an adversarial tamper-proof journal. Keep SQLite
on a validated local filesystem; library admission locking requires the default
rollback-journal mode, not WAL or unqualified shared HPC storage.

LoopX still owns goal/todo/gates, admission and accepted progress. Exact records
live here; trackers are projections and reusable lessons remain advisory. This
milestone adds the durable learning path, not native agent adapters, scheduler
dispatch, statistical comparison/replication policy, or long-running autonomous
campaign supervision.
