# Atlant generation pipeline

Atlant supports generation and editing through one `app.run` entry point. It
initializes the SDK dispatcher and registers module-level CORE, ARCH and CODER
handlers. Each handler returns an SDK Update for its next transition.

```text
Original task → CORE specification → ARCH proposal → CORE REVIEW_PLAN
                                                       ↓
                            accepted versioned project contract
                                                       ↓
                  coordinator → ready file tasks → isolated CODER workers
                       ↑                               ↓
                 publish + checkpoint ← validation + structured reports
                       ↓
              project tests/build → CORE FINALIZE → completion
```

## Decision ownership

CORE owns project-wide decisions; ARCH proposes file plans and dependency edges.
The original task is preserved as `source_task`, separately from `tech_task`.
When clarification rounds are disabled (`ATLANT_ASK_LOOPS=0`), the supplied task
goes directly to ARCH without a model paraphrase. CORE still approves the proposal
before any worker starts and audits the completed project.
The architecture is the existing MADE envelope: `conventions` and `files`.
Each file specifies description, priority, dependencies and explicit parallel
safety; high-priority files also specify a concrete plan and public exports.
Web conventions include shared UI components, CSS tokens, props and API payloads.

CORE accepts `REVIEW_PLAN`, `RESOLVE`, `FINALIZE`. It returns `APPROVE`, `REPLAN`
(with existing affected paths and concrete corrections), or `STOP`. Worker
summaries supplement actual source/interface/check evidence; they do not replace
the original task. CORE can inspect the read-only generated workspace if excerpts
are insufficient. Ordinary successful files do not incur a CORE model call.

## Scheduling and publication

The scheduler in `handlers/coder.py` is the only owner of project state and
file publication. It replenishes a bounded `ThreadPoolExecutor` after each result,
without whole-layer barriers. A file becomes ready only when all its dependencies
have passed checks and been published. A singleton dependency unlocks the same
parallel scheduler. `parallel_safe=false` contract owners run exclusively.

Workers receive dependency-scoped immutable snapshots. They never update shared
memory, write output files, or save checkpoints. Each task/attempt has its own
APIRouter, clients, histories and remote sessions. One MAKE carries the complete
context; there are no separate TASK/STRUCTURE acknowledgement calls per file.
Think loops and Python self-review use the existing configured policy. Web files
use typecheck/build/tests and the project-wide CORE audit. Continuations are
bounded and assembled in original order. MADE includes a factual summary with
`behavior`, `exports` and `issues`; an unresolved contract produces `CONFLICT`.

Wrong-target output, invalid protocol, missing exports and failed checks cannot
be published as complete. Python uses Ruff, compile/import checks and AST interface
extraction; web dependency context includes actual direct dependency source.
Shared CSS/SCSS sources propagate through transitive component dependencies too,
so screens see the exact layout class names and tokens owned by the stylesheet.
Coordinator writes are atomic; a failed post-write smoke check restores the previous
file. The output root, including symlinks and reserved engine files, is validated.

A conflict pauses new dispatch while running tasks finish. CORE can keep the
contract and request a repair or ask ARCH for a complete corrected proposal.
An approved change advances `contract_revision` and invalidates changed files and
their consumers; global convention changes invalidate all files. Unrelated results
remain accepted. Invalidating source also discards its obsolete project check
results; CORE reviews the proposed repair before regenerated code is tested again.
There is one REPLAN per directly affected file and three failed
implementation attempts per file, preserved across resume. Transport failures stop
with a checkpoint instead of consuming source-code repair attempts.

## Gateway and context

Generation sessions enforce `read-only`; the coordinator owns writes. The gateway
model still comes from `CODEX_GATEWAY_MODEL`, with the configured cheap model used
where requested. Local model labels alone do not determine the effective gateway
model. `APIRouter.reset_context()` resets remote sessions through the existing
`POST /sessions/{id}/reset`, then clears local history. A missing remote session is
reported to the coordinator; a partial CONTINUE is never silently replayed into an
empty conversation. SDK exhaustion raises an error; SDK worker threads do not save
application state.

## Checkpoints and validation

`.atlant_checkpoint.json` contains the full architecture, contract revision,
completed files, source hashes, summaries and persistent repair counters. Legacy
map-based checkpoints are upgraded from their saved plan and on-disk files.
Missing or changed files invalidate their consumers. An all-written checkpoint
still reruns final checks. Checkpoints are removed only after checks and CORE
FINALIZE succeed.

Final gates run Python smoke/tests and frontend package scripts `typecheck`,
`build`, `test` as applicable. Web dependencies are installed with lifecycle scripts
disabled when absent. A missing validation environment is reported as unavailable,
not passed. `.atlant_metrics.json` retains timings, model calls, queue waiting,
validation results and failure status after completion.

Run from the generator checkout:

```sh
python3 main.py --home /path/to/output-containing-task-atl --workers 6
python3 -B -m pytest -q -p no:cacheprovider tests
```

`--workers 1` uses the same pipeline serially. Existing checkpoints are resumed
automatically; a changed original task requires a separate output directory.
All Atlant and gateway regressions live under `tests/`. Generated task applications
are outside the generator implementation and are not collected by its test suite.

## Application boundaries

SDK `APIRouter` owns model API requests. Application handlers pass explicit role
schemas; they never subclass or replace the SDK transport. CORE owns decisions,
ARCH owns proposals, and CODER owns the dependency queue and publication.
The coordinator yields SDK updates at review points and resumes through registered
continuation handlers. Each file task has isolated SDK routers using module-level
worker handlers; MAKE occurs once, REQUEST handles subsequent requests, and MADE
contains the completed result. THINK moves through the CODER_THINK router.

Workspace read/list/search/run/patch operations live in `macroses`. ToolBroker
delivers a callable and arguments to the coordinator without parsing commands.
Prompt guidance is packaged data with a bounded role/task selector in utils.
Runtime owns detached processes, cancellation, input and approval delivery; TUI
only displays and controls those runs. Tests share one `tests/` root, and plain
`python -B -m pytest -q -p no:cacheprovider` runs the full suite.
