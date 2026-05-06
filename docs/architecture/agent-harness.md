# Multi-Agent Development Harness

MiniSolver is developed with multiple coding and review agents. A single large
instruction skill is not enough for long-running solver work: agents tend to
obey only part of the rule set, self-review is often too lenient, and small
patches can pass tests while leaving poor phase contracts behind.

This document defines the project harness: five core roles that cover planning,
implementation, evidence, architecture review, and final integration. Specialist
roles are started only when the change needs domain-specific scrutiny.

The design follows the same principle as the Anthropic long-running app harness:
split generation and evaluation, make completion criteria explicit before
coding, use artifacts for handoff, and keep the harness as small as the task
allows.

## Source Signals

Session history reviewed:

- Codex rollout:
  `/home/quyaonan/.codex/sessions/2026/04/09/rollout-2026-04-09T22-22-25-019d729f-71e2-72f2-b128-a35b2742dddd.jsonl`
- Main MiniSolver session: 696 user messages.

Approximate intervention counts:

- process / sequencing control: 227
- default semantics and solver contract guidance: 192
- evidence / benchmark / test demands: 122
- design-boundary corrections: 85
- explicit negative corrections: 55

The counts are not a scientific metric. They show that repeated failures are
mostly process and architecture-boundary failures, not isolated coding failures.

## Core Roles

### 1. Planner / Contract Agent

Purpose: turn a request into a small, testable contract before implementation.

Responsibilities:

- Classify the task: bug, design debt, performance issue, hardening, docs, CI,
  benchmark, or unconfirmed review claim.
- Decide which layer owns the change: solver core, MiniModel/codegen, tests,
  docs, examples/tools, or MiniSolver-Bench.
- Define the acceptance evidence: red test, benchmark, allocation test,
  generated-code compile test, docs inspection, or CI check.
- Define explicit non-goals to prevent scope drift.
- Decide which specialists, if any, should be started.

Output artifact:

```text
Task contract
- Problem:
- Owner layer:
- Evidence before fix:
- Allowed files / modules:
- Non-goals:
- Required validation:
- Specialist review needed:
```

### 2. Builder Agent

Purpose: implement the contracted change without expanding scope.

Responsibilities:

- Add or tighten the agreed red test / benchmark first when applicable.
- Implement the smallest clean fix that satisfies the contract.
- Preserve MiniSolver rules: `SolverConfig` remains the user strategy surface,
  solve-time zero-malloc is protected, model semantics stay in MiniModel/codegen,
  cross-solver benchmark comparison stays in MiniSolver-Bench.
- Avoid public APIs unless the contract explicitly justifies them.
- Keep commits behavior-scoped.

Builder is allowed to propose a contract revision if the implementation reveals
that the planned fix would create a worse architecture.

Output artifact:

```text
Builder handoff
- Changed files:
- Red evidence before:
- Fix summary:
- Validation run:
- Known risks / deferred items:
```

### 3. Evidence / QA Agent

Purpose: verify that the evidence chain is real and complete.

Responsibilities:

- Confirm the red test or benchmark failed before the fix when possible.
- Confirm the same evidence passes after the fix.
- Run neighboring tests and the required validation matrix.
- For zero-malloc claims, require allocation instrumentation such as
  `test_memory`.
- For performance claims, compare before/after runtime together with success
  rate, residuals, iterations, and accuracy.
- Reject claims based only on plausible review prose.

Evidence / QA does not judge architecture elegance. It answers: did the patch
prove the intended behavior?

Output artifact:

```text
Evidence report
- Before evidence:
- After evidence:
- Commands:
- Pass/fail:
- Missing evidence:
```

### 4. Architecture Reviewer Agent

Purpose: catch design smells that tests do not catch.

Responsibilities:

- Detect shotgun surgery: the same semantic threaded through many unrelated
  files.
- Detect side channels: `last_*()` getters, mutable member flags, bare
  out-parameters, or temporal coupling such as "call solve, then query side
  state".
- Detect overdesign: new public plugin layers, adapters, DTOs, or framework
  seams without current call sites.
- Detect overdefense: hot-path checks for states that should be impossible under
  valid invariants.
- Check ownership boundaries: solver core vs MiniModel/codegen vs benchmark
  repo.
- Prefer explicit internal phase result objects when they remove temporal
  coupling, for example `LinearSolveResult`, `GlobalizationResult`, and
  `TerminationSnapshot`.

Architecture Reviewer does not block a patch just because it is larger than the
minimal diff. It blocks patches that leave historical workarounds or unclear
contracts in core solver code.

Output artifact:

```text
Architecture review
- Accepted:
- Blocking smell:
- Suggested cleaner contract:
- Deferred architecture debt:
```

### 5. Maintainer / Integrator Agent

Purpose: make the final decision and keep repository history coherent.

Responsibilities:

- Reconcile Builder, Evidence / QA, and Architecture Reviewer outputs.
- Decide whether to accept, request revision, split commits, squash, or defer.
- Ensure docs, review ledgers, and testing matrix are updated when behavior or
  process changes.
- Inspect pre-commit formatting changes before committing.
- Run full CTest before push for solver-core changes.
- Keep the final user-facing summary short and evidence-based.

Output artifact:

```text
Integration decision
- Accepted / revision required / deferred:
- Commit plan:
- Final validation:
- Push readiness:
```

## Specialist Roles

Specialists are optional. Start them only when their domain materially affects
the decision.

| Specialist | Use When | Focus |
| --- | --- | --- |
| Numerics Reviewer | IPM, SQP, Riccati, line search, barrier update, SOC, termination, restoration | Mathematical contract, convergence semantics, mature-solver precedent |
| Performance Reviewer | profiling, MiniMatrix, Riccati kernels, codegen speed, benchmark anomalies | before/after runtime, hotspots, cache/stack/heap behavior, fair measurement |
| Codegen Reviewer | MiniModel, generated C++, symbolic derivatives, constraint packets | generated-code correctness, compile safety, identifier safety, model/core boundary |
| Embedded Reviewer | zero-malloc, exceptions, logging, stack pressure, `-ffast-math`, RT behavior | hard-real-time constraints, fixed-size storage, embedded build profile |
| Docs / Release Reviewer | README, ADRs, review ledgers, roadmap, GitHub metadata | user-facing accuracy, stale claims, release notes, documentation organization |

Specialist output should be short:

```text
Specialist finding
- Issue:
- Evidence:
- Recommendation:
- Must fix now / can defer:
```

## When To Use Which Harness Size

Use the smallest harness that covers the risk.

| Task Type | Required Roles |
| --- | --- |
| Docs-only typo / ledger status update | Maintainer only |
| Small confirmed bug with narrow test | Planner, Builder, Evidence / QA, Maintainer |
| Solver-core semantic change | Planner, Builder, Evidence / QA, Architecture Reviewer, Maintainer |
| Numerical algorithm change | Core 5 + Numerics Reviewer |
| Performance optimization | Core 5 + Performance Reviewer |
| MiniModel/codegen change | Core 5 + Codegen Reviewer |
| Zero-malloc / embedded claim | Core 5 + Embedded Reviewer |
| README / release positioning | Planner, Docs / Release Reviewer, Maintainer |

## Standard Flow

1. Planner writes a task contract.
2. Architecture Reviewer reviews the contract for scope and ownership risk if
   the task touches solver-core architecture.
3. Builder implements only the accepted contract.
4. Evidence / QA verifies before/after evidence and required tests.
5. Architecture Reviewer reviews the patch for smells.
6. Maintainer integrates, commits, updates docs/ledgers, and decides push
   readiness.

Do not skip Evidence / QA for solver behavior. Do not skip Architecture Reviewer
for cross-module changes, even when all tests pass.

## MiniSolver-Specific Gates

### Status Re-Anchor Gate

Use this gate before answering or acting on questions such as "what is still
open?", "is this already implemented?", "what remains to fix?", or "does this
feature have a default strategy?".

Do not rely on session memory or an old review ledger alone. Re-anchor the
answer against current artifacts:

1. inspect the owning config/API declarations;
2. inspect the implementation path;
3. inspect targeted tests and benchmark evidence;
4. inspect docs or review ledgers only after code and tests.

Classify each item explicitly:

- `implemented`: the core mechanism exists;
- `tested`: behavior is covered by regression, allocation, or benchmark
  evidence;
- `documented`: the intended contract is written down;
- `default-config`: the behavior is enabled by the default `SolverConfig`;
- `optional-profile`: the behavior exists but requires user configuration;
- `api-ergonomics`: only helper/profile/setter convenience is missing.

Do not call an item "missing" when only `default-config` or `api-ergonomics` is
absent. For example, the warm-start kernel and recommended MPC warm-start
configuration exist; the remaining question is whether to expose a helper
profile or more granular setters, not whether warm start itself is unimplemented.

### Evidence Gate

For any bug, review finding, numerical change, or high-risk refactor:

1. classify the claim;
2. reproduce with red test, benchmark, compile check, or allocation test;
3. make the smallest clean fix;
4. re-run the same evidence;
5. run neighboring tests;
6. commit behavior and evidence together.

### Overdesign Gate

Use this gate before implementing review suggestions, safer-looking
validations, new public APIs, new config knobs, or theory features.

Classify the proposal first:

- `hard-invariant`: protects a solver invariant that valid inputs must satisfy.
  Examples include enum range checks, finite values, positive slack/dual
  interior, and line-search factors in `(0, 1)`.
- `algorithm-convention`: follows a mature solver convention but is not a
  universal invariant. Examples include filter switching rules, Mehrotra step
  variants, scaling profiles, and globalization profiles.
- `modeling-choice`: constrains how a user formulates a model. Examples include
  forcing `dt > 0`, forcing model-unit feasibility into solver status, or
  requiring handwritten models to define replay fingerprints.
- `product-boundary`: affects release, replay, diagnostics, logging, docs, or
  user expectations more than solver correctness.

Required questions:

1. Does this protect a MiniSolver invariant, or does it restrict user modeling
   freedom?
2. Is there a red test, benchmark, failure trace, or mature-solver reference?
3. Does it add public API, config surface, hot-path branches, or cross-module
   coupling?
4. Can docs, diagnostics, or a deferred ledger entry solve the issue with less
   coupling?

Decision rules:

- `hard-invariant`: write the red test first, then fix.
- `algorithm-convention`: research and design first; implement only with a
  concrete failure case or benchmark target.
- `modeling-choice`: do not add solver-core validation without explicit user
  approval.
- `product-boundary`: prefer docs, report fields, diagnostics, or defer before
  adding knobs.

Record rejected, deferred, docs-only, or modified review suggestions in
`docs/reviews/overdesign-ledger.md`.

### Architecture Gate

Stop before implementation if a fix:

- spreads the same semantic through three or more modules;
- needs a `last_*()` getter, mutable member side-channel, or bare out-param for
  long-lived diagnostics;
- adds a public strategy/plugin/framework object;
- exposes internal trajectory, knot, or workspace data;
- moves model-specific geometry or dataset semantics into solver core;
- adds hot-path checks for states that should be guaranteed by invariants.

In these cases, write the narrowest internal contract first.

### Boundary Gate

- Users configure solver behavior through `SolverConfig` and existing setters.
- Strategy selection is resolved at construction, `set_config`, or solve
  pre-build boundaries, not repeatedly inside hot loops.
- MiniSolver core owns generic NMPC/IPM/Riccati concepts.
- MiniModel/codegen owns symbolic model semantics and generated numerical
  packets.
- MiniSolver-Bench owns cross-solver comparisons and full benchmark datasets.

### Validation Matrix

- Core solver behavior: targeted test, `test_memory`, full `ctest`.
- Line search / SOC / restoration: targeted regression, `test_memory`, full
  `ctest`.
- Termination/status/info: targeted status/residual tests, `test_memory`, full
  `ctest`.
- MiniModel/codegen: generated-code tests and compile target.
- Matrix kernels: microbenchmark before/after plus correctness tests.
- Docs-only: `git diff --check` and stale-claim inspection.

## Current Local Skills

The current local MiniSolver skill remains useful as a router and checklist:

`/home/quyaonan/.agents/skills/minisolver-engineering-harness/SKILL.md`

It should not become a monolithic rule book. Its long-term role is to trigger
this multi-agent harness and point each role to the relevant checklist.
