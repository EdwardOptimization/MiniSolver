# MiniSolver Documentation

This directory is organized by document purpose. Keep long-running plans,
evidence logs, and design decisions separate so each file has a clear role.

## Entry Points

- [Roadmap](ROADMAP.md): milestone-level project direction and completed work.
- [Testing Matrix](testing/testing-matrix.md): what must be tested per commit and
  what belongs in benchmark/nightly coverage.
- [Contract Coverage Matrix](testing/contract-coverage-matrix.md): behavior
  contract IDs mapped to tests, replay, benchmark, memory evidence, and gaps.
- [Contract Rollout Completion Audit](testing/contract-rollout-completion-audit.md):
  current evidence that the initial contract-driven development rollout is
  complete.
- [Replay Corpus Plan](testing/replay-corpus-plan.md): in-repository
  correctness/replay scenarios and metrics.
- [MiniSolver Behavior Contracts](contracts/README.md): stable behavior
  contracts for solver internals and evidence ownership.
- [Module Inventory](modules/README.md): implementation module boundaries,
  inputs, outputs, ownership, and known gaps.
- [Solver Refactor Plan](architecture/solver-refactor-plan.md): canonical solve
  route, phase boundaries, and refactor rules.
- [Agent Harness](architecture/agent-harness.md): project-specific agent rules
  extracted from Codex session interventions.
- [Review Triage Checklist](reviews/review-triage-checklist.md): classify review
  findings before deciding fix, docs-only, defer, or reject.
- [Solver Development Principles](architecture/solver-development-principles.md):
  the small principles behind MiniSolver changes; detailed process remains in
  the harness and testing documents.
- [Solver Capability Adoption Plan](architecture/solver-capability-adoption-plan.md):
  which mature-solver ideas MiniSolver should absorb, and which are explicitly
  out of scope.
- [Solver Status Semantics](architecture/solver-status-semantics.md): terminal
  status layering and postsolve precedence.
- [Termination Design](architecture/termination-design.md): residual semantics
  for `OPTIMAL`, `FEASIBLE`, barrier `mu`, and `SolverInfo`.
- [Snapshot And Replay Design](architecture/snapshot-replay-design.md): debug
  snapshot scope, failure-capture pattern, and model fingerprint contract.

## Directories

| Directory | Purpose |
| --- | --- |
| [`adr/`](adr/) | Accepted or deferred architectural decisions. ADRs should be stable and concise. |
| [`architecture/`](architecture/) | Larger design plans and implementation roadmaps. |
| [`contracts/`](contracts/) | Stable behavior contracts and contract ID policy. |
| [`matrix/`](matrix/) | MiniMatrix policy, tuning, and benchmark notes. |
| [`modules/`](modules/) | Module inventory and ownership boundary documents. |
| [`reviews/`](reviews/) | Static/deep review artifacts, evidence-driven follow-up plans, and overdesign decisions. |
| [`testing/`](testing/) | Test matrix, coverage gaps, and test process documents. |
| [`archive/`](archive/) | Historical notes kept for context; not authoritative current guidance. |

## Conventions

- Prefer adding an ADR for stable decisions that should outlive an active plan.
- Prefer `architecture/` for evolving implementation plans.
- Prefer `reviews/` for dated review results and fix queues.
- Prefer `testing/` for test coverage contracts and gaps.
- Do not delete review findings after they are fixed. Preserve the original
  discovery, then add status, evidence, and resolution in a follow-up ledger.
- Treat dated review files as historical snapshots. If current code diverges
  from a review, update the linked fix plan or testing matrix instead of
  rewriting the original review narrative.
- Keep benchmark result dumps out of this repository unless they explain a
  MiniSolver design decision. Full NMPC benchmark data belongs in
  `MiniSolver-Bench`.
