# MiniSolver Documentation

This directory is organized by document purpose. Keep long-running plans,
evidence logs, and design decisions separate so each file has a clear role.

## Entry Points

- [Roadmap](ROADMAP.md): milestone-level project direction and completed work.
- [Testing Matrix](testing/testing-matrix.md): what must be tested per commit and
  what belongs in benchmark/nightly coverage.
- [Solver Refactor Plan](architecture/solver-refactor-plan.md): canonical solve
  route, phase boundaries, and refactor rules.
- [Solver Capability Adoption Plan](architecture/solver-capability-adoption-plan.md):
  which mature-solver ideas MiniSolver should absorb, and which are explicitly
  out of scope.
- [Solver Status Semantics](architecture/solver-status-semantics.md): terminal
  status layering and postsolve precedence.
- [Termination Design](architecture/termination-design.md): planned residual
  semantics for `OPTIMAL`, `FEASIBLE`, barrier `mu`, and future `SolverInfo`.

## Directories

| Directory | Purpose |
| --- | --- |
| [`adr/`](adr/) | Accepted or deferred architectural decisions. ADRs should be stable and concise. |
| [`architecture/`](architecture/) | Larger design plans and implementation roadmaps. |
| [`matrix/`](matrix/) | MiniMatrix policy, tuning, and benchmark notes. |
| [`reviews/`](reviews/) | Static/deep review artifacts and evidence-driven follow-up plans. |
| [`testing/`](testing/) | Test matrix, coverage gaps, and test process documents. |

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
