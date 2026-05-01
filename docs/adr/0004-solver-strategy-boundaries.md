# ADR 0004: Solver Strategy Boundaries

Date: 2026-05-01

Status: Accepted

## Context

MiniSolver has accumulated many feature switches in `SolverConfig`: barrier
update rules, line search variants, Hessian approximation, integrator behavior,
Riccati variants, restoration, slack reset, and iterative refinement. Directly
threading every option through `solve()` / `step()` as local `if/else` blocks is
convenient initially, but it couples independent algorithm choices and makes
future correctness/performance comparisons harder.

The project also needs a stable correctness baseline. Python-generated C++
models and the C++ core should keep a simple, conservative path that is easy to
test, while optimized paths can be enabled by default when they are proven
equivalent and faster.

## Decision

Use a config-driven Strategy boundary for solver variability:

- The public API remains `SolverConfig` plus `solve()`. Users should not choose
  template policy stacks or plugin objects.
- The solver main loop should read as a fixed algorithmic pipeline:
  evaluate model, update barrier, solve direction, globalize step, recover if
  needed, check convergence.
- Each variation point should have one narrow strategy seam. Existing examples
  are `LineSearchStrategy` and `LinearSolver`.
- Hot generated paths may still use compile-time dispatch, SFINAE, and fused
  kernels. These are implementation details behind a config-driven boundary.
- Every optimized path must have a baseline/reference path available for tests
  or benchmarks. Optimizations may be default-on, but they should be defeatable
  or independently comparable before being trusted.

## Landing Order

1. Centralize repeated model evaluation (`cost + dynamics + constraints`) behind
   one internal evaluation strategy function. This removes scattered
   `HessianApproximation` branches without changing the public API.
2. Keep `LineSearchStrategy` and `LinearSolver` as the current runtime
   strategies. Do not add new public strategy classes until a variation point has
   multiple real implementations and repeated call sites.
3. Split `step()` internally into named algorithmic phases only where this
   reduces coupling: model evaluation, barrier/direction solve, line search
   recovery, and convergence verdict.
4. For generated model code, introduce explicit baseline-vs-optimized paths only
   where there is a real optimized path to compare. Avoid adding placeholder
   public methods.
5. Add verification mode only after there are at least two independent
   optimized model-evaluation paths that need systematic equivalence checks.

## Quality Bar

- A strategy seam must remove real duplication or isolate a real variation
  point. Do not create abstractions just to save a few lines.
- Keep user-facing configuration flat. Strategy objects are internal wiring.
- Preserve zero-malloc hot paths.
- Any performance strategy must include a benchmark or profile result.
- Any correctness baseline must be exercised by tests before optimized paths
  depend on it.

## Consequences

The solver can grow features without turning the main loop into nested option
logic. The first step is intentionally small: a shared model-evaluation seam.
More invasive refactors, such as a full barrier-update strategy object or
baseline/optimized generated-model split, should be done only when the next
feature makes the current code materially harder to reason about.
