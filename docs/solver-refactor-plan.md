# MiniSolver Solver Refactor Plan

Date: 2026-05-01

Status: Active plan, Phase 1-4 initial pass implemented locally

Related:

- `docs/adr/0004-solver-strategy-boundaries.md`
- `include/minisolver/solver/solver.h`
- `include/minisolver/core/solver_context.h`

## Goal

MiniSolver should have one canonical solve route that is simple to reason about,
easy to test, and hard to accidentally break. Advanced behavior should be
attached behind narrow internal seams, not spread through the solve loop as
unbounded `if/else` branches.

The target is not a large public plugin framework. The public API should remain
`SolverConfig` plus `solve()`. Internal organization can evolve toward strategy
kernels and a frozen solver plan only after the current phase boundaries prove
useful.

## Non-Goals

- Do not expose template policy stacks to users.
- Do not add public strategy objects until a real external use case requires it.
- Do not move every helper into a new class just for aesthetic cleanup.
- Do not change solver behavior during structural refactors unless a focused
  test or benchmark proves the need.
- Do not mix solver architecture refactors with matrix-kernel tuning, ppoly
  design, or benchmark asset changes.

## Canonical Solve Route

The solver should read as this fixed route:

```text
solve()
  begin_solve_()
    ensure components
    presolve/reset/initialization
  run_solve_loop_()
    execute_solve_iteration_()
      evaluate derivatives / residual summary
      update barrier state
      compute search direction
      check convergence
      globalize step
      update iteration metrics
    loop exit checks
  postsolve()
    refresh final residuals
    evaluate dual residual
    classify final status
```

This route is the reference mental model. Optimized features may be default-on,
but they must not obscure the route.

## Current State After First Refactor Pass

Completed in the current local branch:

- `solve()` now delegates to `begin_solve_()`, `run_solve_loop_()`, and
  `postsolve()`.
- `run_solve_loop_()` now separates step-status exit checks from cost-stagnation
  checks.
- `execute_solve_iteration_()` replaced the old private `step()` name and now
  reads as a short phase pipeline.
- Derivative evaluation, barrier update, direction solve, globalization,
  tiny-step recovery, and post-globalization metrics are separated into helpers.
- Mehrotra predictor-corrector direction logic is isolated from the generic
  direction phase.
- Postsolve residual refresh, dual residual evaluation, and final status verdict
  are separated.
- Presolve runtime reset and primal-dual initialization are separated.

Validation already run after the pass:

```bash
cmake --build .build -j$(nproc)
ctest --test-dir .build --output-on-failure
cmake --build .build_custom -j$(nproc)
ctest --test-dir .build_custom --output-on-failure
```

Both Eigen and custom MiniMatrix builds passed all 19 tests.

## Current State After Context And Kernel Pass

Completed after the initial loop extraction:

- `SolverContext` now owns solve scalars (`mu`, `reg`, `current_iter`,
  `slack_reset_consecutive_count`) and solver metrics.
- Serializer and focused bugfix tests now read/write solve state through
  `SolverContext`, keeping serialized state explicit.
- Internal phase contracts now return small result structs:
  `DirectionResult`, `GlobalizationResult`, `LoopExitDecision`, and
  `PostsolveResiduals`.
- A test-only reference config helper exists in `tests/test_reference_config.h`
  and is exercised by the analytical QP correctness test.
- The first useful internal kernels are extracted:
  `BarrierUpdateKernel`, `TerminationKernel`, and `InitializationKernel`.

Validation run after each commit:

```bash
git diff --check
cmake --build .build -j$(nproc)
ctest --test-dir .build --output-on-failure
cmake --build .build_custom -j$(nproc)
ctest --test-dir .build_custom --output-on-failure
```

Additional nmpc-bench smoke, using a temporary build/output and the local
MiniSolver checkout:

```bash
cmake -S /home/quyaonan/workspace/nmpc-bench \
  -B /tmp/minisolver_refactor_bench_pendulum \
  -DMINISOLVER_SOURCE_DIR=/home/quyaonan/workspace/MiniSolver \
  -DMINISOLVER_CASE=pendulum_on_cart
cmake --build /tmp/minisolver_refactor_bench_pendulum \
  --target minisolver_official_case_benchmark -j$(nproc)
/tmp/minisolver_refactor_bench_pendulum/minisolver_official_case_benchmark \
  --steps 5 --output /tmp/minisolver_refactor_pendulum.csv
```

Result: `pendulum_on_cart` succeeded `5/5`, median `0.041 ms`, p95 `0.041 ms`.

Deferred deliberately:

- `RestorationKernel`: current code is tightly coupled to slack reset,
  `linear_solver`, and `feasibility_restoration()`. Extracting it now would
  require callback-style plumbing without a second implementation.
- `ModelEvaluationKernel`: `model_evaluation.h` already provides the current
  seam. Further extraction should wait until there are two concrete evaluation
  modes that need a shared contract.
- Frozen `SolverPlan`: still premature until more seams have real variants or
  plan construction starts accumulating compatibility checks.

## Target Architecture

### Layer 1: Public Configuration

Keep this simple:

```cpp
SolverConfig config;
solver.set_config(config);
SolverStatus status = solver.solve();
```

Users select behavior with `SolverConfig`. They should not assemble policy
graphs or own internal workspaces.

### Layer 2: Canonical Loop

The loop owns algorithm order. It should not know details such as filter history
rules, Mehrotra sigma heuristics, restoration internals, or Riccati fused-kernel
dispatch.

Allowed responsibilities:

- order phases
- pass context/state between phases
- stop on explicit phase status
- record diagnostics

Disallowed responsibilities:

- strategy-specific math
- hidden fallback decisions
- direct mutation of unrelated state owned by another phase

### Layer 3: Internal Phase Functions

Short term, phase functions are enough. Each phase should have one narrow
contract and no public API impact.

Current and intended seams:

| Seam | Current shape | Direction |
| --- | --- | --- |
| Initialization | helper functions in `solver.h` | move initialization state into context later |
| Model evaluation | `evaluate_step_model_()` and postsolve refresh | centralize all stage evaluation variants |
| Barrier update | `update_barrier_for_step_()` / `update_barrier()` | eventually internal `BarrierUpdateKernel` |
| Direction solve | `compute_search_direction_()` plus Mehrotra helper | split regularization, linear solve, refinement |
| Globalization | `globalize_step_()` and `LineSearchStrategy` | keep line-search variants behind one seam |
| Restoration | `attempt_tiny_step_recovery_()` / `feasibility_restoration()` | isolate as recovery/restoration phase |
| Termination | convergence checks plus loop exit checks | make termination criteria explicit and testable |
| Diagnostics | timer, alpha log, metrics | avoid allocation in hot paths |

### Layer 4: SolverContext

`SolverContext` should become the boundary between the canonical loop and
internal kernels, but only gradually.

Current:

- `SolverMetrics`
- `StepResidualSummary`

Future internal state groups:

- `SolveState`: `mu`, `reg`, `current_iter`, `slack_reset_consecutive_count`
- `ResidualState`: primal, dual, complementarity, objective cost
- `DirectionState`: solve success, affine metrics, alpha-aff, invalid-direction
  status
- `GlobalizationState`: accepted alpha, recovery result, line-search status
- `TerminationState`: loop exit reason and final status inputs
- `DiagnosticsState`: timing phase ids, event counters, optional logs

Do not move all state at once. Move state only when a phase can consume it
without increasing coupling.

### Layer 5: Frozen SolverPlan, Deferred

The final shape may become:

```text
SolverConfig
  -> build internal SolverPlan
  -> solve executes frozen phase ops
```

But this should be deferred until at least two or three seams have multiple real
implementations. Today, a full `StrategySpec -> StrategyKernel` framework would
be premature.

## Reference Path vs Default Path

MiniSolver needs both:

- Reference path: conservative, readable, correctness-first.
- Default path: faster and more robust, may use advanced strategies by default.

Initial candidate split:

| Capability | Reference path | Default path |
| --- | --- | --- |
| Barrier | monotone/adaptive simple update | adaptive or Mehrotra |
| Direction | Riccati generic path | Riccati with optimized/fused variants when valid |
| Globalization | standard merit line search | filter/merit/none depending on config |
| Restoration | disabled or minimal | enabled when configured |
| Refinement | disabled | defect-rollout refinement when useful |
| Diagnostics | deterministic minimal metrics | profiling/logging only when enabled |

This split should initially be expressed as helper config presets or tests, not
as a new public solver class.

## Implementation Plan

### Phase 0: Canonical Loop Extraction

Status: implemented locally.

Scope:

- Extract solve lifecycle helpers.
- Extract iteration phase helpers.
- Extract direction/globalization/postsolve/presolve helpers.
- Keep behavior unchanged.

Validation:

- `git diff --check`
- Eigen build and ctest
- MiniMatrix build and ctest

### Phase 1: SolverContext Expansion

Scope:

- Move scalar iteration state behind internal context accessors where it reduces
  argument passing.
- Keep serializer compatibility explicit. Do not move serialized fields without
  updating serializer tests.
- Keep public API unchanged.

Likely first moves:

- `mu`, `reg`, `current_iter`
- last primal/dual residual
- last alpha and affine diagnostics

Validation:

- serializer tests
- `test_bugfixes`
- full Eigen and MiniMatrix ctest

Exit criteria:

- `execute_solve_iteration_()` receives/updates context instead of many local
  scalar references.
- No new public API.

### Phase 2: Explicit Internal Phase Results

Scope:

- Replace ad hoc reference parameters with small internal result structs where
  they improve contracts.
- Candidate structs:
  - `DirectionResult`
  - `GlobalizationResult`
  - `PostsolveResiduals`
  - `LoopExitDecision`

Rule:

- Add a result struct only when it removes multiple loosely-related output
  parameters or makes a phase contract testable.

Validation:

- Existing tests should be enough for structural changes.
- Add focused tests only if a phase result exposes a previously implicit status.

Exit criteria:

- Direction, globalization, and postsolve contracts are explicit.
- No behavior changes.

### Phase 3: Reference Configuration Preset

Scope:

- Define an internal or documented reference configuration.
- Use it in tests and benchmark correctness gates.
- Do not add a new public solver type yet.

Candidate:

```cpp
SolverConfig make_reference_config();
```

This can be public only if examples/benchmarks need it directly. Otherwise keep
it test/internal first.

Validation:

- correctness tests run both reference and default config where cheap.
- nmpc-bench can compare reference/default behavior on selected cases.

Exit criteria:

- Every advanced default has an obvious lower-complexity comparison path.

### Phase 4: Strategy Seams For Real Variation Points

Scope:

Convert only proven variation points into internal kernels.

Priority order:

1. `BarrierUpdateKernel`
2. `TerminationKernel`
3. `InitializationKernel`
4. `RestorationKernel`
5. `ModelEvaluationKernel`

Do not convert `LinearSolver` and `LineSearchStrategy` just for consistency;
they already have seams.

Validation:

- one commit per seam
- tests before/after
- zero-malloc test when a seam touches solve hot path

Exit criteria:

- `solver.h` main loop contains phase calls, not strategy-specific branching.
- strategy-specific branching lives in one narrow implementation per seam.

### Phase 5: SolverPlan Build Step

Scope:

- Introduce an internal build step only if phase-kernel construction becomes
  expensive or compatibility checks become scattered.
- `set_config()` should mark components dirty, not rebuild immediately.
- `solve()` or explicit component ensure path should rebuild once when needed.

Validation:

- config mutation tests
- backend invariant tests
- zero-malloc solve tests after plan build

Exit criteria:

- solve hot path executes frozen phase choices.
- config changes remain simple for users.

## Evidence Rules

Every nontrivial change must follow:

1. Confirm the current behavior or friction.
2. Add or identify the focused test/benchmark.
3. Make the smallest change.
4. Run the same validation after the change.
5. Commit by behavior unit.

Structural-only refactors may rely on existing tests if:

- no behavior changes are intended,
- the affected code is covered by current tests,
- both matrix backends pass.

Performance or numerical changes require benchmark data.

## Commit Grouping

Use separate commits for:

- loop/phase extraction
- context state movement
- phase result structs
- reference config
- each internal strategy seam
- docs only

Do not mix:

- solver semantics and matrix tuning
- codegen API changes and solver loop refactors
- benchmark assets and core solver architecture
- formatting and algorithm changes

## Open Risks

- `solver.h` is still large. More extraction may be useful, but moving code into
  new files too early can hide coupling instead of removing it.
- `SolverContext` can become a dumping ground. Move state only with a clear
  owner and contract.
- A frozen `SolverPlan` may be overkill until several seams have real variants.
- Reference/default split can drift unless tests and benchmarks exercise both.
- Zero-malloc guarantees must be verified whenever diagnostics or strategy state
  changes.

## Near-Term Next Steps

1. Add small internal phase result structs only where current reference
   parameters are awkward.
2. Move `mu`, `reg`, and iteration metrics into `SolverContext` behind a minimal
   internal API.
3. Add a reference-config helper in tests first, then decide whether it deserves
   public documentation.
4. Convert barrier update into the first internal kernel only after context state
   movement is stable.
5. Re-run nmpc-bench after context/phase-result work to ensure no runtime
   regression from the refactor.
