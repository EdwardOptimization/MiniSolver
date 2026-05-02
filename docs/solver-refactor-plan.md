# MiniSolver Solver Refactor Plan

Date: 2026-05-01

Status: first refactor stage complete; further expansion requires new evidence

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
  and is exercised by analytical QP correctness tests, including a
  reference/default agreement check.
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

## Restoration Route

`feasibility_restoration()` is a recovery phase, not one fixed algorithm. Its
contract is to take a trajectory that the normal IPM step/globalization could
not advance and try to return it to a region where the primal residuals and
barrier interior are usable again. It may change `x/u/s/lam/soft_s`, but it
must leave the main IPM with duals/slacks that are valid for the original OCP
barrier problem.

The current implementation is a quadratic-penalty feasibility restoration. It
temporarily solves a surrogate problem of the form:

```text
min 0.5 ||dx||^2 + 0.5 ||du||^2
  + 0.5 * rho * ||C dx + D du + g||^2
```

using the existing Riccati direction machinery. In code this appears as
stage-local additions such as `Q += rho * C^T C`, `R += rho * D^T D`, and
`q += rho * C^T g`. This is an ALM/ALADIN-adjacent penalty idea, but it is not
a full ALADIN solver and not an augmented-Lagrangian outer loop: there is no
separate consensus QP, multiplier coordination, or adaptive penalty update.

That distinction matters for future architecture. `feasibility_restoration`
should remain the phase name; quadratic-penalty restoration is only the current
implementation. Other possible implementations include:

- slack-reset-only recovery.
- min-norm correction.
- dynamics rollout repair.
- constraint projection.
- elastic-mode restoration.
- trust-region restoration.
- ALM-style restoration with explicit multiplier and penalty updates.

`RestorationKernel` is therefore the right future seam, but not yet the right
current extraction. Introduce it only when a second restoration implementation
or a focused correctness/debuggability issue needs a narrow recovery contract.

## SOC Route

Second-order correction (SOC) is a globalization feature, not model semantics.
The solver core should remain geometry-agnostic: it should not know whether a
constraint row represents a circle, ellipse, obstacle, friction cone, or generic
nonlinear inequality.

The core contract is instead:

```text
active linearization: A/B/C/D and cost curvature from the accepted iterate
trial/candidate residual: true nonlinear residuals after the main step
SOC correction: solve with active structure and trial residuals, then test by true residuals
```

The current generic SOC implementation follows that contract for filter line
search in multiple-shooting mode:

- `solve_soc()` reuses the active Riccati structure.
- The SOC RHS uses the trial candidate's nonlinear residuals.
- The primal-dual baseline for SOC (`s/lam/soft_s`) is seeded from the trial
  candidate, because the correction is applied to the candidate.
- SOC correction has its own fraction-to-boundary damping before it is applied.
- The Filter implementation calls a private `try_soc_correction()` helper, keeping
  SOC construction, damping, application, and re-acceptance in one internal seam.
- `enable_line_search_rollout=true` skips this SOC path for now, because rollout
  mode needs a distinct control-space SOC definition.

Future MiniModel/codegen work should introduce a usage-aware constraint packet
rather than putting geometry logic into the core. The three uses must remain
separate:

| Packet | Used by | Must contain |
| --- | --- | --- |
| true constraint evaluation | filter, merit, convergence, final report | true nonlinear residuals and true derivatives |
| QP/IPM linearization | main Riccati/IPM direction | residual/Jacobian chosen for the main QP |
| SOC correction | SOC RHS and optional correction Jacobian | trial residuals or projected-boundary residuals |

For circle/ellipse/projected-boundary constraints, MiniModel can generate an
SOC override packet. The first geometry-aware version should only override the
SOC RHS/intercept while keeping the active `C/D` rows fixed, so Riccati structure
reuse remains possible. A later version may update projected/trial tangents if
benchmarks justify the extra factorization cost.

Current codegen status:

- The solver core detects an optional `Model::compute_soc_constraints(active,
  trial)` hook.
- Generated models provide that hook. By default it recomputes trial
  constraints; for `quad_boundary_proj`, it overrides only `trial.g_val(row)`
  using the active projected-boundary normal and the trial point.
- Generated models also provide `compute_true_constraints()` and
  `compute_terminal_true_constraints()`. They also provide explicit
  `compute_qp_constraints()` and `compute_terminal_qp_constraints()` entry
  points; `compute_constraints()` remains a compatibility alias. `g_val` remains
  the QP/IPM residual packet paired with `C/D`; `g_true` is the true nonlinear
  residual used by filter, merit, convergence, final reporting, and public
  constraint access.
- The hook does not change solver core geometry semantics. The core still only
  sees numerical residuals and active Riccati structure.
- Full constraint packet objects are intentionally deferred. The current
  first-stage split is field-level: `g_true` vs `g_val/C/D`.

## Hessian Approximation Contract

MiniSolver's current Hessian modes are NMPC-oriented approximations, not a full
generic NLP Hessian stack.

The complete OCP Lagrangian Hessian would contain:

```text
objective Hessian
+ path-constraint Hessians weighted by constraint duals
+ dynamics-defect Hessians weighted by dynamics multipliers
```

The current generated model and Riccati path intentionally use the following
contract:

| Mode | Cost Hessian | Path constraint Hessian | Dynamics Hessian |
| --- | --- | --- | --- |
| `OBJECTIVE_HESSIAN_ONLY` | exact Hessian for `minimize()` terms plus `Jᵀ W J` for `add_residual()` terms | ignored | ignored |
| `EXACT` | exact Hessian of `minimize() + 0.5 * sum(w_i * r_i²)` | included as `sum(lambda_i * Hessian(g_i))` | ignored |

MiniModel exposes the least-squares structure explicitly through:

```python
model.add_residual(residual, weight=1.0)
```

The method accepts a scalar residual or a list/tuple/SymPy vector of residuals.
A scalar weight is broadcast to all residuals; a list/tuple/vector weight is
treated as a diagonal weight and must match the residual length. Dense weight
matrices are intentionally not part of the first API. Existing `minimize(expr)`
continues to mean a general scalar objective term and is not auto-detected as a
least-squares residual.

Dynamics still uses exact first-order information: generated or runtime
integrator code computes `f_resid`, `A = df/dx`, and `B = df/du`. Those
Jacobians are used by the Riccati recursion. The second-order dynamics term
`pi^T * d²f/dz²` is not currently generated or assembled.

This is deliberate for the near-term MiniSolver target:

- realtime NMPC and SQP-RTI commonly prioritize accurate dynamics Jacobians
  over full dynamics curvature;
- dropping dynamics Hessians keeps the Riccati path smaller, more predictable,
  and less likely to introduce indefinite curvature;
- path-constraint Hessians are optional via `EXACT`, but the default remains
  `OBJECTIVE_HESSIAN_ONLY` for speed and robustness.

If full OCP exact Hessians are added later, they should be introduced as a
separate, explicitly named mode and protected by reference tests and benchmark
comparisons. Do not silently reinterpret the current `EXACT` mode as including
dynamics Hessians.

## Current State After Solver Build-State Pass

Completed after the kernel pass:

- Added an internal `SolverBuildState` / `SolverPlanInfo` boundary.
- `set_config()` now marks the internal build state dirty instead of rebuilding
  line-search components immediately.
- `solve()` rebuilds dirty internal components once before presolve.
- Serializer load now routes through the same dirty-build path after restoring
  config and algorithmic state.
- The plan info records the frozen component choices currently needed by the
  solver: backend, line-search type, integrator, component readiness, and
  fused-Riccati/integrator compatibility.
- Line-search kernels are preallocated at construction/build time. A config
  change can switch the active line-search strategy at the solve boundary
  without heap allocation inside `solve()`.

Validation:

```bash
.build/test_bugfixes --gtest_filter=BugfixTest.SetConfigDefersPlanRebuildUntilSolve
.build/test_memory --gtest_filter=MemoryTest.ZeroMalloc_SolveAfterSetConfigDoesNotAllocate
git diff --check
cmake --build .build -j$(nproc)
ctest --test-dir .build --output-on-failure
cmake --build .build_custom -j$(nproc)
ctest --test-dir .build_custom --output-on-failure
```

Both Eigen and custom MiniMatrix builds passed all 19 tests.

Additional nmpc-bench smoke after the build-state pass:

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

Result: `pendulum_on_cart` succeeded `5/5`, median `0.040 ms`, p95
`0.047 ms`.

This is intentionally not a full `StrategySpec -> StrategyKernel` framework.
The useful boundary today is: public config changes are cheap, and internal
component rebuilds happen once at the solve boundary.

## First Stage Complete

The first solver-architecture refactor stage is complete. The solver now has:

- a readable canonical solve route;
- explicit internal phase contracts for the main loop;
- `SolverContext` ownership for core runtime state;
- first useful internal kernels for barrier update, termination, and
  initialization;
- a lightweight build-state boundary for config changes and component rebuilds;
- zero-malloc coverage for the `set_config()` then `solve()` path;
- reference/default correctness coverage on a simple analytical problem.

This is the right stopping point for architecture work. The next priority should
be numerical correctness, benchmark coverage, and measured performance work.
Do not expand the architecture just because the target design mentions future
strategy objects.

## Deferred Architecture Decisions

These features are intentionally deferred. They should be introduced only after
tests, benchmarks, or a second real implementation prove the current structure
is blocking progress.

| Item | Current decision | Trigger to revisit |
| --- | --- | --- |
| `RestorationKernel` | Do not extract now. Restoration is still coupled to slack reset, the linear solver, and feasibility restoration. | A focused bugfix or second restoration implementation needs a narrow recovery contract. |
| `ModelEvaluationKernel` | Do not add another wrapper now. `model_evaluation.h` is the existing seam. | There are at least two real evaluation modes, e.g. baseline generated evaluation, optimized generated evaluation, and piecewise/ppoly evaluation. |
| Full `SolverPlan` | Keep the lightweight `SolverBuildState` / `SolverPlanInfo`. | Compatibility checks or phase-kernel construction become scattered, expensive, or allocation-sensitive beyond the current build-state boundary. |
| Function-pointer phase table | Do not introduce now. | Phase kernels multiply and `solve()` has measurable runtime branch/dispatch pressure. |
| `StrategySpec` / OOP configuration layer | Do not introduce now. Keep public API as `SolverConfig + solve()`. | Multiple stable backends/globalization/restoration strategies need user-facing composition without exposing hot-path virtual calls. |
| Multiple plans inside one solver | Do not introduce now. Use multiple solver/config instances for benchmark A/B tests. | Solver construction cost is benchmark-proven expensive, or an in-solve fallback such as Riccati-to-dense solve requires state-isolated plans. |

The default rule is: prefer multiple `MiniSolver` instances over multi-plan
switching until construction cost or fallback behavior is proven to be a real
problem.

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
| SOC | filter-only multiple-shooting correction via `try_soc_correction()` and optional model SOC hook | introduce full packet objects only after another real variation point needs them |
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

### Layer 5: Solver Build State First, Frozen SolverPlan Later

The current shape is a minimal internal build state:

```text
SolverConfig
  -> mark SolverBuildState dirty
  -> solve() rebuilds internal components once if needed
  -> solve executes the canonical route
```

The final shape may still become:

```text
SolverConfig
  -> build internal SolverPlan
  -> solve executes frozen phase ops
```

But this should wait until at least two or three seams have multiple real
implementations. Today, a full `StrategySpec -> StrategyKernel` framework is
still premature.

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

## Linear Solver Direction

`RiccatiSolver` should remain MiniSolver's primary linear-solver path for NMPC.
This is a deliberate design choice, not just the current implementation status.

The NMPC Newton/KKT system has a block time structure. The current Riccati path
uses that structure directly:

- inequality, slack, and barrier variables are locally eliminated into
  `Q_bar`, `R_bar`, `H_bar`, `q_bar`, and `r_bar`;
- the dynamics-coupled state/control system is solved by backward/forward
  Riccati sweeps;
- generated fused Riccati kernels can exploit model sparsity and fixed problem
  dimensions;
- the memory footprint and runtime behavior stay predictable for embedded and
  real-time use.

This is different from full condensing. MiniSolver should not first eliminate
all states into a dense control-only QP unless benchmark evidence shows that a
specific problem class benefits from it. For the current project goal, preserving
the sparse OCP structure is more important.

Other possible `LinearSolver` implementations are useful, but they are not the
main line today:

| Candidate | Role | Current priority |
| --- | --- | --- |
| `DenseKKTLDLTSolver` | Direct full KKT solve for reference/debug. | Future, only if needed for correctness oracle. |
| `SparseKKTLDLTSolver` | More generic sparse KKT backend. | Future, after real sparse non-OCP cases exist. |
| `CondensingSolver` | State-condensed dense/sparse QP route. | Future, benchmark-driven only. |
| `GpuRiccatiSolver` | GPU implementation of the Riccati path. | Future, after CPU Riccati correctness/performance stabilizes. |
| `PDLPSolver` / first-order solver | Large-scale approximate solves. | Low priority for current embedded NMPC scope. |

Near-term linear-solver work should therefore focus on the current Riccati path:

1. correctness against benchmark/reference solutions;
2. numerical robustness of regularization, inertia handling, and line search
   interaction;
3. generated/fused Riccati coverage where model structure is known;
4. MiniMatrix and small-matrix kernel optimization based on profiling;
5. GPU Riccati only after the CPU path is stable and measured bottlenecks justify
   it.

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
- At least one simple analytical problem checks default behavior against the
  reference config.

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

Status: lite version implemented.

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

- config mutation is cheap and defers component rebuild to the solve boundary.
- solve hot path executes frozen component choices currently represented in
  `SolverPlanInfo`.
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

1. Keep the current lite build-state boundary stable; do not expand it into a
   public strategy framework without a second real implementation.
2. Keep zero-malloc instrumentation on any future build-boundary or diagnostics
   changes. The current `set_config()` then `solve()` path is covered.
3. Extend reference/default checks only when the problem has a clear correctness
   oracle or stable reference metric.
4. Consider `RestorationKernel` only after a focused test shows the current
   restoration coupling blocks a correctness or debuggability fix.
5. Keep SOC geometry-aware logic in MiniModel/codegen. The next SOC step is full
   true/QP/SOC packet separation, not circle or ellipse branches in solver core.
6. Re-run nmpc-bench after any further phase-boundary change to ensure no
   runtime regression from structural refactors.
