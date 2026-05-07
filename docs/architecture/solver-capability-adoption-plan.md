# Solver Capability Adoption Plan

Date: 2026-05-02

Status: active roadmap

Related:

- [`warm-start-strategy.md`](warm-start-strategy.md)
- [`solver-refactor-plan.md`](solver-refactor-plan.md)
- [`../testing/testing-matrix.md`](../testing/testing-matrix.md)

## Purpose

Solver landscape research is input, not the MiniSolver product plan. This
document converts that research into the capabilities MiniSolver should absorb
from mature NMPC, NLP, QP, and conic solvers.

The goal is not to make MiniSolver a clone of Ipopt, CasADi, acados, HPIPM, or
OSQP. MiniSolver should stay focused on fixed-size NMPC / optimal-control
problems with generated derivatives, Riccati-structured linear solves, and
solve-time allocation discipline.

## Adoption Principles

1. Keep the core route NMPC/IPM/Riccati-specific.
2. Absorb mechanisms that improve real NMPC robustness, not broad problem
   domains.
3. Add instrumentation before large algorithm changes.
4. Treat scaling, warm-start, and diagnostics as infrastructure, not optional
   polish.
5. Add profiles and modes only when they make behavior clearer to users.
6. Validate every performance-sensitive change with a red benchmark and
   before/after data.

## Priority Stack

### P0: Scaling And Normalization

Source systems:

- OSQP: Ruiz equilibration and residual scaling discipline.
- Ipopt: NLP scaling, slack/bound push, and scale-aware stopping criteria.
- acados/HPIPM: conditioning-aware QP setup and regularization.

MiniSolver should add explicit scaling before deeper RTI or Riccati-mode work.
Poor scaling can look like line-search failure, Riccati indefiniteness,
excessive regularization, or warm-start failure. Without scaling diagnostics,
those symptoms are hard to separate.

First implementation target:

- Per-state, per-control, per-parameter scaling.
- Per-constraint row scaling / normalization.
- Cost/residual scaling for `add_residual`-style least-squares terms.
- Clear distinction between scaled residuals used internally and unscaled
  residuals reported to the user.
- Consistent scaling across true constraints, QP linearization constraints, and
  SOC correction constraints.

Validation:

- Add a deliberately badly-scaled NMPC test or benchmark.
- Compare success rate, iterations, line-search backtracking count,
  regularization escalation count, and final unscaled feasibility.

### P0: Warm-Start Strategy Group

Source systems:

- Ipopt: explicit warm-start mode, interior push, and target-mu concepts.
- HPIPM/acados: graded primal/dual warm starts and same-structure repeated
  solves.
- OSQP: repeated-solve state reuse and update/factorization separation.

MiniSolver's accepted direction is documented in
[`warm-start-strategy.md`](warm-start-strategy.md).

Recommended general MPC preset:

```cpp
initialization = InitializationMode::REUSE_PRIMAL_DUAL;
warm_start_barrier = WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP;
warm_start_regularization = WarmStartRegularizationMode::RESET_TO_REG_INIT;
barrier_strategy = BarrierStrategy::ADAPTIVE;
```

This keeps ordinary `solve()` useful for repeated MPC solves without requiring
the user to enable a fixed one-iteration RTI mode.

Future work:

- Add stronger benchmark coverage across changing references, parameters, and
  active constraints.
- Add explicit failure fallback policy: invalid warm-start data must fall back
  to repaired/cold interior initialization, not produce a degraded solve state.

### P0: Structured Diagnostics

Source systems:

- Clarabel and HiGHS expose structured solver info instead of relying only on
  logs.
- Ipopt reports detailed iteration state, restoration state, and linear solver
  failures.

MiniSolver should make solver behavior queryable, not just printable. This is
especially important once scaling, warm-start, RTI-lite, SOC, and fallback
Riccati modes coexist.

Recommended diagnostic fields:

- Requested backend and actual backend.
- Whether the solve used a degraded fallback path.
- Linear solver status and factorization status.
- Regularization value and escalation count.
- Barrier parameter history summary.
- Line-search type, accepted alpha, and backtracking count.
- SOC attempted / accepted / rejected counts.
- Restoration attempted / accepted / rejected counts.
- Warm-start mode and whether the warm-start was repaired or rejected.
- Scaled and unscaled primal/dual/complementarity residual summaries.

This should extend existing metrics/state first. A public `SolverDiagnostics`
object is justified only after the internal fields are stable.

### P1: Solver Profiles

Source system:

- HPIPM exposes mode presets such as speed, balance, and robust.

MiniSolver should expose behavior profiles rather than forcing users to tune a
large number of boolean and numeric options by hand.

Candidate profiles:

| Profile | Purpose | Expected behavior |
| --- | --- | --- |
| `Reference` | Debug and regression oracle | Conservative, simple, minimal heuristics |
| `Default` | General NMPC use | Stable warm-start, scaling, adaptive barrier, filter/merit defaults |
| `Speed` | Low-latency well-scaled MPC | More aggressive steps, limited recovery, fewer safeguards |
| `Robust` | Difficult or poorly-scaled problems | Stronger scaling, more regularization, restoration/SOC enabled |

Profiles should be presets over `SolverConfig`, not a new solver hierarchy.
Users must still be able to override individual config fields.

Implementation note: profiles and individual config fields should be resolved
into an internal execution plan at construction, `set_config()`, or first
`solve()` after a dirty config. This is a runtime resolve step that selects from
kernel implementations already compiled into the binary; it is not runtime code
generation. The hot solve loop should prefer static kernel paths over repeated
virtual or function-pointer dispatch when the added implementation surface is
justified.

### P1: RTI-Lite

Source system:

- acados separates preparation and feedback phases and offers AS-RTI levels.

MiniSolver should not jump directly to a full AS-RTI framework. A first useful
step is RTI-lite: reuse selected linearization/factorization data when it is
safe, and force a refresh when the reuse assumptions break.

Candidate safety gates:

- Same model dimensions, horizon, backend, integrator, scaling, and active
  solver profile.
- `linearization_age <= max_linearization_age`.
- Parameter/reference change below a configured threshold.
- Primal defect and constraint violation below a reuse threshold.
- No NaN/Inf, restoration, SOC rejection, line-search collapse, or large
  regularization escalation in the previous solve.
- Force refresh after any config change that rebuilds solver components.

Initial implementation should be explicit and conservative:

- Track `linearization_age`.
- Add diagnostics showing whether a solve reused or refreshed linearization.
- Benchmark repeated MPC solves with small changes.

### P1: Riccati Robustness Modes

Source systems:

- HPIPM has multiple Riccati paths: square-root, classical, and robust LQ-style
  factorization variants.
- Ipopt and HiGHS make factorization failures visible and escalate
  regularization/perturbation.

MiniSolver's current Riccati path is appropriate for NMPC and should remain the
main path. Additional modes should be added only when diagnostics or benchmarks
show the current path is the bottleneck or failure source.

Candidate modes:

- Fast/default Riccati.
- More robust regularized Riccati.
- Square-root-style path if positive-definite structure can be guaranteed.
- Debug/reference dense KKT path only if needed for correctness investigation.

Every fallback must be visible in diagnostics. Silent subproblem changes should
not be reported as normal solves.

#### Stage 1 (shipped): inertia-correction visibility

`RiccatiRobustMode::STANDARD` (default) and `INERTIA_AWARE_DIAGNOSTICS` are
implemented today. They share the existing Riccati algorithm — no square-root
or LDLT rewrite — and only differ in how visible the existing fallbacks are:

- The general-path SPD escalation, the small-Nu freeze fallback, and the
  `SATURATION` / `IGNORE_SINGULAR` repair sweeps each bump
  `LinearSolveResult::riccati_indefinite_blocks` and update
  `riccati_max_diagonal_perturbation` whenever they fire.
- The small-Nu freeze fallback additionally sets
  `LinearSolveResult::degraded_step` and bumps
  `LinearSolveResult::degraded_riccati_freeze_count`. This is the
  pre-existing N-DEG-1 contract: a frozen `du` is a degraded step in
  *both* modes and cannot be suppressed.
- `Solver::record_linear_solver_diagnostics_` accumulates the counters into
  `SolverInfo::riccati_indefinite_blocks`,
  `SolverInfo::riccati_max_diagonal_perturbation`, and
  `SolverInfo::degraded_riccati_freeze_count` for every solve, regardless of
  the mode. `SolverInfo::degraded_step` is always set when the freeze
  fallback fires.
- `INERTIA_AWARE_DIAGNOSTICS` additionally sets `SolverInfo::degraded_step`
  to true whenever `riccati_indefinite_blocks > 0` — i.e. any of the four
  fallback paths fired, not only the freeze path. This is the only
  per-mode behavioural difference; in `STANDARD` the non-freeze inertia
  events update the counters but leave `degraded_step` untouched.

Square-root and `FACTORIZATION_MODIFY` paths remain explicit non-goals until a
benchmark-confirmed failure that this stage cannot diagnose appears. The
diagnostic stage is the contract anchor that future stages must keep
populated, so the visibility invariant survives any future Riccati rewrite.

### P0: Constraint Scaling / Per-Row Normalization

This is listed separately from general scaling because it is the most likely
source of real NMPC failures.

NMPC constraints often mix meters, radians, normalized forces, collision
distance margins, actuator limits, and soft-constraint penalties. A row with
natural scale `1e-3` and a row with natural scale `10` should not contribute to
filter acceptance, merit penalties, or KKT residual checks as if they were
equivalent.

First implementation target:

- Generated or user-provided constraint row scales.
- Default automatic row normalization based on Jacobian/residual magnitude when
  safe.
- Diagnostics that report the worst scaled and unscaled constraint rows.
- Tests proving that filter, merit, SOC, and convergence checks use consistent
  scaled semantics.

### P2: Cost And Constraint Modeling Boundary

Source systems:

- CasADi: codegen and function boundaries.
- CROCODDYL: residual + activation cost composition.

MiniSolver should continue moving geometry- and model-specific knowledge into
`MiniModel` / codegen, not the solver core.

Current direction:

- `add_residual` is the right high-level interface for least-squares costs.
- True constraint residuals, QP linearization residuals, and SOC correction
  residuals should remain semantically distinct.
- Circle/ellipse/obstacle-specific projected-boundary logic belongs in
  MiniModel/codegen-generated packets, not in the solver core.

## Explicit Non-Goals

Do not absorb these in the near term:

- A full general NLP framework like Ipopt.
- A general symbolic system like CasADi.
- Homogeneous self-dual embedding as the main MiniSolver route.
- Simplex, MIP, or active-set LP/QP machinery from HiGHS.
- ALM/PANOC as the default solve route.
- Public solver-phase plugins or OOP strategy objects. MiniSolver users should
  configure behavior through `SolverConfig`; phase strategies, kernels, and
  plans remain internal implementation details.

## Implementation Order

1. Extend roadmap/docs to reflect this capability adoption plan.
2. Design and test constraint scaling / normalization first.
3. Expand structured diagnostics enough to evaluate scaling and warm-start
   behavior.
4. Stabilize solver profiles as config presets.
5. Add RTI-lite only after diagnostics can explain reuse vs refresh decisions.
6. Add Riccati robustness modes only with benchmark evidence.

## Validation Contract

Each adoption item needs:

- A red test or benchmark that fails or underperforms before the change.
- A minimal implementation.
- Before/after numbers for correctness and runtime when relevant.
- Diagnostics showing which path was used.
- Documentation of default behavior and expert tuning knobs.
