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

### P2: Barrier And Globalization Strategy Scope

Source systems:

- Ipopt: monotone/adaptive barrier updates, quality-function centering, filter
  line search, watchdog, and restoration.
- acados/HPIPM: real-time SQP/RTI globalization choices and aggressive
  warm-started steps.
- Clarabel/ECOS/SCS: homogeneous self-dual embedding for conic certificate
  workflows, which is not MiniSolver's current NMPC/Riccati route.

MiniSolver should not become a catalog of every NLP globalization variant. The
current strategy families cover the important fixed-size NMPC/IPM skeleton:

- Barrier update: `MONOTONE`, `ADAPTIVE`, and `MEHROTRA`.
- Step acceptance: `NONE`, `MERIT`, and `FILTER`.

Near-term work should strengthen these existing strategies before adding new
ones:

- Improve the robustness of the existing Mehrotra path, including clearer
  diagnostics, better fallback behavior, and benchmark evidence for any change
  to centering or affine-step handling.
- Keep filter diagnostics and SOC/restoration triggers explainable before
  extending the filter theory surface.
- Keep `MONOTONE + MERIT` as the simple reference-style path and avoid
  weakening it with advanced heuristics.

Future additions require a concrete failure case or benchmark:

- Watchdog / nonmonotone filter support is the most plausible next
  globalization feature for warm-started NMPC, because it can reduce overly
  conservative backtracking while preserving a rollback path.
- Trust-region globalization is useful for bad initial guesses and strongly
  nonlinear problems, but it changes the step/globalization contract and should
  be designed as a separate phase, not patched into line search.
- Quality-function or oracle-style barrier updates may be useful if
  `ADAPTIVE` and `MEHROTRA` repeatedly fail on benchmarked cases, but they
  should not be added as another default knob without evidence.

Deferred or out-of-scope for the current route:

- Homogeneous self-dual embedding as a primary solve route.
- A broad menu of named barrier schedules that duplicate the existing three
  barrier families.
- Pareto-frontier filter history, funnel methods, or penalty-filter hybrids
  without a real NMPC failure that current `MERIT`/`FILTER` cannot explain.

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
- `subject_to_quad(..., rhs_mode="norm2")` belongs to MiniModel/codegen. It
  represents `sqrt((x-c)^T Q (x-c) + eps) <= rhs` or the corresponding outside
  form. Numeric `rhs` and `Q` are checked at generation time; symbolic or
  parameter-dependent `rhs/Q` are a model contract, so users must ensure
  `rhs >= 0` and `Q` is PSD over the operating domain.
- Circle/ellipse/obstacle-specific projected-boundary logic belongs in
  MiniModel/codegen-generated packets, not in the solver core.

### P2: External Solver Lessons To Track

These projects are useful sources of mechanisms, not roadmaps to copy. Absorb
only the parts that fit MiniSolver's fixed-size C++ NMPC route.

| Project family | What MiniSolver can learn | Boundary |
| --- | --- | --- |
| `iit-DLSLab/mpx` | GPU Riccati / KKT parallel-scan formulation, especially how a time recursion can be represented as associative operators before mapping to CUDA or another accelerator. | Treat `GPU_MPX` / `GPU_PCR` as future parallel-scan Riccati research until a CPU reference scan, correctness tests, and batched or long-horizon benchmarks justify implementation. |
| `cuOSQP` | Backend honesty: GPU support should be explicit, separately tested, and never silently fall back to CPU when a GPU backend is requested. | Do not replace MiniSolver's primal-dual NMPC/IPM route with ADMM/QP-first logic. |
| `cuRobo` | Batched seeds, collision/geometric oracles, and GPU-friendly robotics kernels can be more valuable than a single-problem GPU linear solve for robotics workloads. | MiniSolver should not become a motion-planning stack; keep geometry in MiniModel/callback/oracle layers. |
| `ALTRO` / `ALTRO-C` | Trajectory initialization, augmented-Lagrangian/iLQR-style profiles, and conic/norm constraint handling are useful references for future profiles and MiniModel constraint semantics. | Do not mix an ALTRO-like AL/iLQR route into the current primal-dual IPM core without a separate profile and benchmark evidence. |
| `Trajax` | JAX-style batching and differentiable optimal-control experiments are useful for Python-side references, replay generation, and research prototypes. | Do not make the C++ hot path depend on JAX. |
| `JAXopt` | Solver-state APIs, implicit differentiation, and batchable optimizer interfaces are useful design references for future Python tooling. | Treat it as design input, not a runtime dependency or maintenance foundation. |

Potential adoption sequence:

1. Document `GPU_MPX` and `GPU_PCR` as future parallel-scan Riccati backends,
   not supported GPU features.
2. Define a CPU reference parallel-scan Riccati contract before CUDA work.
3. Extend callback/oracle examples and replay cases for batched robotics and
   collision-heavy workloads before deciding whether GPU acceleration matters.
4. Build conic/norm-constraint corpus coverage around MiniModel semantics before
   adding any ALTRO-like profile.

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
