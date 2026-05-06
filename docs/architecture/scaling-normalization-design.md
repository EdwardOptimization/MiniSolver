# Scaling And Normalization Design

Date: 2026-05-03

Status: Stage 0 through Stage 4 implemented; Stage 5 minimal coordinate-scaling
hint shipped (termination metric only); full coordinate equilibration still
deferred

Related:

- [`solver-capability-adoption-plan.md`](solver-capability-adoption-plan.md)
- [`termination-design.md`](termination-design.md)
- [`../reviews/review-fix-plan-2026-05-02-deep.md`](../reviews/review-fix-plan-2026-05-02-deep.md)

## Purpose

MiniSolver currently assumes the user's variables, costs, and constraints are
already close to order one. That is not a safe assumption for real NMPC:
positions, angles, forces, distances, soft penalties, and obstacle margins can
easily differ by `1e3` or more. Poor scaling can appear as line-search failure,
excessive regularization, Riccati indefiniteness, weak warm-start behavior, or
incorrectly loose/tight termination.

This document defines the first scaling route. The goal is not to copy a general
NLP/QP solver. MiniSolver should add the smallest scaling layer that improves
fixed-size NMPC robustness while preserving generated-code performance and
solve-time zero heap allocation.

## Evidence From Mature Solvers

Local source survey:

- Ipopt exposes `nlp_scaling_method`, with `gradient-based`,
  `equilibration-based`, and `user-scaling` modes. Its documentation explicitly
  ties scaling to derivative magnitude and provides lower bounds on scaling
  factors so unscaled final violations are not hidden.
- OSQP computes setup-time problem scaling vectors for primal variables,
  constraints/duals, and the objective. It stores `D/Dinv`, `E/Einv`, and `c`,
  supports `scaled_termination`, and unscales solutions before reporting.
- Clarabel exposes equilibration controls with min/max scaling bounds and tests
  that those bounds are respected. This is useful for MiniSolver because
  unbounded scale factors are a numerical risk in embedded deployments.
- acados has a QP-scaling layer with `scaled_qp_in`, `scaled_qp_out`,
  constraint scaling factors, objective scaling factors, rescaling of QP
  solutions, and adaptive QP tolerances derived from the active scaling.

Common pattern:

1. Scaling is resolved outside the hot algebra kernels.
2. Internal residuals and subproblems may be scaled.
3. User-facing solution values remain in user units.
4. Diagnostics must make scaling visible.
5. Scale factors are bounded.

## Design Decision

MiniSolver will use a config-selected internal scaling plan resolved at the
build boundary. Users should still configure behavior through `SolverConfig` and
MiniModel/codegen should keep emitting physical model equations. Users should
not pass public scaling strategy objects or manual per-row scale metadata.

The implementation is staged:

1. Add a badly-scaled NMPC regression case and metrics before behavior changes.
   Implemented by `test_scaling_regressions`.
2. Add automatic constraint row scaling as the primary user-facing path.
   Implemented as `ConstraintScalingMethod::ROW_INF_NORM`.
3. Add diagnostics for scaled and unscaled residuals.
4. Add objective curvature scaling with a bounded Gershgorin kernel.
5. Add a conservative problem-level profile that combines row and objective
   scaling without transforming user variables or Riccati dynamics coordinates.
6. Extend to state/control/parameter coordinate scaling only after these
   conservative kernels are proven.

## Default Policy

The default `SolverConfig` keeps all scaling methods disabled:

```cpp
constraint_scaling = ConstraintScalingMethod::NONE;
objective_scaling = ObjectiveScalingMethod::NONE;
problem_scaling = ProblemScalingMethod::NONE;
```

This is deliberate. Scaling changes internal residual units, objective
curvature magnitude, merit/filter comparisons, and termination interpretation.
Keeping `NONE` as the default preserves compatibility and keeps the reference
path easy to reason about.

The recommended robust configuration for badly scaled NMPC problems is:

```cpp
config.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;
```

This is a config profile, not a public strategy/plugin. It composes the current
bounded row and objective kernels and leaves user variables in physical units.
If later benchmark coverage shows no regressions across ordinary well-scaled
models, this profile can become part of a future `Robust` preset.

## Ownership Boundary

`SolverConfig` owns built-in scaling-kernel selection:

```cpp
enum class ConstraintScalingMethod {
    NONE,
    ROW_INF_NORM
};

enum class ObjectiveScalingMethod {
    NONE,
    HESSIAN_GERSHGORIN
};

enum class ProblemScalingMethod {
    NONE,
    RUIZ_EQUILIBRATION
};

struct SolverConfig {
    ConstraintScalingMethod constraint_scaling = ConstraintScalingMethod::NONE;
    ObjectiveScalingMethod objective_scaling = ObjectiveScalingMethod::NONE;
    ProblemScalingMethod problem_scaling = ProblemScalingMethod::NONE;
};
```

The exact enum names are not fixed, but the boundary is:

- `SolverConfig` selects whether constraint scaling is disabled or which
  built-in constraint-scaling kernel is used.
- Objective and problem-level scaling use separate config enums. Each enum
  selects one built-in kernel; unsupported enum values or invalid bounds fail at
  the build boundary.
- MiniModel/codegen should not expose manual row-scale knobs for ordinary
  users. Generated models should encode physical residuals; solver config
  selects whether rows are scaled.
- Solver core owns applying scale factors consistently to residuals,
  linearized rows, Hessian/gradient terms, termination metrics, and diagnostics.
- Benchmarks own cross-solver comparisons and must report whether scaling was
  used.

Do not put dimension-dependent arrays directly into `SolverConfig`. MiniSolver
models are template-sized, while `SolverConfig` is currently a dimension-free
runtime object. Automatic row-scale storage lives in the fixed-size knot state;
candidate and SOC trajectories inherit the active scale so slack/dual variables
stay in the same scaled units during line search.

## Scaled Vs Unscaled Semantics

MiniSolver must not blur these quantities:

| Quantity | Use |
| --- | --- |
| Unscaled state/control/parameter values | User API, warm-start inputs, final solution, logs in physical units |
| Unscaled true constraint residuals | Final feasibility report and debugging |
| Scaled true constraint residuals | Filter/merit/convergence when scaling is enabled |
| Scaled QP linearization rows | Riccati/KKT search direction |
| Scaled SOC correction rows | SOC RHS and acceptance metrics |
| Unscaled objective value | User report |
| Scaled objective/gradient/Hessian | Internal linear solve when objective scaling is enabled |

The existing `g_true` vs `g_val/C/D` separation is a useful seam, but scaling
requires a clearer convention:

- `g_true` should remain available for unscaled final reporting.
- The rows consumed by Riccati and line search may be scaled according to the
  active scaling plan.
- Diagnostics should expose both scaled and unscaled worst rows so a user can
  tell whether scaling is helping or hiding a modeling problem.

## Tolerance Semantics

`tol_con`, `tol_dual`, and `tol_mu` are interpreted in the solver's internal
units:

- With scaling disabled, internal constraint residual units equal user model
  units.
- With constraint or problem scaling enabled, `tol_con` applies to the scaled
  constraint residual and unscaled dynamics defects. `SolverInfo::primal_inf`
  reports this internal metric.
- `SolverInfo::unscaled_primal_inf` reports the raw model constraint residual
  plus slacks transformed back to user units where row scaling is active.
- `tol_dual` applies to stationarity residuals built from the active internal
  cost and constraint packets. Objective scaling can therefore change the
  stationarity residual magnitude by design.
- `tol_mu` applies to the primal-dual complementarity quantities in the active
  internal slack/dual units.

User-facing state, control, parameter values, and `get_stage_cost()` remain
unscaled. A final report should always inspect both `primal_inf` and
`unscaled_primal_inf` when scaling is enabled: the former explains solver
termination, while the latter explains physical feasibility.

Current limitation: dynamics defects are not coordinate-scaled. Full
state/control/parameter scaling is a later stage because it must transform
`dx/du`, dynamics Jacobians, Riccati feedback gains, warm starts, SOC
corrections, and diagnostics consistently.

## Stage 0: Badly-Scaled Case And Metrics

Before implementing scaling, add a focused regression or benchmark case that
captures the current failure mode. The test should not depend on external
solvers.

Candidate case:

- A small multiple-shooting double-integrator or bicycle model.
- Two path constraints with intentionally different natural scales, e.g. one
  distance-like row near `1e-3` and one position/force-like row near `1e3`.
- The same physical feasible set should be expressible in a well-scaled and a
  badly-scaled formulation.

Required metrics:

- Solver status and termination reason.
- Iteration count.
- Accepted alpha and backtracking count.
- Regularization escalation count.
- Restoration/SOC attempted and accepted counts.
- Scaled and unscaled primal infeasibility.
- Scaled and unscaled complementarity residual.
- Final unscaled feasibility.
- Solve-time allocation count.

If some metrics are not yet queryable, add diagnostics first or explicitly mark
the benchmark as limited. Do not infer scaling success from status alone.

Implementation note: `test_scaling_regressions` now fixes the baseline that two
equivalent hard-constraint rows can produce a roughly `1000x` difference in
`SolverInfo::primal_inf` when scaling is disabled.

## Stage 1A: Automatic Constraint Row Scaling

This is the first user-facing behavior change. It does not change user variable
units and does not require transforming state dynamics. The user enables it with:

```cpp
config.constraint_scaling = ConstraintScalingMethod::ROW_INF_NORM;
```

The solver computes a bounded row scale from the active constraint packet,
stores it on each knot, and applies the same scale to `g_val`, `g_true`, `C`,
`D`, and SOC residual overrides. The first automatic profile only down-scales
large rows; it does not scale up tiny rows until tolerance semantics are
designed in Stage 2.

Core rules:

- The same row scale must apply to true residual, QP linearization, and SOC
  correction semantics.
- Scale factors are bounded to a finite range, initially `[1e-4, 1e4]`, matching
  the mature-solver pattern of bounded equilibration.
- Candidate and SOC evaluations must reuse the active row scale rather than
  recomputing it independently; otherwise `g`, slack, and dual values drift into
  different units during line search.
- With scaling disabled, current behavior remains unchanged.

Implementation note: Stage 1A scales `g_val`, `g_true`, `C`, and `D` for
internal Riccati/globalization metrics while preserving `g_unscaled` for
diagnostics. `SolverInfo::primal_inf` is the internal active feasibility metric;
`SolverInfo::unscaled_primal_inf` exposes the same active residual, including
current slack variables, transformed back to model units.
`test_scaling_regressions` covers both the unscaled baseline and automatic
normalization. MiniModel generated models no longer emit or accept public
`row_scale=` metadata.

Validation:

- Badly-scaled case improves or becomes no worse in status, iteration count,
  backtracking, regularization, and final unscaled feasibility.
- Existing well-scaled tests remain unchanged within tolerances.
- `test_memory` confirms no solve-time heap allocation.

## Stage 2: Diagnostics And Termination Contract

Scaling changes the meaning of tolerances. The termination layer must make this
explicit.

Recommended contract:

- Internal acceptance uses scaled residuals when scaling is enabled.
- `SolverInfo` reports scaled residual summaries and unscaled residual
  summaries.
- Public tolerance documentation states whether each tolerance is interpreted in
  scaled units or user units under each scaling mode.
- Final feasibility reporting always includes unscaled constraint violation.

This stage is tied to N-OBS-1 and N-PREC-2. Avoid adding a broad diagnostics
framework before the scaling fields are known; extend the current fixed-size
metrics/state first.

## Stage 3: Objective Curvature Scaling

Objective scaling is implemented as:

```cpp
config.objective_scaling = ObjectiveScalingMethod::HESSIAN_GERSHGORIN;
```

The kernel computes a Gershgorin upper bound for the local objective Hessian
packet:

```text
H = [ Q  H^T ]
    [ H   R  ]

row_bound = max_i sum_j abs(H_ij)
objective_scale = clamp(1 / max(1, row_bound),
                        objective_scale_min,
                        objective_scale_max)
```

It then scales `cost`, `q`, `r`, `Q`, `R`, and `H` by that scalar before the
linear solver and globalization consume them. `cost_unscaled` keeps the raw
objective value, and `get_stage_cost()` continues to report user-unit cost.

This is intentionally a scalar objective scaling, not residual-level
least-squares weighting. Least-squares residual weighting still belongs in
MiniModel/codegen through `add_residual`.

## Stage 4: Conservative Problem-Level Scaling Profile

Problem scaling is implemented as:

```cpp
config.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;
```

The current profile is conservative:

- it activates automatic constraint row scaling;
- it activates objective Gershgorin scaling;
- it records `problem_scaling_active` in `SolverInfo`;
- it does not transform state/control/parameter coordinates;
- it does not rescale the Riccati dynamics recursion.

This is a bounded Ruiz-inspired equilibration profile rather than full
coordinate equilibration. That choice is deliberate: full variable scaling would
change `dx/du`, dynamics defects, Riccati feedback gains, SOC corrections,
restoration, warm-start deltas, and unscaled diagnostics in one patch. That is
too much semantic surface without a dedicated benchmark proving the need.

Manual benchmark evidence from 2026-05-03:

```text
cmake -S . -B .build
cmake --build .build --target scaling_bench -j2
./.build/scaling_bench 50 5

Eigen backend, CarModel obstacle, N=50:
NONE                  success 100%, 1.605 ms, 87 iters, primal 1.78e-15, raw 1.78e-15, dual 6.81e-04
ROW_INF_NORM          success 100%, 1.656 ms, 87 iters, primal 8.88e-16, raw 1.78e-15, dual 1.23e-03
HESSIAN_GERSHGORIN    success 100%, 0.507 ms, 24 iters, primal 1.78e-15, raw 1.78e-15, dual 7.27e-05
RUIZ_EQUILIBRATION    success 100%, 0.527 ms, 24 iters, primal 8.88e-16, raw 1.78e-15, dual 7.27e-05

cmake -S . -B .build_custom -DUSE_EIGEN=OFF -DUSE_CUSTOM_MATRIX=ON
cmake --build .build_custom --target scaling_bench -j2
./.build_custom/scaling_bench 50 5

Custom MiniMatrix backend, same case:
NONE                  success 100%, 1.915 ms, 87 iters, primal 1.78e-15, raw 1.78e-15, dual 1.22e-03
ROW_INF_NORM          success 100%, 1.943 ms, 87 iters, primal 8.88e-16, raw 1.78e-15, dual 6.78e-04
HESSIAN_GERSHGORIN    success 100%, 0.591 ms, 24 iters, primal 8.88e-16, raw 8.88e-16, dual 7.27e-05
RUIZ_EQUILIBRATION    success 100%, 0.610 ms, 24 iters, primal 8.88e-16, raw 1.78e-15, dual 7.27e-05
```

Conclusion from this benchmark:

- Objective scaling is the dominant improvement in this mixed-weight CarModel
  case.
- Row scaling is neutral on this particular case because the constraint rows are
  already similarly scaled.
- `RUIZ_EQUILIBRATION` matches the objective-scaling iteration improvement while
  also enabling row scaling for badly scaled constraints.
- The default remains `NONE` until ordinary and pathological benchmark coverage
  is broader; `RUIZ_EQUILIBRATION` is the recommended robust opt-in.

## Stage 5: State, Control, And Parameter Coordinate Scaling

This stage is deeper and should not be mixed with row scaling.

Required transformations for the *full* version:

- Variable scaling changes the interpretation of `dx/du`, dynamics Jacobians
  `A/B`, gradients, Hessians, bounds, warm-start deltas, and Riccati blocks.
- Parameter scaling must not leak into user-facing `set_parameter()` values.
- Objective scaling affects merit/filter comparisons, dual residuals, and
  regularization thresholds.
- Least-squares residual scaling from `add_residual` should be handled at the
  MiniModel/codegen residual level, not as an after-the-fact Hessian patch.

The full version still needs its own red benchmark and exact transformation
tests before it can ship.

### Stage 5a: Termination-only coordinate-scaling hint (shipped)

The current minimal-viable subset is a *hint* that never rescales primal
variables, the search direction, or the Riccati recursion. It only changes
how the dual-stationarity infinity norm is reported and compared:

```cpp
config.coordinate_scaling = CoordinateScalingMethod::USER_SUPPLIED;

solver.set_state_scale("x_pos", 100.0);   // x_pos has typical magnitude ~100
solver.set_control_scale("u_force", 1.0); // u_force is already O(1)
```

Convention: the user-supplied scale `s_i` represents the typical magnitude of
coordinate `i` in model units. With `USER_SUPPLIED` active, the solver
evaluates `dual_inf = max_i |r_i| * control_scale_i` so coordinates with
naturally large gradients do not mask convergence on coordinates with small
gradients.

Contract guarantees:

- `CoordinateScalingMethod::NONE` (default) keeps the legacy unweighted
  inf-norm bit-for-bit; the `test_coordinate_scaling.NoneStrategyKeepsBaselineDualInfBitForBit`
  test pins this guarantee.
- API setters return `ApiStatus` and reject non-finite values, indices out of
  range, and scales outside `[config.coordinate_scale_min,
  config.coordinate_scale_max]` (defaults `[1e-6, 1e6]`).
- `SolverInfo::coordinate_scaling_active` is `true` only when the strategy is
  `USER_SUPPLIED` *and* at least one scale differs from `1.0`. All-unity
  scales fall back to legacy semantics.
- The hint is consumed exclusively by `compute_dual_infeasibility_` (iteration
  loop) and the postsolve dual-residual evaluation. It does **not** change
  Riccati, line search, SOC, restoration, or warm-start delta application.

Limitations and follow-ups:

- Stage 5a only weights the control-stationarity vector. State and parameter
  scales are stored on the solver instance but currently only validated and
  surfaced via getters; they are wired up here to keep the public API stable
  for the eventual full-equilibration version.
- Full coordinate equilibration (rescaling `dx/du`, `A/B`, Hessian blocks,
  warm-start deltas, etc.) remains gated on a focused benchmark proving the
  extra complexity is justified.

## Stage 6: Additional Scaling Kernels

Future scaling kernels should stay conservative and bounded. They should be
added only after a focused benchmark or regression shows that row-inf-norm
constraint scaling is insufficient.

Implemented objective scaling kernel:

```text
HESSIAN_GERSHGORIN:
    max_abs_eig = Gershgorin upper bound of stage Hessian blocks
    obj_scale = min(1.0, max_allowed_eig / max_abs_eig)
```

Implemented conservative problem-level profile:

```text
RUIZ_EQUILIBRATION:
    activate ROW_INF_NORM constraint row scaling
    activate HESSIAN_GERSHGORIN objective scaling
    do not transform user variables or Riccati dynamics coordinates
```

Open design points:

- Whether a future `RUIZ_COORDINATE_EQUILIBRATION` kernel is worth the extra
  workspace and transformation complexity for small fixed-size NMPC.
- Whether additional scaling kernels should be default in `Default` profile or
  only enabled in a future `Robust` profile.

Do not enable additional scaling kernels by default until the badly-scaled case,
ordinary well-scaled cases, and nmpc-bench smoke cases all show stable behavior.

## Non-Goals

- No public OOP scaling plugin framework.
- No unit system in MiniModel yet.
- No generic dense Ruiz equilibration for the full KKT matrix in the first
  implementation.
- No hidden change to user-facing solution units.
- No benchmark claims without accuracy, status, and allocation evidence.

## Implementation Order

1. Add `test_scaling_regressions` or a small benchmark-style test that captures
   the badly-scaled case and current metrics.
2. Add scaling diagnostics needed by that test if they are missing.
3. Add `ConstraintScalingMethod` with `NONE` default and build-boundary validation.
4. Add `ObjectiveScalingMethod` and `ProblemScalingMethod` as explicit config
   categories.
5. Add automatic constraint row scales.
6. Apply row scaling consistently to true/QP/SOC constraint paths.
7. Add bounded objective Gershgorin scaling while preserving unscaled user cost.
8. Add conservative problem-level scaling as a config profile that composes the
   implemented row and objective kernels.
9. Update termination diagnostics to report scaled and unscaled residuals.
10. Run focused scaling tests, `test_memory`, and full `ctest`.
11. Only then consider coordinate-level variable scaling.

## Review Gates

Before code lands:

- A red badly-scaled case must exist.
- The patch must not add solve-time heap allocation.
- Scale application must be centralized; no scattered ad-hoc multipliers across
  line search, Riccati, termination, and restoration.
- The patch must preserve the MiniSolver rule: users choose behavior through
  `SolverConfig`; internal kernels and scaling plans are not public plugin API.
- The final report must state both scaled and unscaled feasibility.
