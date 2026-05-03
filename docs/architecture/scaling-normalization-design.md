# Scaling And Normalization Design

Date: 2026-05-03

Status: design accepted, implementation not started

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
MiniModel/codegen metadata; they should not pass public scaling strategy
objects.

The first implementation should be staged:

1. Add a badly-scaled NMPC regression case and metrics before behavior changes.
2. Add model-provided constraint row scaling.
3. Add diagnostics for scaled and unscaled residuals.
4. Extend to state/control/parameter scaling only after row scaling is proven.
5. Add automatic scaling only after explicit/model-provided scaling has tests.

## Ownership Boundary

`SolverConfig` owns mode selection:

```cpp
enum class ScalingMode {
    NONE,
    MODEL_PROVIDED,
    AUTO_CONSTRAINT_ROWS
};
```

The exact enum names are not fixed, but the boundary is:

- `SolverConfig` selects whether scaling is disabled, model-provided, or
  automatically computed.
- MiniModel/codegen owns model-specific scale metadata when scales are known
  from units or modeling semantics.
- Solver core owns applying scale factors consistently to residuals,
  linearized rows, Hessian/gradient terms, termination metrics, and diagnostics.
- Benchmarks own cross-solver comparisons and must report whether scaling was
  used.

Do not put dimension-dependent arrays directly into `SolverConfig`. MiniSolver
models are template-sized, while `SolverConfig` is currently a dimension-free
runtime object. Scale storage should live in generated model traits or
fixed-size solver workspace selected by the scaling plan.

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

## Stage 1: Model-Provided Constraint Row Scaling

This is the first behavior change because it is the lowest-risk scaling layer.
It does not change user variable units and does not require transforming state
dynamics.

Expected shape:

```cpp
struct ModelScalingTraits {
    static constexpr bool has_constraint_row_scales = true;
    static constexpr std::array<double, NC> constraint_row_scales = {...};
};
```

Generated models may emit equivalent fixed-size metadata. Hand-written models
can opt in by defining the trait or static member expected by `model_traits`.

Core rules:

- A scale value multiplies one constraint row.
- The same row scale must apply to true residual, QP linearization, and SOC
  correction semantics.
- Scale factors are clamped to a finite range, initially `[1e-4, 1e4]`, matching
  the mature-solver pattern of bounded equilibration.
- Invalid scale values are configuration/model errors, not silently repaired in
  the hot path.
- With scaling disabled, current behavior remains unchanged.

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

## Stage 3: State, Control, Parameter, And Objective Scaling

This stage is deeper and should not be mixed with row scaling.

Required transformations:

- Variable scaling changes the interpretation of `dx/du`, dynamics Jacobians
  `A/B`, gradients, Hessians, bounds, warm-start deltas, and Riccati blocks.
- Parameter scaling must not leak into user-facing `set_parameter()` values.
- Objective scaling affects merit/filter comparisons, dual residuals, and
  regularization thresholds.
- Least-squares residual scaling from `add_residual` should be handled at the
  MiniModel/codegen residual level, not as an after-the-fact Hessian patch.

This stage needs its own red benchmark and exact transformation tests.

## Stage 4: Automatic Scaling

Automatic scaling should be conservative and bounded.

Candidate rule for constraint rows:

```text
row_norm = max(inf_norm(C_row), inf_norm(D_row), abs(g_row), scale_floor)
scale = clamp(1.0 / row_norm, min_scale, max_scale)
```

Open design points:

- Whether automatic scales are recomputed once at solver build, once per solve,
  or only after a large model/reference change.
- Whether automatic scaling should be default in `Default` profile or only in
  `Robust` profile at first.
- How to avoid oscillating scale factors across repeated MPC solves.

Do not enable automatic scaling by default until the badly-scaled case, ordinary
well-scaled cases, and nmpc-bench smoke cases all show stable behavior.

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
3. Add `ScalingMode` with `NONE` default and build-boundary validation.
4. Add model-provided constraint row scales.
5. Apply row scaling consistently to true/QP/SOC constraint paths.
6. Update termination diagnostics to report scaled and unscaled residuals.
7. Run focused scaling tests, `test_memory`, and full `ctest`.
8. Only then consider automatic row scaling and variable scaling.

## Review Gates

Before code lands:

- A red badly-scaled case must exist.
- The patch must not add solve-time heap allocation.
- Scale application must be centralized; no scattered ad-hoc multipliers across
  line search, Riccati, termination, and restoration.
- The patch must preserve the MiniSolver rule: users choose behavior through
  `SolverConfig`; internal kernels and scaling plans are not public plugin API.
- The final report must state both scaled and unscaled feasibility.
