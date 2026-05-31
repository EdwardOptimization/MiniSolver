# Scaling Contract

Status: draft

Owner modules:

- `MOD-ALG-EVAL`
- `MOD-CORE-CONFIG`
- `MOD-CORE-SEMANTICS`

Related modules:

- `MOD-SOLVER-ROUTE`
- `MOD-MODEL-CODEGEN`
- `MOD-MATRIX`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Constraint row scaling.
- Objective scaling.
- Problem scaling profile.
- Scaled vs unscaled residual reporting.
- Non-finite scale norm behavior.

Out of scope:

- Variable scaling.
- Physical-unit safety acceptance.
- Full Ruiz variable equilibration.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Scaling config fields | `MOD-CORE-CONFIG` | Bounds validated. |
| Cost/Hessian packet | Model evaluation | Used for objective scale. |
| Constraint residual/Jacobian packet | Model evaluation | Used for row scale. |
| Existing row scales | Trajectory/candidate | Reused when auto refresh is false. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| `objective_scale` | Active objective scaling factor. | Solver diagnostics and cost packet. |
| `constraint_row_scale` | Active per-row scale. | Constraint packets and unscaled residuals. |
| Scaled packets | Internal solver units. | Riccati/termination |
| `g_unscaled` | Raw true constraint residual. | Diagnostics/postsolve |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `KnotState::objective_scale` | `MOD-CORE-TYPES` | Per knot. |
| `KnotState::constraint_row_scale` | `MOD-CORE-TYPES` | Per knot. |
| `KnotState::g_unscaled` | `MOD-CORE-TYPES` | Per knot. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `SCALE-001` | Scaling changes internal packets only; user states, controls, parameters, and reported raw diagnostics remain in model units. | `covered` |
| `SCALE-002` | Constraint row scaling uses bounded inverse row norm when active. | `covered` |
| `SCALE-003` | Objective scaling uses bounded inverse Hessian Gershgorin row-sum when active. | `covered` |
| `SCALE-004` | Problem scaling activates the configured objective and constraint scaling profile. | `covered` |
| `SCALE-005` | Unscaled residual reporting must transform soft and hard residuals back to raw model units consistently. | `covered` |
| `SCALE-006` | NaN in scale norm reductions must be preserved. | `covered` |
| `SCALE-007` | Inf scale norm reductions may clamp to the minimum scale but must not hide later non-finite residuals. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Invalid scale bounds in config | `ApiStatus::InvalidArgument` or `NonFiniteValue` | Config boundary. |
| NaN scale value reaches residuals | `NUMERICAL_ERROR` at residual/postsolve boundary | Preserve evidence. |
| Inf packet norm | Minimum scale for scale factor, later non-finite packet still visible | Avoid neutral sanitize. |

## Numeric And Performance Constraints

- Scaling loops must be allocation-free.
- Scaling should not introduce variable transformations until a separate
  contract is accepted.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `SCALE-001` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingScalesConstraintPacketsOnly`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric` | `covered` |
| `SCALE-002` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric` | `covered` |
| `SCALE-003` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.HessianGershgorinScalesObjectivePacketOnly` | `covered` |
| `SCALE-004` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.ProblemScalingActivatesBoundedConstraintAndObjectiveScaling` | `covered` |
| `SCALE-005` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.L1SoftUnscaledResidualScalesSharedRelaxation`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.L2SoftUnscaledResidualScalesDualRelaxation`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.MixedSoftUnscaledResidualScalesSharedRelaxation` | `covered` |
| `SCALE-006` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.HessianGershgorinPropagatesNaNObjectiveScale`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingPropagatesNaNRowScale` | `covered` |
| `SCALE-007` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.HessianGershgorinOverflowUsesMinimumObjectiveScale` | `covered` |

## Open Gaps

- Replay/benchmark cases for scaled generated models remain useful but are not
  required for the current unit-level scaling contract.
