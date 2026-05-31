# Row Scaling Contract

Status: draft

Owner modules:

- `MOD-ALG-EVAL`
- `MOD-CORE-SEMANTICS`

Related modules:

- `MOD-CORE-TYPES`
- `MOD-ALG-LS`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Constraint row scale storage, refresh/reuse rules, transformation of residuals
and Jacobians, unscaled residual reporting, and candidate/SOC consistency.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `SCALE/CON-001` | Active row scaling multiplies `g_val`, `g_true`, `C`, and `D` by the row scale. | `covered` |
| `SCALE/CON-002` | `g_unscaled` preserves the raw true residual before row scaling. | `covered` |
| `SCALE/CON-003` | Row scales are refreshed only when the caller requests automatic scaling refresh. | `covered` |
| `SCALE/CON-004` | Candidate and SOC paths reuse active row scales unless explicitly refreshing. | `covered` |
| `SCALE/CON-005` | Unscaled hard residual transforms slack contribution by inverse row scale. | `covered` |
| `SCALE/CON-006` | Unscaled soft residual transforms relaxation terms by inverse row scale consistently for L1, L2, and mixed rows. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Raw constraints/Jacobians, row scale config, active row scales |
| Outputs | Scaled packets, `g_unscaled`, `constraint_row_scale`, unscaled diagnostics |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Invalid row scale bounds | Config rejection. |
| Non-finite row norm | Preserve NaN or clamp Inf according to scaling contract. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `SCALE/CON-001` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingScalesConstraintPacketsOnly` | `covered` |
| `SCALE/CON-002` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric` | `covered` |
| `SCALE/CON-003` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.RowScalingReusesExistingScaleWhenRefreshDisabled` | `covered` |
| `SCALE/CON-004` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.RowScalingReusesExistingScaleWhenRefreshDisabled`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.SocConstraintRowScalingReusesCandidateScale` | `covered` |
| `SCALE/CON-005` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric` | `covered` |
| `SCALE/CON-006` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.L1SoftUnscaledResidualScalesSharedRelaxation`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.L2SoftUnscaledResidualScalesDualRelaxation`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.MixedSoftUnscaledResidualScalesSharedRelaxation` | `covered` |

## Open Gaps

- No open P0 row-scaling unit gaps.
