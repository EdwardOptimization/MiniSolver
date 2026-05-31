# Hard Constraints Contract

Status: draft

Owner modules:

- `MOD-CORE-SEMANTICS`
- `MOD-ALG-INIT`
- `MOD-SOLVER-RICCATI`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-ALG-LS`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Structural hard row semantics for inequalities represented by slack variables.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `CON-001` | A hard row has no structural L1 or L2 soft flag. | `covered` |
| `CON-002` | Hard-row primal equation is `g + s = 0` in internal solver units. | `covered` |
| `CON-003` | Hard-row complementarity is `s*lam -> 0` for KKT quality and `s*lam = mu` on the barrier path. | `covered` |
| `CON-004` | Hard rows are checked first in hot branch order when practical. | `covered` |
| `CON-005` | Hard-row unscaled residual uses raw constraint residual plus scaled-back slack contribution. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | `g_val`, `g_true`, `g_unscaled`, `s`, `lam`, row scale |
| Outputs | Barrier derivatives, residuals, line-search boundary constraints |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Hard row infeasible after postsolve | `INFEASIBLE` or primal failure path. |
| Non-finite hard residual | `NUMERICAL_ERROR` at residual boundary. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `CON-001` | `tests/test_soft_constraints.cpp::InitializationTest.HardConstraintInitializesPositiveCentralPath`, source inspection of `detail::hard_constraint_row` call sites | `covered` |
| `CON-002` | `tests/test_soft_constraints.cpp::InitializationTest.HardConstraintInitializesPositiveCentralPath`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal` | `covered` |
| `CON-003` | `tests/test_soft_constraints.cpp::InitializationTest.HardConstraintInitializesPositiveCentralPath`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceRejectsLargeTrueComplementarityAtMuFinal`, `tests/test_soft_constraints.cpp::SoftConstraintTest.HardBarrierDerivativeUsesLamOverS` | `covered` |
| `CON-004` | Source inspection: hard rows are first in `compute_barrier_derivatives`, `recover_dual_search_directions`, `compute_max_violation`, `compute_unscaled_max_violation`, `MeritLineSearch`, and `FilterLineSearch` row branches. | `covered` |
| `CON-005` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingScalesConstraintPacketsOnly` | `covered` |

## Open Gaps

- No open P0 hard-row semantic coverage gaps. Any future claim that branch order
  improves runtime still needs benchmark evidence; this contract only records
  the current hard-first implementation shape.
