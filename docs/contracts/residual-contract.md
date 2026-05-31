# Residual Contract

Status: draft

Owner modules:

- `MOD-ALG-EVAL`
- `MOD-SOLVER-ROUTE`
- `MOD-CORE-TYPES`

Related modules:

- `MOD-SOLVER-RICCATI`
- `MOD-ALG-TERM`
- `MOD-CORE-SEMANTICS`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Primal residual metrics.
- Unscaled primal residual diagnostics.
- Dual residual metrics.
- Complementarity and barrier centrality residuals.
- Average complementarity gap.

Out of scope:

- Exact KKT derivative formulas.
- External physical-unit acceptance policy.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Active trajectory packets | Model evaluation | Current residual/Jacobian/cost packets. |
| Slack/dual/soft slack state | Initialization/line search | Interior under valid IPM state. |
| `mu` | Barrier update/solver context | Positive finite under valid algorithm state. |
| Scaling fields | Model evaluation | Active row/objective scales. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| `primal_inf` | Internal scaled primal feasibility metric. | Termination/postsolve |
| `unscaled_primal_inf` | Raw model-unit diagnostic. | User/benchmark |
| `dual_inf` | Stationarity residual. | Termination/postsolve |
| `complementarity_inf` | True KKT complementarity quality. | Termination/postsolve |
| `barrier_centrality_inf` | Barrier centrality diagnostic. | Logs/tests |
| `avg_complementarity_gap` | Barrier update input. | Barrier update |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `ResidualState` | `MOD-CORE-TYPES` | Reset per iteration. |
| `StepResidualSummary` | `MOD-CORE-TYPES` | Per residual evaluation. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `NUM-020` | `primal_inf` is the internal residual used by termination and final feasible classification. | `covered` |
| `NUM-021` | `unscaled_primal_inf` is diagnostic and must preserve raw model-unit interpretation under row scaling. | `covered` |
| `NUM-022` | `dual_inf` must be refreshed before strict final convergence classification. | `covered` |
| `NUM-023` | `complementarity_inf` must measure true complementarity quality, not `mu` alone. | `covered` |
| `NUM-024` | `barrier_centrality_inf` may diagnose central path quality but is not alone a strict KKT certificate. | `covered` |
| `NUM-025` | Average complementarity gap must not be reduced by invalid negative L1 soft-dual gaps. | `covered` |
| `NUM-026` | Residual reductions must preserve non-finite evidence for boundary classification. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Non-finite residual metric | `NUMERICAL_ERROR` / `NUMERICAL_ERROR` | Boundary classification. |
| Internal feasible but unscaled large | Solver may return feasible in internal units | External safety policy is out of scope. |
| Dual residual refresh fails | Preserve loop failure or numerical/linear failure | Postsolve owns final classification. |

## Numeric And Performance Constraints

- Residual calculations must be allocation-free.
- Scaled and unscaled metrics must not be mixed in termination unless a future
  contract explicitly changes that policy.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `NUM-020` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcAcceptedStepRefreshesPrimalResidual`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal` | `covered` |
| `NUM-021` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingScalesConstraintPacketsOnly` | `covered` |
| `NUM-022` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual` | `covered` |
| `NUM-023` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesTrueComplementarityGapSnapshot`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesKktComplementarityNotBarrierTarget`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceRejectsLargeTrueComplementarityAtMuFinal` | `covered` |
| `NUM-024` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.StepResidualSummaryRecordsBarrierMuSnapshot`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveResidualsRecordBarrierMuSnapshot`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceRejectsLargeTrueComplementarityAtMuFinal` | `covered` |
| `NUM-025` | `tests/test_soft_constraints.cpp::SoftConstraintTest.L1NegativeSoftDualDoesNotReduceAverageComplementarity` | `covered` |
| `NUM-026` | `tests/test_bugfixes.cpp::BugfixTest.MaxViolationPropagatesNaNConstraintResidual`, `tests/test_bugfixes.cpp::BugfixTest.UnscaledMaxViolationPropagatesNaNConstraintResidual`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsInfConstraintResidual`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual` | `covered` |

## Open Gaps

- No open P0 residual-metric coverage gaps. External physical-unit acceptance
  remains a separate application policy; `unscaled_primal_inf` is diagnostic.
