# Tolerance Contract

Status: draft

Owner modules:

- `MOD-CORE-CONFIG`
- `MOD-ALG-TERM`
- `MOD-SOLVER-ROUTE`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-ALG-BARRIER`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Meaning of core tolerances.
- Which residuals use internal scaled units.
- Which tolerances participate in strict convergence vs feasible fallback.

Out of scope:

- Automatic tolerance tuning.
- Physical-unit safety limits.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| `tol_con` | Config | Positive finite. |
| `tol_dual` | Config | Positive finite. |
| `tol_mu` | Config | Positive finite. |
| `tol_cost` | Config | Non-negative finite. |
| `feasible_tol_scale` | Config | Positive finite. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Strict convergence predicate | Uses `tol_con`, `tol_dual`, `tol_mu`. | Termination/postsolve |
| Feasible fallback predicate | Uses primal residual and feasible bound. | Termination/postsolve |
| Cost stagnation predicate | Uses `tol_cost` and primal feasibility context. | Termination |

## Owned State

No owned runtime state; tolerances live in `SolverConfig`.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `NUM-050` | `tol_con` applies to internal primal residual units. | `covered` |
| `NUM-051` | `tol_dual` applies to internal stationarity residual units. | `covered` |
| `NUM-052` | `tol_mu` applies to true complementarity residual units. | `covered` |
| `NUM-053` | Strict `OPTIMAL` requires all strict tolerances and `linear_ok`. | `covered` |
| `NUM-054` | `feasible_tol_scale` may be used for generic feasible fallback but not for acceptable NMPC primal shortcut when that profile requires `tol_con`. | `covered` |
| `NUM-055` | `tol_cost` only supports stagnation checks when primal residual is feasible enough. | `covered` |
| `NUM-056` | Scaled termination tolerances do not by themselves certify raw physical-unit feasibility. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Tolerance invalid in config | `ApiStatus::InvalidArgument` or `NonFiniteValue` | Config boundary. |
| Residual exceeds strict tolerance but primal acceptable | `FEASIBLE` if feasible fallback applies | Not `OPTIMAL`. |
| Residual exceeds primal tolerance/fallback bound | `UNSOLVED`, `MAX_ITER`, or postsolve failure | Depends on loop context. |

## Numeric And Performance Constraints

- Tolerance checks must remain scalar comparisons.
- Any future unscaled/physical-unit tolerance must be an explicit new contract,
  not a hidden reinterpretation of `tol_con`.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `NUM-050` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcPrimalFeasibleSkipsDirectionFailure`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric` | `covered` |
| `NUM-051` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal` | `covered` |
| `NUM-052` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesTrueComplementarityGapSnapshot`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceRejectsLargeTrueComplementarityAtMuFinal` | `covered` |
| `NUM-053` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesTrueComplementarityGapSnapshot`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesKktComplementarityNotBarrierTarget` | `covered` |
| `NUM-054` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcPrimalFeasibleSkipsDirectionFailure`, `tests/test_termination.cpp::TerminationTest.AcceptableNmpcCallbackDoesNotSkipDirectionSolve` | `covered` |
| `NUM-055` | `tests/test_bugfixes.cpp::BugfixTest.CostStagnationNotGatedOnMuFinal`, `tests/test_termination.cpp::TerminationTest.GenericInsufficientProgressReasonDoesNotPretendCostStagnation` | `covered` |
| `NUM-056` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric`, docs | `covered` |

## Open Gaps

- Tolerance contract IDs are covered by current unit/regression tests.
