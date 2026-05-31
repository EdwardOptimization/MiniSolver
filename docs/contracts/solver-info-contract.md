# SolverInfo Contract

Status: draft

Owner modules:

- `MOD-CORE-TYPES`
- `MOD-SOLVER-ROUTE`

Related modules:

- `MOD-ALG-TERM`
- `MOD-ALG-EVAL`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Public `SolverInfo` fields, reset/projection, final residuals, diagnostics, and
counter semantics.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `DIAG-001` | `SolverInfo` is reset for each solve before new diagnostics are projected. | `covered` |
| `DIAG-002` | `status`, `loop_status`, and `termination_reason` preserve final-vs-loop distinction. | `covered` |
| `DIAG-003` | Residual fields report final postsolve residuals. | `covered` |
| `DIAG-004` | Scaling active flags report the active config/plan state. | `covered` |
| `DIAG-005` | SOC, restoration, regularization, and degraded-step counters reflect actual phase attempts/outcomes. | `covered` |
| `DIAG-006` | `alpha` reports the accepted step alpha for diagnostics. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Phase failure before some diagnostics are available | Fields remain reset/default or last meaningful value by documented owner. |
| Non-finite final residual | Status/reason must be numerical failure; residual fields preserve evidence. |

## Field Owners

| Field(s) | Owner phase | Evidence |
| --- | --- | --- |
| `status`, `loop_status`, `termination_reason` | Postsolve status projection, with loop status preserved from solve loop | `tests/test_termination.cpp`, `tests/test_barrier_residual_contract.cpp` |
| `iterations` | Solve loop counter projection | `tests/test_status.cpp` |
| `primal_inf`, `unscaled_primal_inf`, `dual_inf`, `complementarity_inf`, `barrier_centrality_inf`, `mu`, `linear_ok` | Postsolve residual projection for final public info | `tests/test_barrier_residual_contract.cpp`, `tests/test_scaling_regressions.cpp` |
| `alpha` | Last accepted globalization result, reset to `1.0` per solve | `tests/test_status.cpp`, `tests/test_line_search.cpp` |
| `line_search_failed` | Globalization failure classification | `tests/test_line_search.cpp` |
| `restoration_used`, `restoration_attempt_count`, `restoration_success_count` | Restoration route | `tests/test_status.cpp`, `tests/test_robustness.cpp` |
| `degraded_step`, `degraded_riccati_freeze_count` | Linear solve result projection | `tests/test_status.cpp::StatusTest.SolverInfoReportsDegradedRiccatiStep` |
| `regularization_escalation_count` | Linear-solve retry/escalation route | `tests/test_robustness.cpp`, `tests/test_status.cpp` |
| `soc_attempt_count`, `soc_accept_count`, `soc_reject_count` | Line-search SOC result projection | `tests/test_line_search.cpp`, `tests/test_status.cpp` |
| `constraint_scaling_active`, `objective_scaling_active`, `problem_scaling_active` | Build plan projection | `tests/test_scaling_regressions.cpp` |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `DIAG-001` | `tests/test_status.cpp::StatusTest.SolverInfoResetsBeforeSolveEntryCallback` | `covered` |
| `DIAG-002` | `tests/test_termination.cpp::TerminationTest.ResidualStagnationLoopStatusWaitsForPostsolveFeasibleQuality`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual` | `covered` |
| `DIAG-003` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal` | `covered` |
| `DIAG-004` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingNormalizesInternalPrimalMetric`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.ProblemScalingActivatesBoundedConstraintAndObjectiveScaling` | `covered` |
| `DIAG-005` | `tests/test_status.cpp::StatusTest.SolverInfoReportsDegradedRiccatiStep`, `tests/test_termination.cpp::TerminationTest.LinearSolveRetriesEscalateRegularizationWithinBounds`, `tests/test_bugfixes.cpp::BugfixTest.TinyStepRecoveryFailureReturnsRestorationFailed`, `tests/test_bugfixes.cpp::BugfixTest.SolverInfoRecordsSocLineSearchDiagnostics` | `covered` |
| `DIAG-006` | `tests/test_status.cpp::StatusTest.RtiFixedIterationProfileStopsAfterOneIteration`, `tests/test_line_search.cpp::LineSearchTest.FilterAcceptance`, `tests/test_line_search.cpp::LineSearchTest.NoLineSearchAppliesFractionToBoundary` | `covered` |

## Open Gaps

- No open P0 `SolverInfo` field coverage gaps. Future diagnostic fields should
  add an owner row and a reset/projection test in the same change.
