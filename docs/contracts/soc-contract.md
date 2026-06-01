# SOC Contract

Status: draft

Owner modules:

- `MOD-ALG-LS`
- `MOD-ALG-EVAL`
- `MOD-SOLVER-RICCATI`

Related modules:

- `MOD-CORE-SEMANTICS`
- `MOD-SOLVER-ROUTE`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Second-order correction trigger, trial constraint packet refresh, row-scale and
soft-weight consistency, accept/reject semantics, and diagnostics.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `SOC-001` | SOC is attempted only when enabled and the accepted/trial alpha meets the configured trigger. | `covered` |
| `SOC-002` | SOC candidate constraints are evaluated at the SOC trial point. | `covered` |
| `SOC-003` | SOC trial soft weights are refreshed for the trial knot before soft semantics use them. | `covered` |
| `SOC-004` | SOC row scaling uses the active scale policy consistently with normal candidates. | `covered` |
| `SOC-005` | SOC accept/reject counters in `SolverInfo` reflect attempts and outcomes. | `covered` |
| `SOC-006` | Rejected SOC candidate must not corrupt the active trajectory. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Trial trajectory, active linearization, linear solver, `mu`, `reg`, config |
| Outputs | Accepted SOC candidate or rejection, SOC counters |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| SOC linear solve failure | Reject SOC or propagate failure according to caller path. |
| SOC candidate worse/not acceptable | Reject and keep original line-search candidate. |
| Non-finite SOC packet | Numerical failure if it reaches boundary checks. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `SOC-001` | `tests/test_line_search.cpp::LineSearchTest.FilterSocSkippedInRolloutMode`, `tests/test_line_search.cpp::LineSearchTest.FilterSocDampsCorrectionToStayInterior` | `covered` |
| `SOC-002` | `tests/test_line_search.cpp::LineSearchTest.FilterSocUsesCandidateSlackAsCorrectionBase`, `tests/test_line_search.cpp::LineSearchTest.FilterSocUsesModelSocConstraintOverride`, `tests/test_replay_corpus.cpp::ReplayCorpusTest.SocNonlinearObstaclePathAttemptsAndAcceptsCorrection` | `covered` |
| `SOC-003` | `tests/minimodel/test_constraints.py::test_generated_soc_refreshes_parameterized_soft_weights` | `covered` |
| `SOC-004` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.SocConstraintRowScalingReusesCandidateScale`, `tests/minimodel/test_constraints.py::test_generated_soc_refreshes_parameterized_soft_weights` | `covered` |
| `SOC-005` | `tests/test_bugfixes.cpp::BugfixTest.SolverInfoRecordsSocLineSearchDiagnostics`, `tests/test_status.cpp::StatusTest.SolverInfoResetsBeforeSolveEntryCallback` | `covered` |
| `SOC-006` | `tests/test_line_search.cpp::LineSearchTest.FilterRejectsTrialAboveThetaMax` | `covered` |

## Open Gaps

- No open P1 evidence gaps.
