# Model Callback Contract

Status: draft

Owner modules:

- `MOD-SOLVER-ROUTE`
- `MOD-MODEL-CODEGEN`

Related modules:

- `MOD-ALG-TERM`
- `MOD-ALG-EVAL`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Model-update callback timing, allowed mutations, forbidden structural mutation,
and callback interaction with early termination.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `MODEL-CB-001` | Callback is invoked before presolve/model evaluation and before iteration model evaluation according to solve route. | `covered` |
| `MODEL-CB-002` | Callback may update model data such as references, parameters, and oracle values through supported APIs. | `covered` |
| `MODEL-CB-003` | Callback must not structurally mutate solver configuration, horizon, or callback wiring. | `covered` |
| `MODEL-CB-004` | Structural mutators invoked during callback return `ApiStatus::InvalidArgument` when the API has a status return, or mark the callback contract violation for void mutators. | `covered` |
| `MODEL-CB-005` | Callback presence disables warm-start zero-step acceptable NMPC shortcut. | `covered` |
| `MODEL-CB-006` | Callback errors propagate as solve failure instead of using stale model packets. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | `MiniSolver&`, user pointer, current iteration context |
| Outputs | Updated model data or `ApiStatus` failure |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Callback returns non-OK | Solver reports invalid/numerical failure according to callback path. |
| Callback structural mutation | Mutator returns `ApiStatus::InvalidArgument` when possible and solve reports `INVALID_INPUT`. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `MODEL-CB-001` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforeFirstEvaluation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforePresolveSlackInitialization` | `covered` |
| `MODEL-CB-002` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforeFirstEvaluation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforePresolveSlackInitialization` | `covered` |
| `MODEL-CB-003` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackSetConfigIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackResizeHorizonIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackSetCallbackIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackClearCallbackIsRejectedWithoutMutation` | `covered` |
| `MODEL-CB-004` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackSetConfigIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackResizeHorizonIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackResetIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackSetCallbackIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackClearCallbackIsRejectedWithoutMutation` | `covered` |
| `MODEL-CB-005` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcCallbackDoesNotSkipDirectionSolve` | `covered` |
| `MODEL-CB-006` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackFailureStopsSolveAsInvalidInput`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackFailureDuringIterationStopsBeforeEvaluation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackDirtyPlanDuringIterationIsRejected` | `covered` |

## Open Gaps

- Covered callback mutator inventory currently includes `set_config()`,
  `resize_horizon()`, `reset()`, `solve()`, `set_model_update_callback()`, and
  `clear_model_update_callback()`.
