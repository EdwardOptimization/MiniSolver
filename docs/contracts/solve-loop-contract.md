# Solve Loop Contract

Status: draft

Owner modules:

- `MOD-SOLVER-ROUTE`
- `MOD-CORE-TYPES`
- `MOD-CORE-TRAJ`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-SOLVER-RICCATI`
- `MOD-ALG-LS`
- `MOD-ALG-TERM`
- `MOD-DEBUG-SNAPSHOT`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Public `MiniSolver::solve()` phase ordering.
- Component rebuild boundaries.
- Callback timing and structural mutation guard.
- Active/candidate trajectory ownership during a solve.
- Loop status and final postsolve handoff.

Out of scope:

- The exact Riccati formulas.
- The exact line-search acceptance formulas.
- Cross-solver benchmark policy.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| `SolverConfig` | `MOD-CORE-CONFIG` | Validated at construction or `set_config()`. |
| Active trajectory | `MOD-CORE-TRAJ` | Contains current guess and model data. |
| Model update callback | `MOD-SOLVER-ROUTE` | May mutate model data, not solver structure. |
| Solver components | `MOD-SOLVER-ROUTE` | Rebuilt when build state is dirty. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Final `SolverStatus` | Postsolve-classified public result. | User/tests |
| `SolverInfo` | Final diagnostics plus loop status/reason. | User/tests |
| Active trajectory | Accepted final iterate. | User/next solve |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `Trajectory` | `MOD-CORE-TRAJ` | Solver lifetime. |
| `SolverContext` | `MOD-CORE-TYPES` | Solver lifetime; reset per solve where needed. |
| `SolverBuildState` | `MOD-CORE-TYPES` | Solver lifetime. |
| Callback pointer/user data | `MOD-SOLVER-ROUTE` | Solver lifetime until changed. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `SOLVE-001` | `solve()` must reset solve-local context before classifying the new solve. | `covered` |
| `SOLVE-002` | Dirty solver components must be rebuilt before a solve uses config-dependent strategies. | `covered` |
| `SOLVE-003` | Callback is invoked only at defined model-update points and must not structurally mutate the solver. | `covered` |
| `SOLVE-004` | The canonical loop must run through evaluation, residual/termination check, direction solve, globalization, and loop-exit classification unless an earlier phase returns a terminal failure/success. | `covered` |
| `SOLVE-005` | Candidate trajectory becomes active only through an accepted swap. | `covered` |
| `SOLVE-006` | Public final status must be produced by postsolve, not by stale loop residuals. | `covered` |
| `SOLVE-007` | Unsupported backend selection must fail explicitly instead of silently changing backend. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Invalid config at boundary | `ApiStatus`, or construction exception | Config is rejected before solve execution. |
| Callback structural mutation | `ApiStatus::InvalidArgument` or callback contract violation | Solve route guards structural APIs. |
| Linear solve failure | `LINEAR_SOLVE_FAILED` / `LINEAR_SOLVE_FAILED` | Retry policy belongs to solve route and regularization contract. |
| Line-search numerical failure | `NUMERICAL_ERROR` / `NUMERICAL_ERROR` | Scalar non-finite boundary. |
| Loop success-like exit | Final status decided by postsolve | Prevents stale success. |

## Numeric And Performance Constraints

- The default solve path must preserve solve-time zero allocation.
- Strategy decisions should be resolved at construction, `set_config()`, or
  rebuild boundaries, not repeatedly as public polymorphic checks in hot loops.
- Side diagnostics should travel through fixed-size result/state structs.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `SOLVE-001` | `tests/test_status.cpp::StatusTest.SolverInfoResetsBeforeSolveEntryCallback`, `tests/test_solver.cpp`, `tests/test_status.cpp` | `covered` |
| `SOLVE-002` | `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigDefersPlanRebuildUntilSolve` | `covered` |
| `SOLVE-003` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforeFirstEvaluation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforePresolveSlackInitialization`, callback structural mutation tests | `covered` |
| `SOLVE-004` | `tests/test_termination.cpp`, `tests/test_line_search.cpp`, `tests/test_barrier_residual_contract.cpp` | `covered` |
| `SOLVE-005` | `tests/test_line_search.cpp::LineSearchTest.FilterAcceptance`, `tests/test_line_search.cpp::LineSearchTest.MeritArmijoDoesNotBuildFiniteDifferenceProbe`, `tests/test_line_search.cpp::LineSearchTest.NoLineSearchAppliesFractionToBoundary` | `covered` |
| `SOLVE-006` | `tests/test_status.cpp`, `tests/test_termination.cpp`, `tests/test_barrier_residual_contract.cpp` | `covered` |
| `SOLVE-007` | `tests/test_features.cpp::FeaturesTest.GPUBackendUnsupportedFailsExplicitly`, `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigPreservesConstructorBackend` | `covered` |

## Open Gaps

- No open P0 solve-loop coverage gaps. Benchmark-level route timing and
  profile tuning remain outside this contract.
