# Status Semantics Contract

Status: draft

Owner modules:

- `MOD-CORE-TYPES`
- `MOD-SOLVER-ROUTE`
- `MOD-ALG-TERM`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Meaning of `SolverStatus`.
- Meaning of `TerminationReason`.
- Relationship between final `status`, `loop_status`, and postsolve residuals.
- `SolverInfo` diagnostic projection.

Out of scope:

- Exact tolerances for convergence.
- Public API naming changes.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Loop exit status/reason | `MOD-SOLVER-ROUTE` | Produced by current solve loop. |
| Postsolve residuals | `MOD-SOLVER-ROUTE` | Freshly recomputed at final iterate. |
| `TerminationSnapshot` | `MOD-ALG-TERM` | Contains current residual metrics. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| `SolverInfo::status` | Final public solve quality. | User/tests |
| `SolverInfo::loop_status` | Why the main loop stopped before postsolve. | Diagnostics/tests |
| `SolverInfo::termination_reason` | Stable reason for status/log analysis. | Diagnostics/tests |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `SolverInfo` | `MOD-CORE-TYPES` | Reset per solve, final value persists after solve. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `STATUS-001` | `OPTIMAL` means strict KKT quality at the final iterate. | `covered` |
| `STATUS-002` | `FEASIBLE` means primal-acceptable final iterate without strict KKT proof. | `covered` |
| `STATUS-003` | Failure statuses must not be overwritten by RTI/fixed-iteration budget exits. | `covered` |
| `STATUS-004` | `loop_status` records loop-level exit even when postsolve changes final status. | `covered` |
| `STATUS-005` | `termination_reason` must distinguish convergence, primal-feasible shortcut, fixed iteration, stagnation, numerical error, and postsolve infeasibility. | `covered` |
| `STATUS-006` | Status and reason string conversion must be total over known enum values and return `"UNKNOWN"` otherwise. | `covered` |
| `STATUS-007` | `SolverInfo` residual fields report the final residual snapshot, not stale loop metrics. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Fresh postsolve primal infeasible after success-like loop exit | `INFEASIBLE` / `POSTSOLVE_INFEASIBLE` | Protects public status from stale success. |
| Nonfinite final residual | `NUMERICAL_ERROR` / `NUMERICAL_ERROR` | Numeric boundary. |
| Linear solve failure | `LINEAR_SOLVE_FAILED` / `LINEAR_SOLVE_FAILED` | Fatal precedence over budget exits. |
| Line-search failure | `STEP_TOO_SMALL` or `NUMERICAL_ERROR` with matching reason | Depends on failure class. |

## Numeric And Performance Constraints

- Status projection must not trigger model re-evaluation except through the
  explicit postsolve path.
- Public diagnostics must remain fixed-size.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `STATUS-001` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveProvesStrictOptimalWhenFreshKktResidualsPass`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal` | `covered` |
| `STATUS-002` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcPrimalFeasibleSkipsDirectionFailure`, `tests/test_termination.cpp::TerminationTest.ResidualStagnationLoopStatusWaitsForPostsolveFeasibleQuality` | `covered` |
| `STATUS-003` | `tests/test_termination.cpp::TerminationTest.RtiFixedIterationDoesNotMaskLinearSolveFailure` | `covered` |
| `STATUS-004` | `tests/test_termination.cpp::TerminationTest.ResidualStagnationLoopStatusWaitsForPostsolveFeasibleQuality`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual` | `covered` |
| `STATUS-005` | `tests/test_status.cpp::StatusTest.SolverStatusAndTerminationReasonStringsCoverKnownEnums`, `tests/test_termination.cpp::TerminationTest.AcceptableNmpcPrimalFeasibleSkipsDirectionFailure`, `tests/test_status.cpp::StatusTest.RtiFixedIterationProfileStopsAfterOneIteration`, `tests/test_termination.cpp::TerminationTest.ResidualStagnationLoopStatusWaitsForPostsolveFeasibleQuality`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveProvesStrictOptimalWhenFreshKktResidualsPass` | `covered` |
| `STATUS-006` | `tests/test_status.cpp::StatusTest.SolverStatusAndTerminationReasonStringsCoverKnownEnums` | `covered` |
| `STATUS-007` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.BadlyScaledBaselineExposesAvailableSolveMetrics` | `covered` |

## Open Gaps

- No open P0 status unit gaps.
