# Postsolve Contract

Status: draft

Owner modules:

- `MOD-SOLVER-ROUTE`
- `MOD-ALG-EVAL`
- `MOD-CORE-TYPES`

Related modules:

- `MOD-ALG-TERM`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Final residual refresh.
- Final status classification.
- Failure precedence after loop exit.
- `SolverInfo` projection from final residuals.

Out of scope:

- Main-loop exit decision logic.
- User-level safety acceptance outside solver units.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Active final trajectory | `MOD-CORE-TRAJ` | Last accepted iterate or current initial guess. |
| Loop exit status/reason | `MOD-SOLVER-ROUTE` | May be success-like or failure. |
| Solver config/tolerances | `MOD-CORE-CONFIG` | Validated. |
| Linear solver scratch | `MOD-SOLVER-RICCATI` | Used for fresh dual residual when needed. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Final `SolverStatus` | Public result. | User/tests |
| Final `SolverInfo` residuals | Fresh residual diagnostics. | User/tests |
| Postsolve failure reason | Explanation for postsolve downgrade. | User/tests |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `PostsolveResiduals` | `MOD-CORE-TYPES` | Postsolve call. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `POST-001` | Postsolve must refresh primal residuals at the final active trajectory. | `covered` |
| `POST-002` | Postsolve must refresh dual residuals before final strict KKT classification. | `covered` |
| `POST-003` | Postsolve must check strict `OPTIMAL` before primal-only `FEASIBLE`. | `covered` |
| `POST-004` | Success-like loop exits that fail final primal feasibility become postsolve infeasible. | `covered` |
| `POST-005` | Non-finite postsolve residuals produce numerical failure rather than success. | `covered` |
| `POST-006` | `SolverInfo` residual fields must reflect postsolve residuals. | `covered` |
| `POST-007` | Loop failure precedence must be preserved when postsolve cannot prove acceptable quality. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Final strict KKT proven | `OPTIMAL` / `CONVERGED` | Highest final quality. |
| Final primal-only acceptable | `FEASIBLE` / appropriate primal reason | Not strict KKT. |
| Final primal infeasible after success-like loop exit | `INFEASIBLE` / `POSTSOLVE_INFEASIBLE` | Stale success correction. |
| Final non-finite residual | `NUMERICAL_ERROR` / `NUMERICAL_ERROR` | Numerical boundary. |
| Prior fatal loop failure remains unproven | Preserve failure | Postsolve does not hide fatal loop failures. |

## Numeric And Performance Constraints

- Postsolve may do fresh residual work; it is outside the hottest inner kernel.
- Postsolve must not mutate model structure or public config.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `POST-001` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal` | `covered` |
| `POST-002` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual` | `covered` |
| `POST-003` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal` | `covered` |
| `POST-004` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal` | `covered` |
| `POST-005` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsInfConstraintResidual`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual` | `covered` |
| `POST-006` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal` | `covered` |
| `POST-007` | `tests/test_termination.cpp::TerminationTest.GenericInsufficientProgressReasonDoesNotPretendCostStagnation`, `tests/test_termination.cpp::TerminationTest.RtiFixedIterationDoesNotMaskLinearSolveFailure` | `covered` |

## Open Gaps

- No open P0 postsolve unit gaps.
