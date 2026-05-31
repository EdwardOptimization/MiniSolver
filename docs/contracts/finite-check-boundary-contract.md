# Finite Check Boundary Contract

Status: draft

Owner modules:

- `MOD-ALG-EVAL`
- `MOD-SOLVER-ROUTE`
- `MOD-MATRIX`

Related modules:

- `MOD-CORE-CONFIG`
- `MOD-ALG-LS`
- `MOD-SOLVER-RICCATI`
- `MOD-DEBUG-SNAPSHOT`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Where MiniSolver checks for NaN/Inf.
- Where MiniSolver intentionally trusts model/codegen inputs.
- Status mapping for non-finite solver boundary values.
- Preservation of non-finite evidence in reductions.

Out of scope:

- Full validation of arbitrary model functions.
- Per-scalar defensive checks inside every hot predicate.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| User config values | `MOD-CORE-CONFIG` | Must be finite where required. |
| Model packets | `MOD-MODEL-CODEGEN` or handwritten model | Trusted until solver semantic boundaries. |
| Residual reductions | `MOD-ALG-EVAL` and solver route | Must not hide NaN/Inf. |
| Line-search scalars | `MOD-ALG-LS` | Must be finite before acceptance logic can trust them. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| `ApiStatus::NonFiniteValue` | Public config/input rejection. | User/snapshot |
| `SolverStatus::NUMERICAL_ERROR` | Solve-time non-finite arithmetic boundary. | User/tests |
| Preserved NaN/Inf diagnostics | Evidence for failure classification. | Postsolve/tests |

## Owned State

No owned state.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `NUM-001` | Public config validation rejects required-finite config values. | `covered` |
| `NUM-002` | Solver should check non-finite values at semantic boundaries, not every hot helper. | `covered` |
| `NUM-003` | Residual max reductions must preserve NaN/Inf instead of silently returning a finite neutral value. | `covered` |
| `NUM-004` | Scaling reductions must preserve NaN and classify Inf according to scaling contract. | `covered` |
| `NUM-005` | Line-search merit/filter scalar non-finites must surface as numerical failure. | `covered` |
| `NUM-006` | Postsolve non-finite residuals must prevent success status. | `covered` |
| `NUM-007` | Snapshot load rejects non-finite serialized solver data. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Non-finite config | `ApiStatus::NonFiniteValue` | Public/config boundary. |
| Non-finite residual | `NUMERICAL_ERROR` / `NUMERICAL_ERROR` | Solve/postsolve boundary. |
| Non-finite merit value or derivative | `NUMERICAL_ERROR` / `NUMERICAL_ERROR` | Line-search boundary. |
| Non-finite snapshot data | `SnapshotStatus::NonFiniteData` | Debug/replay boundary. |

## Numeric And Performance Constraints

- Checks belong at boundaries with diagnostic meaning.
- Avoid adding hot-path checks for states that valid invariants already exclude.
- If a reduction can swallow NaN due to `std::max` ordering, use explicit
  non-finite preserving logic.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `NUM-001` | `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigRejectsInvalidConfigWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigRejectsInvalidNumericalControlParameters` | `covered` |
| `NUM-002` | `docs/architecture/solver-development-principles.md`, `tests/test_line_search.cpp`, `tests/test_barrier_residual_contract.cpp`, `tests/test_solver_snapshot.cpp` | `covered` |
| `NUM-003` | `tests/test_bugfixes.cpp::BugfixTest.MaxViolationPropagatesNaNConstraintResidual`, `tests/test_bugfixes.cpp::BugfixTest.UnscaledMaxViolationPropagatesNaNConstraintResidual`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsInfConstraintResidual` | `covered` |
| `NUM-004` | `tests/test_scaling_regressions.cpp::ScalingRegressionTest.HessianGershgorinPropagatesNaNObjectiveScale`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.AutomaticRowScalingPropagatesNaNRowScale`, `tests/test_scaling_regressions.cpp::ScalingRegressionTest.HessianGershgorinOverflowUsesMinimumObjectiveScale` | `covered` |
| `NUM-005` | `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteDphiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteInitialPhiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteTrialPhiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteDphiPropagatesToSolverStatus` | `covered` |
| `NUM-006` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsInfConstraintResidual`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRejectsNonFiniteDualResidual` | `covered` |
| `NUM-007` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsNonFiniteTrajectoryDataAtomically` | `covered` |

## Open Gaps

- No open P0 finite-boundary coverage gaps. New non-finite checks should still
  be justified at semantic boundaries rather than added to hot predicates by
  default.
