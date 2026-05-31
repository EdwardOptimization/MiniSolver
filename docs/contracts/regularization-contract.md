# Regularization Contract

Status: draft

Owner modules:

- `MOD-SOLVER-ROUTE`
- `MOD-SOLVER-RICCATI`
- `MOD-ALG-INIT`

Related modules:

- `MOD-CORE-CONFIG`
- `MOD-CORE-TYPES`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Linear-solve retry count.
- Regularization escalation/downscale.
- Warm-start regularization reuse.
- Degraded step diagnostics.

Out of scope:

- New factorization modification strategies.
- Backend-specific linear solver APIs.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Current `reg` | Solver context | Clamped by config. |
| Linear solve result | Riccati solver | Structured success/failure. |
| `SolverConfig` | Config | Regularization ranges validated. |
| Previous `reg` | Solver context | May be reused/clamped. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Updated `reg` | Next linear-solve regularization. | Solver route/Riccati |
| Regularization escalation count | Diagnostic. | `SolverInfo` |
| Degraded step fields | Diagnostic. | `SolverInfo` |
| Linear failure status | Terminal failure if retries exhausted. | Solver route |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `SolveState::reg` | `MOD-CORE-TYPES` | Solver algorithmic state. |
| `DirectionState` degraded fields | `MOD-CORE-TYPES` | Per iteration. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `REG-001` | Linear solve may retry up to `linear_solve_max_attempts`. | `covered` |
| `REG-002` | Failed factorization/solve escalates regularization within configured bounds. | `covered` |
| `REG-003` | Successful solve may downscale/cool regularization according to config. | `covered` |
| `REG-004` | Exhausted retries produce `LINEAR_SOLVE_FAILED` and must not be masked by RTI/fixed-iteration exit. | `covered` |
| `REG-005` | Warm-start regularization reuse must clamp invalid previous values. | `covered` |
| `REG-006` | Degraded/frozen step diagnostics must be projected into `SolverInfo`. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Exhausted linear solve attempts | `LINEAR_SOLVE_FAILED` / `LINEAR_SOLVE_FAILED` | Fatal. |
| Invalid previous `reg` | Fallback/clamp | Setup boundary. |
| Degraded but accepted linear solve | Final status decided by later phases | Diagnostics must reflect degradation. |

## Numeric And Performance Constraints

- Retry policy should not allocate.
- Regularization state belongs to solver context, not hidden inside Riccati
  solver side channels.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `REG-001` | `tests/test_termination.cpp::TerminationTest.LinearSolveRetriesEscalateRegularizationWithinBounds` | `covered` |
| `REG-002` | `tests/test_termination.cpp::TerminationTest.LinearSolveRetriesEscalateRegularizationWithinBounds` | `covered` |
| `REG-003` | `tests/test_termination.cpp::TerminationTest.SuccessfulLinearSolveDecaysRegularizationWhenAlphaIsHealthy` | `covered` |
| `REG-004` | `tests/test_termination.cpp::TerminationTest.RtiFixedIterationDoesNotMaskLinearSolveFailure` | `covered` |
| `REG-005` | `tests/test_config_regressions.cpp::ConfigRegressionTest.DefaultWarmStartResetsBarrierAndRegularization`, `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartRegularizationModesAreExplicit` | `covered` |
| `REG-006` | `tests/test_status.cpp::StatusTest.SolverInfoReportsDegradedRiccatiStep`, `tests/test_riccati.cpp::RiccatiTest.NonSPDQuuFreezesControlDimsInsteadOfFailing` | `covered` |

## Open Gaps

- No open P0 regularization coverage gaps. Alternative factorization or backend
  retry policies need their own contract rows before implementation.
