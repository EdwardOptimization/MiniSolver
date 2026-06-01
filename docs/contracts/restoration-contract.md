# Restoration Contract

Status: draft

Owner modules:

- `MOD-SOLVER-ROUTE`
- `MOD-ALG-LS`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-ALG-TERM`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Feasibility restoration entry, sufficient improvement, counters, accepted
trajectory handling, and failure precedence.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `REST-001` | Restoration is entered only from configured recovery paths, not as a default replacement for failed semantics. | `covered` |
| `REST-002` | Restoration must show configured sufficient primal improvement before success. | `covered` |
| `REST-003` | Restoration success updates diagnostics and returns to normal final classification. | `covered` |
| `REST-004` | Restoration exhaustion produces `RESTORATION_FAILED` and must not be hidden by budget exits. | `covered` |
| `REST-005` | Restoration counters in `SolverInfo` reflect attempts and successes. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Current infeasible iterate, restoration config, model packets |
| Outputs | Recovered trajectory or restoration failure, counters |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| No sufficient improvement | Keep searching until budget, then restoration failure. |
| Restoration budget exhausted | `RESTORATION_FAILED` / `RESTORATION_FAILED`. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `REST-001` | `tests/test_bugfixes.cpp::BugfixTest.TinyStepRecoveryFailureReturnsRestorationFailed`, `tests/test_line_search.cpp` | `covered` |
| `REST-002` | `tests/test_bugfixes.cpp::BugfixTest.FeasibilityRestorationRequiresViolationImprovement`, `tests/test_bugfixes.cpp::BugfixTest.FeasibilityRestorationRejectsTinyRelativeImprovement` | `covered` |
| `REST-003` | `tests/test_bugfixes.cpp::BugfixTest.FeasibilityRestorationSuccessUpdatesCounters` | `covered` |
| `REST-004` | `tests/test_bugfixes.cpp::BugfixTest.TinyStepRecoveryFailureReturnsRestorationFailed` | `covered` |
| `REST-005` | `tests/test_bugfixes.cpp::BugfixTest.FeasibilityRestorationRequiresViolationImprovement`, `tests/test_bugfixes.cpp::BugfixTest.FeasibilityRestorationRejectsTinyRelativeImprovement`, `tests/test_bugfixes.cpp::BugfixTest.FeasibilityRestorationSuccessUpdatesCounters` | `covered` |

## Open Gaps

- No open P1 evidence gaps. If restoration grows into a larger subsystem, move
  the focused tests out of `test_bugfixes.cpp` into a dedicated restoration
  contract test file.
