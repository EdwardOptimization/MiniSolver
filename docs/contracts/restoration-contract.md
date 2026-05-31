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
| `REST-001` | Restoration is entered only from configured recovery paths, not as a default replacement for failed semantics. | `partial` |
| `REST-002` | Restoration must show configured sufficient primal improvement before success. | `partial` |
| `REST-003` | Restoration success updates diagnostics and returns to normal final classification. | `partial` |
| `REST-004` | Restoration exhaustion produces `RESTORATION_FAILED` and must not be hidden by budget exits. | `partial` |
| `REST-005` | Restoration counters in `SolverInfo` reflect attempts and successes. | `partial` |

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
| `REST-001` | `tests/test_robustness.cpp`, `tests/test_solver_quality.cpp` | `partial` |
| `REST-002` | `tests/test_robustness.cpp` | `partial` |
| `REST-003` | `tests/test_status.cpp` | `partial` |
| `REST-004` | `tests/test_termination.cpp`, `tests/test_robustness.cpp` | `partial` |
| `REST-005` | `tests/test_status.cpp` | `partial` |

## Open Gaps

- Need a focused restoration contract test file if restoration grows.
