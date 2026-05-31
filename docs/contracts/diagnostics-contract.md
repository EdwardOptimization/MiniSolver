# Diagnostics Contract

Status: draft

Owner modules:

- `MOD-SOLVER-ROUTE`
- `MOD-CORE-TYPES`
- `MOD-RUNTIME`

Related modules:

- `MOD-ALG-LS`
- `MOD-SOLVER-RICCATI`
- `MOD-DEBUG-SNAPSHOT`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Internal diagnostic counters and traces that are not core status semantics:
alpha log, SOC/restoration counters, regularization escalation, degraded
Riccati diagnostics, and logging/profiling hooks.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `DIAG-020` | Alpha trace storage must be pre-reserved for zero-allocation solve paths. | `covered` |
| `DIAG-021` | Regularization escalation count increments only on actual escalation attempts. | `partial` |
| `DIAG-022` | Degraded Riccati freeze count reports solver-provided degraded-step data. | `partial` |
| `DIAG-023` | SOC counters distinguish attempt, accept, and reject. | `partial` |
| `DIAG-024` | Restoration counters distinguish attempt and success. | `partial` |
| `DIAG-025` | Diagnostic logging must not alter solver status semantics. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Diagnostic unavailable due to early failure | Keep reset/default value unless owner phase ran. |
| Diagnostic allocation in claimed no-allocation path | Memory test failure. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `DIAG-020` | `tests/test_memory.cpp::MemoryTest.DefaultConfigSolveDoesNotAllocate`, `tests/test_memory.cpp::MemoryTest.ZeroMalloc_SolveAfterSetConfigDoesNotAllocate` | `covered` |
| `DIAG-021` | `tests/test_robustness.cpp`, `tests/test_status.cpp` | `partial` |
| `DIAG-022` | `tests/test_robustness.cpp`, `tests/test_status.cpp` | `partial` |
| `DIAG-023` | `tests/test_status.cpp`, `tests/test_line_search.cpp` | `partial` |
| `DIAG-024` | `tests/test_status.cpp`, restoration tests | `partial` |
| `DIAG-025` | `tests/test_logger.cpp` | `partial` |

## Open Gaps

- Need field-owner table for all diagnostic fields.
