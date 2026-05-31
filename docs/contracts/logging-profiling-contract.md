# Logging And Profiling Contract

Status: draft

Owner modules:

- `MOD-RUNTIME`
- `MOD-SOLVER-ROUTE`

Related modules:

- `MOD-TESTING`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Logger configuration, compile-time log levels, profiling timer behavior, and
allocation/performance boundary.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `LOG-001` | `MINISOLVER_LOG_LEVEL` controls whether log macros compile active stream code. | `partial` |
| `LOG-002` | Logger callback receives level, message, and user pointer when installed. | `partial` |
| `LOG-003` | Without callback, logs go to stdout/stderr according to level. | `partial` |
| `LOG-004` | Profiling is disabled by default and may allocate/use maps when enabled. | `partial` |
| `LOG-005` | Performance and zero-allocation claims must state whether logging/profiling is enabled. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Logger callback throws/fails | User-owned; MiniSolver does not recover. |
| Profiling enabled in performance benchmark | Benchmark must report this setting. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `LOG-001` | build/config inspection | `partial` |
| `LOG-002` | `tests/test_logger.cpp` | `partial` |
| `LOG-003` | `tests/test_logger.cpp` | `partial` |
| `LOG-004` | `tests/test_memory.cpp` | `partial` |
| `LOG-005` | benchmark docs/process | `partial` |

## Open Gaps

- Need explicit profiling-enabled allocation note in testing matrix.
