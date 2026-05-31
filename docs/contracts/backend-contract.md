# Backend Contract

Status: draft

Owner modules:

- `MOD-RUNTIME`
- `MOD-SOLVER-RICCATI`
- `MOD-CORE-CONFIG`

Related modules:

- `MOD-MATRIX`
- `MOD-SOLVER-ROUTE`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Backend enum, CPU serial implementation, reserved GPU backends, and backend
preservation across config mutation.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `BACKEND-001` | `CPU_SERIAL` is the implemented default backend. | `partial` |
| `BACKEND-002` | `GPU_MPX` and `GPU_PCR` are reserved and must fail explicitly until implemented. | `partial` |
| `BACKEND-003` | `set_config()` preserves the constructor backend. | `partial` |
| `BACKEND-004` | CUDA build support is opt-in and incomplete GPU code must not compile silently as a supported backend. | `partial` |
| `BACKEND-005` | Backend benchmark comparisons must state the selected backend and matrix backend. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Unsupported GPU requested | Linear solve failure path with diagnostic. |
| CUDA requested but unavailable | CMake warning or disabled GPU backend according to build config. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `BACKEND-001` | default solver tests | `partial` |
| `BACKEND-002` | backend/config tests | `partial` |
| `BACKEND-003` | `tests/test_config_regressions.cpp` | `partial` |
| `BACKEND-004` | CMake/build inspection | `partial` |
| `BACKEND-005` | benchmark process docs | `partial` |

## Open Gaps

- Need direct unsupported-backend regression if not already present.
