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
| `BACKEND-001` | `CPU_SERIAL` is the implemented default backend. | `covered` |
| `BACKEND-002` | `GPU_MPX` and `GPU_PCR` are reserved and must fail explicitly until implemented. | `covered` |
| `BACKEND-003` | `set_config()` preserves the constructor backend. | `covered` |
| `BACKEND-004` | CUDA build support is opt-in and incomplete GPU code must not compile silently as a supported backend. | `covered` |
| `BACKEND-005` | Backend benchmark comparisons must state the selected backend and matrix backend. | `deferred` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Unsupported GPU requested | Linear solve failure path with diagnostic. |
| CUDA requested but unavailable | CMake warning or disabled GPU backend according to build config. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `BACKEND-001` | default solver tests, `tests/test_features.cpp` | `covered` |
| `BACKEND-002` | `tests/test_features.cpp::FeaturesTest.GPUBackendUnsupportedFailsExplicitly`, `docs/contracts/solve-loop-contract.md::SOLVE-007` | `covered` |
| `BACKEND-003` | `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigPreservesBackendInvariant`, `tests/test_solver_snapshot.cpp` backend-policy tests | `covered` |
| `BACKEND-004` | CMake inspection: CUDA is behind `MINISOLVER_USE_CUDA=OFF` by default and incomplete CUDA code is opt-in only. | `covered` |
| `BACKEND-005` | Deferred to `nmpc-bench` backend-comparison reports when functional backend comparisons are added. | `deferred` |

## Open Gaps

- No open P1 evidence gaps. `BACKEND-005` remains deferred because there is no
  functional alternative backend comparison to report yet.
