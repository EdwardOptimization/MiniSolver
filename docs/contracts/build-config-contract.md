# Build Config Contract

Status: draft

Owner modules:

- `MOD-RUNTIME`
- `MOD-MATRIX`
- `MOD-TESTING`

Related modules:

- `MOD-MODEL-CODEGEN`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

CMake options, dependency fetching, test/example/tool toggles, matrix backend
selection, benchmark compiler flags, and generated asset builds.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `BUILD-001` | Build defaults to Release when `CMAKE_BUILD_TYPE` is unset. | `partial` |
| `BUILD-002` | Eigen is available when Eigen backend or tools require it. | `partial` |
| `BUILD-003` | GoogleTest is available when tests are enabled. | `partial` |
| `BUILD-004` | `MINISOLVER_ENABLE_FAST_MATH` and `MINISOLVER_ENABLE_NATIVE_ARCH` are opt-in benchmark flags. | `partial` |
| `BUILD-005` | `MINISOLVER_BUILD_TESTS`, `MINISOLVER_BUILD_EXAMPLES`, and `MINISOLVER_BUILD_TOOLS` control optional targets. | `partial` |
| `BUILD-006` | CUDA backend build is opt-in and experimental. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Missing dependency with fetch disabled | CMake configure failure. |
| CUDA requested but unavailable | Warning/disabled backend according to CMake logic. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `BUILD-001` | CMake inspection | `partial` |
| `BUILD-002` | CI/local build | `partial` |
| `BUILD-003` | CI/local build | `partial` |
| `BUILD-004` | benchmark process docs | `partial` |
| `BUILD-005` | CI/local build variants | `partial` |
| `BUILD-006` | CMake inspection | `partial` |

## Open Gaps

- Need CI matrix documentation for Eigen/custom matrix builds.
