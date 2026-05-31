# Matrix Backend Contract

Status: draft

Owner modules:

- `MOD-MATRIX`
- `MOD-TESTING`

Related modules:

- `MOD-SOLVER-RICCATI`
- `MOD-INTEGRATOR`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Eigen/MiniMatrix selection, `MatOps` parity, finite checks under fast-math, and
kernel performance-sensitive policy.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `MAT-001` | Exactly one effective matrix backend is selected; custom MiniMatrix overrides Eigen if both are enabled. | `partial` |
| `MAT-002` | If no backend macro is selected, Eigen is the default. | `partial` |
| `MAT-003` | `MatOps` provides the operations required by solver, Riccati, line search, and integrator code. | `partial` |
| `MAT-004` | MiniMatrix and Eigen behavior must stay numerically compatible for covered operations. | `partial` |
| `MAT-005` | Finite checks needed under `-ffast-math` use bit-level helpers where standard checks are unsafe. | `partial` |
| `MAT-006` | Kernel unroll policy changes require tests and, for performance claims, benchmark evidence. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Matrix solve fails | Return false to caller. |
| Backend parity regression | Test failure. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `MAT-001` | CMake/build tests or inspection | `partial` |
| `MAT-002` | CMake/build tests or inspection | `partial` |
| `MAT-003` | `tests/test_matrix.cpp` | `partial` |
| `MAT-004` | `tests/test_mini_matrix.cpp`, full custom build | `partial` |
| `MAT-005` | `tests/test_matrix.cpp`, numeric tests | `partial` |
| `MAT-006` | matrix benchmark notes | `partial` |

## Open Gaps

- Need matrix backend coverage rows for full custom `ctest`.
