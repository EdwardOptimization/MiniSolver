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
| `MAT-001` | Exactly one effective matrix backend is selected; custom MiniMatrix overrides Eigen if both are enabled. | `covered` |
| `MAT-002` | If no backend macro is selected, Eigen is the default. | `covered` |
| `MAT-003` | `MatOps` provides the operations required by solver, Riccati, line search, and integrator code. | `covered` |
| `MAT-004` | MiniMatrix and Eigen behavior must stay numerically compatible for covered operations. | `covered` |
| `MAT-005` | Finite checks needed under `-ffast-math` use bit-level helpers where standard checks are unsafe. | `covered` |
| `MAT-006` | Kernel unroll policy changes require tests and, for performance claims, benchmark evidence. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Matrix solve fails | Return false to caller. |
| Backend parity regression | Test failure. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `MAT-001` | CMake inspection, custom-backend full `ctest` evidence in `docs/testing/contract-rollout-completion-audit.md` | `covered` |
| `MAT-002` | CMake inspection, Eigen default full `ctest` evidence in `docs/testing/contract-rollout-completion-audit.md` | `covered` |
| `MAT-003` | `tests/test_matrix.cpp`, `tests/test_mini_matrix.cpp`, full Eigen/custom `ctest` | `covered` |
| `MAT-004` | `tests/test_mini_matrix.cpp`, full custom build, `docs/testing/contract-rollout-completion-audit.md` | `covered` |
| `MAT-005` | `tests/test_mini_matrix.cpp::MiniMatrixTest.Kernel_HasNanAndAllFiniteBoundaryCases`, `docs/adr/0003-minisolver-matrix-kernels.md` | `covered` |
| `MAT-006` | `docs/matrix/matrix-policy-guide.md`, `docs/matrix/matrix-kernel-benchmark-notes.md`, matrix kernel tests | `covered` |

## Open Gaps

- No open P1 evidence gaps. Future performance claims still require fresh
  benchmark evidence for the touched kernel/backend.
