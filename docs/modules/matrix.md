# Matrix Backend

Module ID: `MOD-MATRIX`

Status: draft

Files:

- `include/minisolver/matrix/matrix_defs.h`
- `include/minisolver/matrix/mini_matrix.h`
- `include/minisolver/matrix/kernels.h`
- `include/minisolver/matrix/policies.h`
- `include/minisolver/matrix/static_for.h`

Owner layer:

- Backend-independent fixed-size matrix type aliases and kernels.

## Purpose

Own Eigen/custom MiniMatrix selection, matrix/vector type aliases, shared
`MatOps`, fixed-size kernel wrappers, and policies for static unrolling and
finite checks under benchmark compiler flags.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Compile definitions | CMake | Select Eigen or custom matrix backend. |
| Fixed-size matrices/vectors | Solver/model/integrator | Dimensions known at compile time. |
| Kernel policy macros | CMake cache | Affect custom backend codegen/unrolling. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| `MSMat`, `MSVec`, `MSDiag` | All solver modules | Backend-neutral types. |
| `MatOps` functions | Solver kernels | Backend-neutral operations. |
| MiniMatrix kernels | Custom backend builds | Fixed-size math implementation. |

## Owned State

No persistent runtime state.

## Public API Surface

- Matrix aliases and `MatOps`
- MiniMatrix backend types under `USE_CUSTOM_MATRIX`

## Internal Contracts

- Custom matrix backend overrides Eigen if both macros are enabled.
- If neither backend is enabled, Eigen is selected by default.
- Finite checks that must survive `-ffast-math` use bit-level helpers.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: kernel policy changes are performance-sensitive and require benchmark
  or parity evidence.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Failed Cholesky/LU solve | Returns false | Caller maps to linear-solve or Newton failure. |
| Backend mismatch | Build/config behavior | CMake and macros own selection. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_matrix.cpp` | Matrix operation behavior. |
| `tests/test_mini_matrix.cpp` | Custom backend behavior. |
| `tests/test_memory.cpp` | Allocation behavior. |
| `docs/matrix/*.md` | Matrix policy and benchmark notes. |

## Known Gaps

- Matrix backend behavior is covered by
  [`../contracts/matrix-backend-contract.md`](../contracts/matrix-backend-contract.md)
  and the zero-allocation evidence in
  [`../contracts/memory-allocation-contract.md`](../contracts/memory-allocation-contract.md).
