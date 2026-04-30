# ADR 0003: MiniSolver Matrix Kernel Layer

Date: 2026-04-30

Status: Accepted

## Context

`MiniMatrix` is becoming a production path for embedded NMPC. The previous
implementation kept most operations as direct two-level or three-level loops
inside `MiniMatrix`, which is simple but makes it hard to specialize hot kernels
without turning the public matrix type into a large general-purpose linear
algebra library.

MiniSolver needs faster fixed-size dense kernels, but the boundary should remain
solver-driven:

- fixed-size, zero-malloc matrices only;
- C++11-compatible generated/embedded code paths;
- predictable compile-time cost;
- no expression-template framework or Eigen feature parity goal.

## Decision

Introduce `include/minisolver/matrix/` as the owner of the custom matrix backend:

- `matrix/mini_matrix.h`: public fixed-size `MiniMatrix`, `MiniDiagonal`, and
  `MiniLLT` types;
- `matrix/matrix_defs.h`: backend selection and `MatOps`;
- `matrix/kernels.h`: small, statically dispatched dense kernels used by
  `MiniMatrix`;
- `matrix/policies.h`: `MatrixPolicy`, the centralized compile-time policy for
  kernel dispatch and platform override thresholds;
- `matrix/static_for.h`: C++11-compatible static unroll utility.

Remove the old `include/minisolver/core/mini_matrix.h` and
`include/minisolver/core/matrix_defs.h` entry points. Solver code, templates, and
tracked generated models must include the new `minisolver/matrix/*` headers
directly.

The kernel layer uses a bounded static-unroll policy: small fixed-size work items
are unrolled at compile time, larger work falls back to ordinary loops. This
keeps generated code size under control while targeting MiniSolver's hot small
matrix workloads.

Policy dispatch is centralized through `minisolver::matrix::MatrixPolicy`.
Kernel code should depend on `MatrixPolicy` directly; do not add parallel
policy aliases or one-off thresholds.

Production policies are intentionally generic defaults, not per-platform
benchmark-winner tables. Users who care about the last few nanoseconds should run
the matrix benchmark on their target compiler, flags, CPU/MCU, and code-size
budget before changing policy thresholds.

## Scope

In scope:

- fixed-size matrix/vector ops already used by MiniSolver;
- fused kernels such as `C += A * B`, `D += A^T * B`, and `d += A^T * b`;
- bit-level NaN/finite checks that remain valid under `-ffast-math`;
- microbenchmarks and correctness tests for non-trivial kernel changes.

Out of scope:

- dynamic matrices;
- sparse matrices;
- expression templates;
- BLAS/LAPACK compatibility;
- replacing Eigen as the reference backend.

## Plan

1. Move matrix backend ownership from `core/` to `matrix/` and remove the old
   `core` matrix entry points.
2. Add C++11 static-unroll kernels and wire current `MiniMatrix` hot operations
   through them.
3. Add focused tests for matrix multiplication, fused transpose products,
   symmetrization, NaN detection, and backend compatibility.
4. Add a small matrix microbenchmark target so future kernel changes can be
   measured before being committed.
5. Only specialize further after profiling shows a real MiniSolver hot path.

## Quality Bar

- Any new kernel must preserve Eigen/custom backend consistency.
- Any performance-motivated kernel change should include a microbenchmark or
  benchmark note.
- Solver code should continue to call `MatOps` or existing `MiniMatrix` methods;
  solver internals should not depend directly on low-level kernel helpers.

## Consequences

The matrix backend now has a clear home and can evolve independently from core
solver types. The old include paths intentionally fail fast, while the kernel
layer gives us a controlled place to add NMPC-specific specializations later.
