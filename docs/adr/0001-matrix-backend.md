# ADR 0001: Matrix Backend Strategy (Eigen vs MiniMatrix)

Date: 2026-04-12

Status: Accepted

## Context

MiniSolver targets embedded NMPC with hard real-time constraints. We need:

- Deterministic memory behavior (no hidden heap allocations during `solve()`).
- Predictable data layout and toolchain portability (MCU and cross-compilers).
- Competitive performance for small-to-medium dense systems.

Eigen is a strong reference implementation, but it can be costly for embedded use due to compile-time, binary size, and platform/toolchain friction.

## Decision

- Keep **two** matrix backends:
- `USE_EIGEN`: reference backend for correctness validation and debugging.
- `USE_CUSTOM_MATRIX` (`MiniMatrix`): default backend for embedded/prod usage.

- `MiniMatrix` is an **NMPC-specialized** backend:
- Fixed-size dense matrices/vectors (`std::array`) with **row-major** storage.
- Zero-malloc by construction.
- Only implements operations that MiniSolver actually needs.

## Scope and Non-Goals

In scope:

- The minimal set of dense kernels on MiniSolver hot paths (Riccati/IPM/line-search).
- Cholesky-based SPD solves used by the solver (with explicit SPD assumptions and fallback behavior where needed).
- Utility ops required by the existing `MatOps` abstraction.

Non-goals:

- Building a general-purpose linear algebra library.
- Expression templates, dynamic-sized matrices, or feature parity with Eigen.
- Large sparse linear algebra (if needed later, it must be introduced as a separate, deliberate design).

## Quality Bar

- Any new `MiniMatrix` kernel must have a `vs Eigen` correctness test.
- Hot-path changes should come with a microbenchmark when performance risk is non-trivial.
- "No heap allocation in solve" remains a hard invariant and must be testable.

## Consequences

- Eigen can remain in the repo as a ground-truth reference without blocking embedded adoption.
- Custom backend evolves in a controlled way, driven by solver needs rather than generic completeness.
- API/behavior differences must be handled at the `MatOps` layer, not leaked into solver logic.

