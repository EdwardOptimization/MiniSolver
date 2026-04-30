# MiniMatrix Policy And Benchmark Guide

Date: 2026-05-01

## Goal

`MiniMatrix` uses conservative compile-time policies for fixed-size kernels. The
default should be good enough for normal users without per-project tuning, while
embedded or platform-sensitive users can override thresholds and validate them
with benchmarks.

This is not an Eigen-style expression-template optimizer and not a generated
per-platform tuning table. MiniMatrix remains an NMPC-specialized fixed-size
backend.

## Default Policy

The central policy is `minisolver::matrix::MatrixPolicy` in
`include/minisolver/matrix/policies.h`.

Current defaults:

- `MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK=256`
- `MINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N=4`
- `MINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N=4`
- `MINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N=8`

The generic static-unroll policy is based on scalar work count. For example,
matrix multiplication dispatches using `Rows * Cols * InnerDim`; tiny work items
get static control flow, larger work items use ordinary loops to avoid excessive
register pressure, code size, and compile time.

The LDLT factor policy is separate because factorization has different hot loops
than simple elementwise or matrix-product kernels.

## Platform Override

Override thresholds at configure time:

```bash
cmake -S . -B .build_custom \
  -DUSE_CUSTOM_MATRIX=ON \
  -DMINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK=128 \
  -DMINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N=4 \
  -DMINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N=4 \
  -DMINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N=8
```

These options are compile-time definitions. Rebuild after changing them.

Keep defaults unless target measurements justify a change. A setting that wins
on a desktop compiler can lose on an MCU due to instruction cache, register
pressure, or different optimization passes.

## Benchmark Workflow

Use the benchmark targets with the same compiler, flags, backend, and target CPU
used in deployment:

```bash
# Benchmark binaries are excluded from the default `all` build; build them explicitly.
cmake --build .build_custom --target matrix_kernel_bench barrier_fusion_bench implicit_lu_bench block_copy_bench -j
.build_custom/matrix_kernel_bench
.build_custom/barrier_fusion_bench
.build_custom/implicit_lu_bench
.build_custom/block_copy_bench
```

Interpretation rules:

- Prefer solve-path benchmarks over isolated factorization-only numbers.
- Keep correctness tests against Eigen or analytical residuals for every
  promoted kernel.
- Do not promote benchmark-local variants unless they improve a real solver hot
  path.
- Measure compile time and binary size when raising unroll thresholds.
- Treat small differences in sub-10 ns kernels as noise unless repeated runs are
  consistent.

## Current Production Decisions

Promoted:

- Weighted transpose products in Riccati barrier modification.
- Matrix-RHS LU solve for implicit integrator Jacobian recovery.

Measured but not promoted:

- Dual-step fused matrix-vector pair: slower than the current expression.
- Line-search AXPY fusion: noise-level and inconsistent by size/backend.
- Block static-copy expansion: assignment often slower for larger blocks.

## Deferred Work

An Eigen-like automatic tuning table is intentionally deferred. It would require
operation-specific and size-specific policies, platform grouping, benchmark data
generation, and policy selection rules. Keep it as future work until MiniSolver
has enough real deployments to justify the complexity.
