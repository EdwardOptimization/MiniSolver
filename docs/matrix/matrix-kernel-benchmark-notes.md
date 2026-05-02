# Matrix Kernel Benchmark Notes

Date: 2026-04-30

## Scope

This note records why the current matrix microbenchmark winners are fast and how
to interpret the results before moving any benchmark-local decomposition kernel
into `include/minisolver/matrix/`.

The benchmark data here comes from `matrix_kernel_bench` on:

- CPU: AMD Ryzen 9 9950X3D under a virtualized Linux environment.
- Build flags observed in `.build/CMakeCache.txt`: `-O2`, `-march=nocona`,
  `-mtune=haswell`, `-ftree-vectorize`.
- Repetition policy: 5 full benchmark runs, selecting medians for the summary.

The current decomposition kernels for LDLT, LU, and QR are benchmark-local
experiments. They are not yet production MiniMatrix APIs.

## External Principles

The relevant public guidance is consistent with the local measurements:

- Eigen's fixed-size objects are static arrays, intended to have zero runtime
  allocation overhead; Eigen relies on alignment and packetization to get best
  vectorized performance for fixed-size objects:
  https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicFixedSizeVectorizable.html
- Eigen's LLT documentation says the square-root-free Cholesky form is more
  stable and even faster for solving self-adjoint problems, and also notes that
  storage order can affect factorization performance:
  https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1LLT.html
- Intel's compiler guidance warns that manual unrolling is not automatically
  better: simple loops often give the compiler more freedom, while manual
  unrolling can interfere with vectorization, loop transforms, and portability:
  https://www.intel.com/content/www/us/en/developer/articles/technical/avoid-manual-loop-unrolling.html
- Intel's unroll pragma documentation explicitly calls out the downside:
  unrolling can increase register pressure and code size, and sometimes should
  be disabled:
  https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/unroll-nounroll.html
- GCC's optimization documentation also exposes register-pressure-sensitive
  scheduling and states that unrolling all loops usually makes programs slower:
  https://gcc.gnu.org/onlinedocs/gcc-6.4.0/gcc/Optimize-Options.html
- LAPACK's Cholesky factorization is specialized for symmetric positive
  definite matrices and only references one triangular half of the input:
  https://www.netlib.org/lapack/explore-html/d1/dd3/group__potrf2_ga7a1158271be5fac6e3d89b7ca8d71a07.html

## Why The Winners Are Fast

Algorithm choice dominates unroll choice.

For SPD matrices, LDLT usually wins because it is square-root-free. LLT performs
`sqrt` on every diagonal pivot. On small matrices, that scalar latency is a large
part of the total runtime. LDLT also keeps the unit diagonal in `L`, which avoids
some diagonal normalization work.

For general dense matrices, partial-pivot LU wins because it is the lowest-cost
robust factorization in this set. QR does more arithmetic: it repeatedly computes
dot products, vector updates, norms, and normalizations. QR is useful when the
numerical problem needs it, but it is not the speed baseline for small square
systems.

Static unrolling is useful only when it removes real loop overhead without
creating too much code or register pressure.

For tiny triangular loops, unrolling the outer loop can expose each column/pivot
as a constant-sized block. This lets the compiler remove loop counters and often
fold guarded prefix/suffix loops. That explains why N=4 SPD LDLT and N=4 general
LU prefer outer-loop unrolling.

For medium sizes, row or inner unrolling often wins because the hot work is in
short row updates or reductions. This removes branch/counter overhead in the
most repeated part without fully expanding the whole factorization.

For larger N in this sweep, full unrolling is usually worse. It grows the
instruction stream and increases live temporaries. That creates register pressure
and can reduce instruction-cache locality. This is why SPD N=16 prefers LDLT
with row unrolling, while general LU around N=10..15 often prefers the plain loop.

Eigen is not slow in general; it is solving a more general problem. Eigen's
fixed-size path is strong, but these decomposition kernels are tiny, triangular,
and data-dependent. MiniMatrix can be faster here by specializing exactly the
known layout and algorithm. The current build also uses `-march=nocona` rather
than native AVX2/AVX512, so these numbers should not be treated as a universal
Eigen ceiling on desktop CPUs.

## Median Results

Selected medians from five full runs:

| Kernel | N | Best MiniMatrix variant | Median ns | Best Eigen comparison | Median ns |
| --- | ---: | --- | ---: | --- | ---: |
| SPD decomposition | 4 | `minimatrix_ldlt_unroll_outer` | 5.687 | `eigen_partial_piv_lu` | 20.544 |
| SPD decomposition | 8 | `minimatrix_ldlt_unroll_row_inner` | 40.607 | `eigen_partial_piv_lu` | 87.790 |
| SPD decomposition | 16 | `minimatrix_ldlt_unroll_row` | 157.325 | `eigen_llt` | 244.508 |
| General decomposition | 4 | `minimatrix_lu_partial_unroll_outer` | 6.216 | `eigen_partial_piv_lu` | 19.268 |
| General decomposition | 8 | `minimatrix_lu_partial_unroll_row` | 52.079 | `eigen_partial_piv_lu` | 76.489 |
| General decomposition | 16 | `minimatrix_lu_partial_unroll_outer` | 230.842 | `eigen_partial_piv_lu` | 342.830 |

Observed winner pattern:

- SPD: LDLT wins nearly all sizes; LLT is not the speed baseline once
  square-root-free factorization is available.
- General: partial-pivot LU wins nearly all sizes; QR is consistently slower.
- Fully unrolling every loop is not the right default.

## Factorization Plus Solve Results

The follow-up benchmark measures the form MiniSolver actually pays for:
factorization plus solving `A x = b` and `A X = B`.

This changed the conclusion in one important way. A naive multi-RHS solve that
loops over columns is slow for MiniMatrix. Processing all RHS columns together in
the forward/backward triangular sweeps restores the expected performance and is
the production shape to use.

Selected medians from three full runs after batched RHS solve:

| Kernel | N | Best MiniMatrix variant | Median ns | Best Eigen comparison | Median ns |
| --- | ---: | --- | ---: | --- | ---: |
| SPD solve, vector RHS | 8 | `minimatrix_ldlt_unroll_inner` | 75.763 | `eigen_llt` | about 150 |
| SPD solve, vector RHS | 16 | `minimatrix_ldlt_unroll_row` | 323.648 | `eigen_llt` | 450.288 |
| SPD solve, matrix RHS | 8 | `minimatrix_ldlt_unroll_inner` | 159.442 | `eigen_llt` | about 258 |
| SPD solve, matrix RHS | 16 | `minimatrix_ldlt_unroll_row` | 1046.250 | `eigen_llt` | 1036.290 |
| General solve, vector RHS | 8 | Mini LU best | 125.486 | `eigen_partial_piv_lu` | 117.821 |
| General solve, vector RHS | 16 | `minimatrix_lu_partial_loop` | 436.779 | `eigen_partial_piv_lu` | 497.723 |
| General solve, matrix RHS | 8 | `minimatrix_lu_partial_unroll_row` | 169.911 | `eigen_partial_piv_lu` | 243.869 |
| General solve, matrix RHS | 16 | `minimatrix_lu_partial_loop` | 1118.030 | `eigen_partial_piv_lu` | 1132.100 |

Updated winner pattern:

- SPD vector RHS: Mini LDLT is consistently faster.
- SPD matrix RHS: Mini LDLT is faster through most of the 1..16 sweep, and is
  roughly tied with Eigen LLT at N=16.
- General vector RHS: Eigen PartialPivLU wins in the middle sizes; Mini LU wins
  again for larger sizes in this sweep.
- General matrix RHS: batched Mini LU wins most sizes and is roughly tied with
  Eigen PartialPivLU at N=16.
- QR remains slower for speed-oriented square solves; keep it as a robustness
  option, not the primary hot path.

## Implementation Guidance

Do not promote the fastest benchmark-local variant directly into production.
Promote a small number of solver-relevant kernels after a solve-path benchmark
confirms impact.

Recommended next production candidates:

- SPD solve kernel: fixed-size no-pivot LDLT for regularized SPD systems.
- General solve kernel: fixed-size partial-pivot LU only if the solver has a real
  general dense solve hot path.
- Multi-RHS triangular solves must be batched across RHS columns. Per-column
  solves lose the main advantage and are not acceptable for the Riccati `K`
  path.
- QR: keep out of the first production matrix kernel pass unless a correctness
  case requires least-squares or stronger rank handling.

Recommended unroll policy:

- Avoid a global "full unroll all loops" rule.
- Use size- and kernel-specific dispatch.
- Keep plain-loop fallbacks because they are often fastest once N grows.
- Retest with `-march=native` separately before making desktop-performance
  claims; embedded builds may prefer different thresholds.

Production default policy:

- The default MiniMatrix policy is deliberately conservative and generic. It is
  not the table of fastest variants from this benchmark run.
- The current LDLT factorization default unrolls outer and row control flow only
  for tiny sizes, unrolls short inner reductions for small sizes, and leaves
  larger sizes as ordinary loops.
- Users should tune on their own target by rebuilding with threshold overrides
  such as `-DMINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK=...`,
  `-DMINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N=...`,
  `-DMINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N=...`, and
  `-DMINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N=...`, then running
  `matrix_kernel_bench` with the same compiler flags used in deployment.
- All production thresholds should flow through `MatrixPolicy` in
  `include/minisolver/matrix/policies.h`; avoid adding one-off unroll decisions
  directly in kernel code.
- A tuned setting should be judged by factorization-plus-solve timing, real
  Riccati/solver profiling, compile time, and code size, not by a single
  factorization-only microbenchmark.

Required before production use:

- Add factorization correctness tests against Eigen for solve residuals, not just
  factorization checksums.
- Benchmark factorization plus solve, because MiniSolver pays for both.
- Measure inside real Riccati/factorization timing, not just isolated kernels.
- Keep compile-time growth visible, because full template expansion makes the
  benchmark target noticeably slower to compile.
