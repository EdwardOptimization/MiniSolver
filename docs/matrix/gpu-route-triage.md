# GPU Route Triage

This branch investigates whether MPX/PCR and related GPU routes are worth
integrating into MiniSolver. It is intentionally scoped to standalone
microbenchmarks. `Backend::GPU_MPX` and `Backend::GPU_PCR` remain unsupported.

## Current Artifacts

| Artifact | Purpose | Evidence |
| --- | --- | --- |
| `tools/parallel_scan_gpu_bench.cu` | MPX/PCR-style affine prefix scan benchmark | Correctness error around `1e-15`; normal NMPC-scale horizons are slower on GPU |
| `docs/matrix/gpu-prefix-scan-microbench.md` | Scan benchmark contract and RTX 5080 results | Records reproduction commands and scale crossover observations |
| `tools/cuda_batched_factor_bench.cu` | Batched small dense Cholesky benchmark | Correctness error around `1e-15`; large batches show GPU speedup |
| `docs/matrix/gpu-batched-factor-microbench.md` | Batched factorization benchmark contract and RTX 5080 results | Records reproduction commands and batch-size crossover observations |

The branch deliberately does not modify:

- `include/minisolver/algorithms/riccati_solver.h`
- `include/minisolver/solver/solver.h`
- `src/cuda/gpu_ops.cu`
- `SolverConfig`
- `Backend::GPU_*` behavior

## Prompt-To-Artifact Audit

Original request:

```text
新建一个分支，实现mpx和pcr，可以参考原来的gpu分支。
确认在不同规模问题上的加速效果，
建议第一步值对比矩阵分解的速度，而不是端到端对比，
这样工作量太大，也不好对比。
除了mpx和pcr，也可以探索其他路线。
```

| Requirement | Current evidence | Status |
| --- | --- | --- |
| New branch | `feat/gpu-mpx-pcr-microbench` pushed to remote | Done |
| Implement MPX/PCR | Standalone MPX-like Thrust scan and PCR-like Hillis-Steele scan exist in `parallel_scan_gpu_bench.cu` | Partial, benchmark-only |
| Reference old GPU branch | Old branch was inspected; direct cherry-pick was rejected because it is highly stale | Done as design input |
| Confirm speedup at different scales | Scan benchmark covers `NX={2,4,8,12}`, `N={64..65536}` | Done for scan microbenchmark |
| First compare matrix decomposition speed | Batched Cholesky benchmark covers `DIM={4,8,12,16}`, `batch={1..65536}` | Done |
| Avoid end-to-end comparison | No end-to-end solver benchmark was added | Done |
| Explore other route | Batched small dense factorization route added | Done |
| Working solver GPU backend | `Backend::GPU_MPX/GPU_PCR` still explicitly unsupported | Not done |

## Interpretation

### Prefix Scan / MPX-PCR Route

The scan route is mathematically relevant for parallel Riccati and recurrence
algorithms, but the current single-problem benchmark does not justify solver
integration. For normal NMPC horizon lengths, launch overhead and generic scan
cost dominate. Positive signals only appear for very large `N` or future batched
multi-problem workloads.

### Batched Factorization Route

Batched small dense Cholesky is more promising for workloads with many
independent small systems. The baseline kernel wins only when the batch is large
enough. This suggests GPU acceleration should first target batched workloads:

- many MPC problems;
- many shooting guesses;
- sampling-based rollout batches;
- replay/corpus processing;
- batched local QP or collision-oracle subproblems.

It does not justify a single-problem GPU backend for ordinary NMPC horizons.

## Backend Integration Gate

Do not enable `Backend::GPU_MPX` or `Backend::GPU_PCR` until all of these are
true:

1. A GPU kernel path matches CPU Riccati/KKT directions on deterministic test
   problems.
2. The benchmark covers the intended workload shape, not only synthetic large
   batches.
3. Runtime is measured with accuracy, residuals, and failure modes, not only
   kernel time.
4. Host/device transfer and assembly cost are included or explicitly fused away.
5. A CPU SIMD/threaded baseline has been measured.
6. The target use case is explicit: single NMPC instance, batched NMPC, sampled
   control, replay corpus, or differentiable/batched training.

Until this gate is met, GPU support should remain an exploratory benchmark
area, not a solver backend claim.

## Recommended Next Steps

1. Add a batched CPU SIMD/threaded baseline for small dense factorization.
2. Add a cooperative one-block-per-matrix Cholesky kernel for `DIM >= 12`.
3. Add a batched Riccati recurrence correctness microbenchmark before touching
   `RiccatiSolver`.
4. If a real workload appears, benchmark batched MPC or sampled-control
   workloads instead of single-horizon solve time.
5. Keep `src/cuda/gpu_ops.cu` as unsupported until the integration gate passes.
