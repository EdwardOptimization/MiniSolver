# CUDA Batched Affine Block LQR Riccati Microbenchmark

This note records an exploratory benchmark for many independent block LQR
Riccati direction recursions. It is closer to a real Riccati workload than the
prefix-scan and block-LFT scan microbenchmarks because it includes both Hessian
and affine feedforward terms, but it is still not a MiniSolver backend.

## Benchmark Contract

Each benchmark problem solves the finite-horizon affine LQR backward recursion:

```text
S = R + B^T P_next B
G = B^T P_next A
g = r + B^T p_next
K = S^-1 G
k = S^-1 g
P = Q + A^T P_next A - G^T K
p = q + A^T p_next - G^T k
```

Implemented variants:

- `CPU`: sequential host loop over all independent Riccati problems.
- `CPU-threaded`: persistent `std::thread` workers over the batch.
- `GPU`: one CUDA thread solves one complete independent Riccati direction
  horizon.

Timing excludes host/device transfer and measures device-resident GPU recursion
time. The benchmark uses fixed synthetic stable `A/B/Q/R/Qf/q/r/qf` packets and
does not include model evaluation, barrier terms, inequality handling, dynamics
defect RHS assembly, or line-search interaction.

## Reproduction

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native

cmake --build .build_cuda_bench --target cuda_batched_lqr_riccati_bench -j
.build_cuda_bench/cuda_batched_lqr_riccati_bench
```

Run environment:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
```

## 2026-05-18 Result Summary

All GPU outputs matched the CPU Riccati outputs for both `P` and `p` with max absolute error around
`1e-15` to `3e-15`.

Key timing observations:

| NX | NU | Horizon N | Batch | Threads | GPU vs threaded CPU | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 2 | 32 | 1 | 1 | 0.00x | GPU launch/occupancy overhead dominates |
| 4 | 2 | 32 | 256 | 2 | 0.49x | GPU still slower |
| 4 | 2 | 32 | 4096 | 32 | 1.99x | Batch crossover appears |
| 4 | 2 | 32 | 65536 | 32 | 2.48x | GPU wins for large batch |
| 4 | 2 | 128 | 4096 | 32 | 1.31x | GPU wins for large batch |
| 4 | 2 | 128 | 65536 | 32 | 2.40x | GPU wins for large batch |
| 8 | 4 | 32 | 4096 | 32 | 1.62x | Batch crossover appears |
| 8 | 4 | 32 | 65536 | 32 | 1.51x | GPU wins for large batch |
| 8 | 4 | 128 | 4096 | 32 | 1.49x | Marginal win |
| 8 | 4 | 128 | 65536 | 32 | 3.41x | GPU wins for large batch |

This is the most Riccati-specific benchmark on the branch. It supports the same
core conclusion as the scalar batched Riccati benchmark: GPU is attractive when
there are many independent horizons, but not as a drop-in backend for one normal
NMPC solve.

The exact speedup varies across runs, especially at the largest batch sizes, so
the result should be read as microbenchmark evidence for a workload shape rather
than a stable end-to-end backend performance claim.

## Backend Implication

This benchmark is useful design evidence because it executes a complete affine
block LQR Riccati direction recursion rather than only a scan primitive. It is
still insufficient for enabling `Backend::GPU_MPX` or `Backend::GPU_PCR`
because it does not cover:

- constrained/barrier Riccati packets;
- full dynamics-defect and inequality RHS assembly;
- per-stage varying model and constraint packets;
- host/device transfer or fused model evaluation;
- solver residuals, globalization, SOC, restoration, or postsolve.

The strongest near-term GPU route remains a separate batched workload path:
batched MPC, sampled control, replay/corpus processing, or differentiable
training workloads that solve many related horizons.
