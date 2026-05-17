# CUDA Batched Block LQR Riccati Microbenchmark

This note records an exploratory benchmark for many independent block LQR
Riccati backward recursions. It is closer to a real Riccati workload than the
prefix-scan and block-LFT scan microbenchmarks, but it is still not a MiniSolver
backend.

## Benchmark Contract

Each benchmark problem solves the finite-horizon LQR backward recursion:

```text
S = R + B^T P_next B
G = B^T P_next A
K = S^-1 G
P = Q + A^T P_next A - G^T K
```

Implemented variants:

- `CPU`: sequential host loop over all independent Riccati problems.
- `CPU-threaded`: persistent `std::thread` workers over the batch.
- `GPU`: one CUDA thread solves one complete independent Riccati horizon.

Timing excludes host/device transfer and measures device-resident GPU recursion
time. The benchmark uses fixed synthetic stable `A/B/Q/R/Qf` packets and does
not include model evaluation, barrier terms, inequality handling, RHS recovery,
or line-search interaction.

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

All GPU outputs matched the CPU Riccati outputs with max absolute error around
`1e-15` to `3e-15`.

Key timing observations:

| NX | NU | Horizon N | Batch | Threads | GPU vs threaded CPU | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 2 | 32 | 1 | 1 | 0.00x | GPU launch/occupancy overhead dominates |
| 4 | 2 | 32 | 256 | 2 | 0.52x | GPU still slower |
| 4 | 2 | 32 | 4096 | 32 | 2.06x | Batch crossover appears |
| 4 | 2 | 32 | 65536 | 32 | 2.33x | GPU wins for large batch |
| 4 | 2 | 128 | 4096 | 32 | 1.22x | Marginal win |
| 4 | 2 | 128 | 65536 | 32 | 2.28x | GPU wins for large batch |
| 8 | 4 | 32 | 4096 | 32 | 1.58x | Batch crossover appears |
| 8 | 4 | 32 | 65536 | 32 | 2.26x | GPU wins for large batch |
| 8 | 4 | 128 | 4096 | 32 | 1.31x | Marginal win |
| 8 | 4 | 128 | 65536 | 32 | 3.16x | GPU wins for large batch |

This is the most Riccati-specific benchmark on the branch. It supports the same
core conclusion as the scalar batched Riccati benchmark: GPU is attractive when
there are many independent horizons, but not as a drop-in backend for one normal
NMPC solve.

## Backend Implication

This benchmark is useful design evidence because it executes a complete block
LQR Riccati backward recursion rather than only a scan primitive. It is still
insufficient for enabling `Backend::GPU_MPX` or `Backend::GPU_PCR` because it
does not cover:

- constrained/barrier Riccati packets;
- affine RHS propagation and feedback direction recovery;
- per-stage varying model and constraint packets;
- host/device transfer or fused model evaluation;
- solver residuals, globalization, SOC, restoration, or postsolve.

The strongest near-term GPU route remains a separate batched workload path:
batched MPC, sampled control, replay/corpus processing, or differentiable
training workloads that solve many related horizons.
