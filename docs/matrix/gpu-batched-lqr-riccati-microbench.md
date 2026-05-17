# CUDA Batched Barrier-Affine Block Riccati Microbenchmark

This note records an exploratory benchmark for many independent block LQR
Riccati direction recursions. It is closer to a real Riccati workload than the
prefix-scan and block-LFT scan microbenchmarks because it includes Hessian,
affine feedforward, stage-varying synthetic barrier-derived constraint packet
terms, dynamics-defect RHS propagation, and mixed hard/L1/L2 slack/dual
direction recovery, but it is still not a MiniSolver backend.

## Benchmark Contract

Each benchmark problem solves the finite-horizon affine LQR backward recursion:

```text
Qbar = Q + C^T Sigma C
Rbar = R + D^T Sigma D
Hbar = H + D^T Sigma C
qbar = q + C^T grad
rbar = r + D^T grad
v = p_next + P_next defect

S = Rbar + B^T P_next B
G = Hbar + B^T P_next A
g = rbar + B^T v
K = S^-1 G
k = S^-1 g
P = Qbar + A^T P_next A - G^T K
p = qbar + A^T v - G^T k

du = -k
constraint_step = C dx_seed + D du
ds, dlam, dsoft_s = mixed hard/L1/L2 primal-dual recovery
```

Implemented variants:

- `CPU`: sequential host loop over all independent Riccati problems.
- `CPU-threaded`: persistent `std::thread` workers over the batch.
- `GPU`: one CUDA thread solves one complete independent Riccati direction
  horizon.

Timing excludes host/device transfer and measures device-resident GPU recursion
time. The benchmark uses fixed synthetic stable `A/B/Q/R/H/Qf/q/r/qf` packets
and fixed synthetic `C/D/Sigma/grad`, defect, slack, lambda, and constraint
residual packets, with stage-varying packet scales. It does not include model
evaluation, host/device integration, line-search interaction, SOC, restoration,
or postsolve.

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

All GPU outputs matched the CPU Riccati outputs for `P`, `p`, and recovered
constraint directions with max absolute error around `9e-16` to `2e-15`.

Key timing observations:

| NX | NU | Horizon N | Batch | Threads | GPU vs threaded CPU | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 2 | 32 | 1 | 1 | 0.00x | GPU launch/occupancy overhead dominates |
| 4 | 2 | 32 | 256 | 2 | 0.71x | GPU still slower |
| 4 | 2 | 32 | 4096 | 32 | 2.08x | Batch crossover appears |
| 4 | 2 | 32 | 65536 | 32 | 3.35x | GPU wins for large batch |
| 4 | 2 | 128 | 4096 | 32 | 1.67x | GPU wins for large batch |
| 4 | 2 | 128 | 65536 | 32 | 3.52x | GPU wins for large batch |
| 8 | 4 | 32 | 4096 | 32 | 1.55x | Batch crossover appears |
| 8 | 4 | 32 | 65536 | 32 | 3.65x | GPU wins for large batch |
| 8 | 4 | 128 | 4096 | 32 | 1.52x | Batch crossover appears |
| 8 | 4 | 128 | 65536 | 32 | 3.71x | GPU wins for large batch |

This is the most Riccati-specific benchmark on the branch. Adding stage-varying
synthetic packet assembly, defect RHS propagation, and mixed hard/L1/L2 recovery
keeps the same core conclusion as the scalar batched Riccati benchmark: GPU is
attractive when there are many independent horizons, but not as a drop-in
backend for one normal NMPC solve.

The exact speedup varies across runs, especially at the largest batch sizes, so
the result should be read as microbenchmark evidence for a workload shape rather
than a stable end-to-end backend performance claim.

## Backend Implication

This benchmark is useful design evidence because it executes a complete
barrier-affine block Riccati direction recursion rather than only a scan
primitive. It is still insufficient for enabling `Backend::GPU_MPX` or
`Backend::GPU_PCR` because it does not cover:

- full dynamics-defect and inequality RHS assembly;
- per-stage varying model and constraint packets;
- host/device transfer or fused model evaluation;
- solver residuals, globalization, SOC, restoration, or postsolve.

The strongest near-term GPU route remains a separate batched workload path:
batched MPC, sampled control, replay/corpus processing, or differentiable
training workloads that solve many related horizons.
