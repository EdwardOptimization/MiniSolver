# CUDA Batched Scalar Riccati Microbenchmark

Last updated: 2026-05-18

> Cross-route note: the aligned route re-run is recorded in
> `docs/matrix/gpu-aligned-route-microbench.md`. Treat this file as a
> route-specific probe note; do not use older heterogeneous rows here for
> cross-route speed conclusions.

This note records an exploratory benchmark for many independent short scalar
Riccati recursions. It tests a GPU route that is different from one long
MPX/PCR scan: batching across many MPC samples, guesses, problems, or replay
cases.

## Benchmark Contract

Each CUDA thread solves one independent scalar Riccati backward recursion:

```text
P_k = q + a^2 r P_{k+1} / (r + b^2 P_{k+1})
```

Implemented variants:

- `CPU`: sequential host loop over all problems and all stages.
- `GPU`: one CUDA thread solves one horizon.

Timing excludes host/device transfer and measures device-resident recursion time
for the GPU variant.

## Reproduction

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native

cmake --build .build_cuda_bench --target cuda_batched_scalar_riccati_bench -j
.build_cuda_bench/cuda_batched_scalar_riccati_bench
```

Run environment:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
```

## 2026-05-18 Result Summary

All GPU results matched the CPU results with max absolute error around `1e-15`
to `1e-14`.

Key timing observations:

| Horizon N | Batch | GPU speedup | Interpretation |
| ---: | ---: | ---: | --- |
| 32 | 16 | 0.09x | Kernel launch dominates |
| 32 | 256 | 1.25x | Crossover begins |
| 32 | 4096 | 20.24x | Strong batched speedup |
| 32 | 65536 | 92.55x | Very strong batched speedup |
| 64 | 256 | 1.60x | Crossover begins |
| 64 | 4096 | 28.01x | Strong batched speedup |
| 128 | 256 | 1.85x | Crossover begins |
| 128 | 4096 | 29.52x | Strong batched speedup |
| 256 | 256 | 1.96x | Crossover begins |
| 256 | 4096 | 34.22x | Strong batched speedup |

This is currently the strongest GPU signal on the branch. It suggests GPU work
should be prioritized for batched workloads rather than a single ordinary NMPC
horizon.

## Backend Implication

This benchmark still does not justify enabling `Backend::GPU_MPX` or
`Backend::GPU_PCR` for the normal solver path. It does suggest a separate future
backend category or tool path for:

- batched MPC;
- sampled MPC / MPPI-style candidates;
- multiple shooting guesses;
- replay/corpus processing;
- differentiable or learning workloads that solve many related problems.

The first production-style GPU target should probably be a batched workload API,
not a drop-in replacement for one CPU Riccati solve.
