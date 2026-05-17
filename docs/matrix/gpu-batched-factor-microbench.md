# CUDA Batched Factorization Microbenchmark

This note records the first exploratory GPU benchmark for small dense matrix
factorization. It is a microbenchmark only: it does not make `Backend::GPU_*`
supported, and it does not measure end-to-end NMPC solve time.

## Benchmark Contract

The benchmark factors batches of small symmetric positive definite matrices with
Cholesky decomposition.

Implemented variants:

- `CPU`: sequential host Cholesky for every matrix in the batch.
- `GPU`: one CUDA thread factors one matrix.

This GPU kernel is intentionally a simple baseline. It is not a tuned
cooperative/shared-memory factorization and should not be treated as the final
GPU backend design.

Timing excludes host/device transfer and measures device-resident factorization
time for the GPU variant.

## Reproduction

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native

cmake --build .build_cuda_bench --target cuda_batched_factor_bench -j
.build_cuda_bench/cuda_batched_factor_bench
```

Run environment:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
```

## 2026-05-18 Result Summary

All GPU factors matched the CPU factors with max factor error around machine
precision and reconstruction error around `1e-15`.

Key timing observations:

| DIM | Batch | GPU speedup | Interpretation |
| --- | ---: | ---: | --- |
| 4 | 1 | 0.00x | Kernel launch dominates |
| 4 | 256 | 0.49x | Still slower than CPU |
| 4 | 4096 | 8.81x | Batch factorization starts to win |
| 4 | 65536 | 30.48x | Strong batched speedup |
| 8 | 256 | 0.62x | Still slower than CPU |
| 8 | 4096 | 9.88x | Batch factorization starts to win |
| 8 | 65536 | 22.63x | Strong batched speedup |
| 12 | 4096 | 6.74x | Batch factorization wins |
| 12 | 65536 | 4.30x | Memory/register pressure reduces benefit |
| 16 | 4096 | 4.68x | Batch factorization wins |
| 16 | 65536 | 5.05x | Moderate batched speedup |

This supports investigating GPU batched matrix kernels for workloads with many
independent small systems. It does not support a single-problem GPU backend for
normal NMPC horizons by itself: batches of `1-256` are still slower than CPU in
this baseline.

## Next Routes To Investigate

- Cooperative one-block-per-matrix Cholesky for `DIM >= 12`.
- Shared-memory/register tiling to reduce global-memory traffic.
- cuSOLVER or CUB-backed batched factorization baselines where available.
- Batching across many MPC problems, samples, shooting guesses, or replay cases.
- Fusion with stage assembly so factorization input is produced directly on
  device.
- CPU SIMD/threaded baselines before claiming GPU wins.

Until these routes show speedups at the intended workload shape,
`Backend::GPU_MPX` and `Backend::GPU_PCR` should remain unsupported
placeholders.
