# CUDA Batched Factorization Microbenchmark

This note records the first exploratory GPU benchmark for small dense matrix
factorization. It is a microbenchmark only: it does not make `Backend::GPU_*`
supported, and it does not measure end-to-end NMPC solve time.

## Benchmark Contract

The benchmark factors batches of small symmetric positive definite matrices with
Cholesky decomposition.

Implemented variants:

- `CPU`: sequential host Cholesky for every matrix in the batch.
- `CPU-threaded`: persistent `std::thread` workers over the batch. The benchmark
  uses one thread for small batches and increases thread count only when each
  worker has enough matrices to amortize launch overhead.
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

| DIM | Batch | Threads | GPU vs seq CPU | GPU vs threaded CPU | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 | 256 | 1 | 0.50x | 0.49x | Still slower than CPU |
| 4 | 4096 | 8 | 8.05x | 4.13x | GPU wins even against threaded CPU |
| 4 | 65536 | 32 | 29.44x | 3.39x | Strong batched speedup, but less dramatic vs threaded CPU |
| 8 | 256 | 1 | 0.61x | 0.63x | Still slower than CPU |
| 8 | 4096 | 8 | 9.62x | 2.36x | GPU wins against threaded CPU |
| 8 | 65536 | 32 | 22.12x | 1.70x | GPU still wins, but threaded CPU closes the gap |
| 12 | 4096 | 8 | 6.63x | 1.30x | Marginal win against threaded CPU |
| 12 | 65536 | 32 | 4.45x | 0.64x | Threaded CPU wins |
| 16 | 4096 | 8 | 5.91x | 1.36x | Marginal win against threaded CPU |
| 16 | 65536 | 32 | 3.06x | 0.62x | Threaded CPU wins |

This supports investigating GPU batched matrix kernels for workloads with many
independent small systems, but the threaded CPU baseline makes the conclusion
more specific: the simple one-thread-per-matrix CUDA kernel is strong for very
small matrices at large batch, and much less convincing for larger `DIM` once a
reasonable CPU threaded baseline is used. It does not support a single-problem
GPU backend for normal NMPC horizons by itself: batches of `1-256` are still
slower than CPU in this baseline.

## Next Routes To Investigate

- Cooperative one-block-per-matrix Cholesky for `DIM >= 12`.
- Shared-memory/register tiling to reduce global-memory traffic.
- cuSOLVER or CUB-backed batched factorization baselines where available.
- Batching across many MPC problems, samples, shooting guesses, or replay cases.
- Fusion with stage assembly so factorization input is produced directly on
  device.
- Better CPU SIMD baselines before claiming GPU wins beyond the simple
  sequential and threaded baselines recorded here.

Until these routes show speedups at the intended workload shape,
`Backend::GPU_MPX` and `Backend::GPU_PCR` should remain unsupported
placeholders.
