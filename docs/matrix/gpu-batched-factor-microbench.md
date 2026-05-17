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
- `GPU-simple`: one CUDA thread factors one matrix.
- `GPU-coop`: one CUDA block factors one matrix with 32 threads and shared
  memory.

These GPU kernels are exploratory baselines. They are not tuned enough to be
treated as the final GPU backend design.

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

| DIM | Batch | Threads | GPU-simple vs threaded CPU | GPU-coop vs threaded CPU | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 | 256 | 1 | 0.49x | 0.58x | GPU is still slower |
| 4 | 4096 | 8 | 3.87x | 1.56x | Simple GPU wins; cooperative overhead hurts |
| 4 | 65536 | 32 | 3.37x | 0.38x | Simple GPU wins; cooperative is worse than threaded CPU |
| 8 | 256 | 1 | 0.61x | 1.85x | Cooperative GPU crosses over |
| 8 | 4096 | 8 | 2.60x | 1.29x | Both win, simple is better |
| 8 | 65536 | 32 | 1.66x | 0.42x | Simple GPU wins; cooperative is worse than threaded CPU |
| 12 | 256 | 1 | 0.41x | 2.06x | Cooperative GPU helps mid-size batches |
| 12 | 4096 | 8 | 1.20x | 0.89x | Around parity; threaded CPU remains competitive |
| 12 | 65536 | 32 | 0.53x | 0.69x | Threaded CPU wins |
| 16 | 256 | 1 | 0.36x | 2.66x | Cooperative GPU helps mid-size batches |
| 16 | 4096 | 8 | 1.09x | 0.98x | Around parity |
| 16 | 65536 | 32 | 0.77x | 1.65x | Cooperative GPU wins |

This supports investigating GPU batched matrix kernels for workloads with many
independent small systems, but the threaded CPU baseline makes the conclusion
more specific: the simple one-thread-per-matrix CUDA kernel is strong for very
small matrices at large batch, while the cooperative kernel helps some larger
matrix cases but is not uniformly better. It does not support a single-problem
GPU backend for normal NMPC horizons by itself.

## Next Routes To Investigate

- Tune cooperative one-block-per-matrix Cholesky for `DIM >= 12`, including
  block size and occupancy.
- Shared-memory/register tiling and batched layout tuning to reduce memory and
  scheduling overhead.
- cuSOLVER or CUB-backed batched factorization baselines where available.
- Batching across many MPC problems, samples, shooting guesses, or replay cases.
- Fusion with stage assembly so factorization input is produced directly on
  device.
- Better CPU SIMD baselines before claiming GPU wins beyond the simple
  sequential and threaded baselines recorded here.

Until these routes show speedups at the intended workload shape,
`Backend::GPU_MPX` and `Backend::GPU_PCR` should remain unsupported
placeholders.
