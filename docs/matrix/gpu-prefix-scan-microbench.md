# CUDA Prefix-Scan Microbenchmark

This note records the first exploratory GPU benchmark for MPX/PCR-style
parallel Riccati building blocks. It is a microbenchmark only: it does not make
`Backend::GPU_*` supported, and it does not measure end-to-end NMPC solve time.

## Benchmark Contract

The benchmark composes affine operators

```text
x_next = A_i x + b_i
```

with an inclusive prefix scan. This isolates the scan/reduction pattern used by
MPX/PCR-style parallel recursions without committing to the full solver backend.

Implemented variants:

- `CPU`: sequential inclusive scan on the host.
- `MPX-like`: device-resident `thrust::inclusive_scan`.
- `PCR-like`: custom Hillis-Steele scan kernels.

Timing excludes host/device transfer and measures device-resident scan time for
the GPU variants.

## Reproduction

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native

cmake --build .build_cuda_bench --target parallel_scan_gpu_bench -j
.build_cuda_bench/parallel_scan_gpu_bench
```

Run environment:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
```

## 2026-05-18 Result Summary

All GPU results matched the CPU prefix scan with max absolute error at roughly
`1e-15`.

Key timing observations:

| NX | N | MPX speedup | PCR speedup | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 2 | 4096 | 0.20x | 0.08x | GPU scan is slower |
| 2 | 65536 | 0.27x | 0.77x | GPU scan is still slower |
| 4 | 4096 | 0.47x | 0.36x | GPU scan is slower |
| 4 | 65536 | 2.16x | 2.42x | Large-batch crossover appears |
| 8 | 4096 | 0.26x | 0.29x | GPU scan is slower |
| 8 | 65536 | 1.97x | 1.11x | Large-batch crossover appears |
| 12 | 4096 | 0.33x | 0.28x | GPU scan is slower |
| 12 | 65536 | 1.05x | 0.79x | Marginal/no useful speedup |

This does not support integrating a GPU backend for normal single-problem NMPC
horizons. For horizons around `50-200`, kernel launch and scan overhead dominate
the device-resident computation. The only positive signal is for much larger
`N` or a future batched-multiple-problem setting.

## Next Routes To Investigate

- Batched multi-problem scan: likely more relevant than one very long horizon.
- Custom CUB/shared-memory scan for small fixed `NX` instead of generic Thrust
  scans over large structs.
- Batched small dense factorization kernels for the local per-stage matrix work.
- Kernel fusion with Riccati RHS assembly to avoid moving large affine operator
  packets between separate kernels.
- CPU SIMD/threaded baselines before claiming GPU wins.

Until those routes show stronger evidence, `Backend::GPU_MPX` and
`Backend::GPU_PCR` should remain unsupported placeholders.
