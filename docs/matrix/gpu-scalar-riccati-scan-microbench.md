# CUDA Scalar Riccati Scan Microbenchmark

This note records an exploratory benchmark for MPX/PCR-style scans applied to a
scalar Riccati backward recursion. It is closer to the old GPU-branch idea than
the generic affine prefix-scan benchmark, but it is still not a full Riccati
solver backend.

## Benchmark Contract

For scalar dynamics and quadratic costs,

```text
x_next = a x + b u
stage cost = q x^2 + r u^2
```

the backward Riccati recursion can be written as a fractional-linear transform:

```text
P_k = q + a^2 r P_{k+1} / (r + b^2 P_{k+1})
    = ((q b^2 + a^2 r) P_{k+1} + q r) / (b^2 P_{k+1} + r)
```

The benchmark scans these fractional-linear transforms in reverse stage order.

Implemented variants:

- `CPU`: sequential host scan of fractional-linear transforms.
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

cmake --build .build_cuda_bench --target cuda_scalar_riccati_scan_bench -j
.build_cuda_bench/cuda_scalar_riccati_scan_bench
```

Run environment:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
```

## 2026-05-18 Result Summary

All scan variants matched the sequential Riccati recursion with errors around
`1e-14`.

Key timing observations:

| N | MPX speedup | PCR speedup | Interpretation |
| ---: | ---: | ---: | --- |
| 64 | 0.01x | 0.01x | GPU launch/scan overhead dominates |
| 256 | 0.05x | 0.03x | GPU is much slower |
| 1024 | 0.19x | 0.11x | GPU is still slower |
| 4096 | 0.53x | 0.33x | GPU is still slower |
| 16384 | 2.03x | 1.18x | Large-horizon crossover appears |
| 65536 | 0.87x | 2.89x | PCR-like scan wins for very large horizon |

This supports the same broad conclusion as the generic prefix-scan benchmark:
MPX/PCR-style scans are credible for large scans or batched workloads, but not
for ordinary single-problem NMPC horizons.

## Backend Implication

This benchmark verifies a Riccati-specific parallel scan identity in a scalar
setting. It does not yet cover:

- multi-state/multi-control block Riccati operators;
- inequality/barrier terms;
- RHS assembly;
- feedback gain recovery;
- line-search interaction;
- host/device transfer or kernel-fusion cost.

Therefore it is useful evidence for future MPX/PCR design, but not enough to
enable `Backend::GPU_MPX` or `Backend::GPU_PCR`.
