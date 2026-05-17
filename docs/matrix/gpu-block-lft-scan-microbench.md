# CUDA Block-LFT Scan Microbenchmark

This note records an exploratory benchmark for MPX/PCR-style scans over block
linear-fractional transform operators. It is closer to a block Riccati
parallelization route than the scalar Riccati scan, but it is still not a solver
backend and does not make `Backend::GPU_*` supported.

## Benchmark Contract

Block Riccati recursions can be represented as linear-fractional transforms of
the value Hessian:

```text
P_next = (A P + B) (C P + D)^-1
```

Composing two such operators is equivalent to multiplying their block operator
matrices. The benchmark isolates that composition scan:

- `CPU`: sequential host inclusive scan of block operators.
- `MPX-like`: device-resident `thrust::inclusive_scan`.
- `PCR-like`: custom Hillis-Steele scan kernels.

Timing excludes host/device transfer and measures device-resident scan time for
the GPU variants.

This benchmark does not evaluate the transform on `P`, invert `(C P + D)`, build
stage Riccati operators, recover feedback gains, or interact with inequalities
and barrier terms.

## Reproduction

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native

cmake --build .build_cuda_bench --target cuda_block_lft_scan_bench -j
.build_cuda_bench/cuda_block_lft_scan_bench
```

Run environment:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
```

## 2026-05-18 Result Summary

All scan variants matched the CPU operator scan with max absolute errors around
`1e-14` to `1e-13`.

Key timing observations:

| NX | N | MPX speedup | PCR speedup | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 2 | 4096 | 0.62x | 0.34x | GPU scan is slower |
| 2 | 16384 | 0.25x | 1.08x | PCR barely crosses over |
| 4 | 4096 | 0.23x | 0.52x | GPU scan is slower |
| 4 | 16384 | 0.69x | 1.09x | PCR barely crosses over |
| 6 | 4096 | 0.33x | 0.54x | GPU scan is slower |
| 6 | 16384 | 0.68x | 1.06x | PCR barely crosses over |

This strengthens the current conclusion: MPX/PCR-style scans may become useful
for very large scans or batched workloads, but they do not justify a normal
single-problem MiniSolver GPU backend for horizons in the usual NMPC range.

## Backend Implication

The result is useful because it moves beyond scalar Riccati and measures a
block-operator composition route. It is still insufficient for backend
integration because it excludes:

- Riccati operator assembly from model, barrier, and constraint packets;
- feedback gain and RHS recovery;
- terminal and stage-specific block structure;
- host/device transfer and kernel fusion costs;
- solver-quality metrics and globalization interaction.

`Backend::GPU_MPX` and `Backend::GPU_PCR` should remain unsupported until a
full deterministic GPU Riccati/KKT direction path passes the backend integration
gate in `gpu-route-triage.md`.
