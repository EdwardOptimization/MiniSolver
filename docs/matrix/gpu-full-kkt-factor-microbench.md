# CUDA Full-KKT Factorization Microbenchmark

This note records an exploratory route-1 benchmark for treating the IPM/Newton
step as an explicit full KKT matrix factorization rather than a Riccati
recursion.

The benchmark is intentionally not a solver backend. It assembles synthetic
regularized OCP KKT systems with the structure:

```text
[ H   A^T ] [dz] = [r_z]
[ A  -R_d ] [dl]   [r_c]
```

where `H` is block diagonal by stage and `A` is a dynamics-like block
bidiagonal constraint Jacobian. The matrix is block-sparse by construction but
stored densely in this first probe. The solve uses no-pivot LDL on CPU and CUDA.
This is a sanity baseline for the "full KKT factorization" route, not a
production sparse factorization.

## Reproduction

Configure the CUDA benchmark build:

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native
```

Build and run:

```bash
cmake --build .build_cuda_bench --target cuda_full_kkt_factor_bench -j
.build_cuda_bench/cuda_full_kkt_factor_bench
```

## Measurement

Machine:

- GPU: NVIDIA GeForce RTX 5080
- Metric: dense no-pivot LDL solve of explicit quasi-definite OCP KKT
- Host/device transfers: excluded
- CPU baselines: sequential CPU and threaded CPU

Observed results:

| NX | NU | Horizon N | KKT dim | Batch | Repeats | MiB | CPU us | CPU threaded us | GPU us | GPU speedup vs best CPU | Solution error | Residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 8 | 38 | 1 | 100 | 0.01 | 2.40 | 2.33 | 921.59 | 0.00x | 8.33e-17 | 8.33e-17 |
| 2 | 1 | 16 | 78 | 64 | 20 | 2.97 | 1087.50 | 1502.35 | 10215.91 | 0.11x | 1.11e-16 | 1.04e-16 |
| 4 | 2 | 8 | 76 | 64 | 20 | 2.82 | 1011.45 | 1736.87 | 9483.66 | 0.11x | 1.11e-16 | 1.39e-16 |
| 4 | 2 | 16 | 156 | 16 | 10 | 2.97 | 2366.48 | 943.14 | 61089.95 | 0.02x | 1.11e-16 | 1.53e-16 |
| 8 | 4 | 8 | 152 | 16 | 10 | 2.82 | 2175.80 | 856.81 | 56490.68 | 0.02x | 1.11e-16 | 1.80e-16 |
| 4 | 2 | 24 | 236 | 8 | 5 | 3.40 | 4273.53 | 1183.81 | 195767.61 | 0.01x | 1.11e-16 | 1.53e-16 |

## Interpretation

The correctness signal is good: the CUDA LDL solve matches the CPU solution and
residual to around `1e-16`. The performance signal is negative for this naive
full-matrix route.

The reasons are structural:

- storing the whole KKT densely destroys the OCP sparsity that Riccati exploits;
- one full matrix per horizon gives too little independent GPU work for small
  batches;
- dense no-pivot LDL has poor GPU occupancy in this simple kernel;
- the benchmark excludes host/device transfer, so end-to-end behavior would be
  even worse.

This does not rule out a real GPU sparse/block KKT factorization. It does rule
out a naive dense "assemble full KKT and factor it on GPU" path for MiniSolver's
normal small-to-medium NMPC workloads.

## Backend Implication

The full KKT route should only continue if a future prototype uses real block
sparse storage and a GPU sparse/block factorization kernel. Until then, the
stronger near-term GPU route remains:

```text
device-side generated packet assembly
  -> batched stage-local block work
  -> batched or horizon-parallel Riccati/KKT recovery
```

Do not enable `Backend::GPU_MPX` or `Backend::GPU_PCR` based on this benchmark.
