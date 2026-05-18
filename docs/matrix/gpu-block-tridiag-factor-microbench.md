# CUDA Block-Tridiagonal Factorization Microbenchmark

This note records an exploratory sparse/block route for MiniSolver's
IPM/Newton linear systems.

Unlike `cuda_full_kkt_factor_bench.cu`, this benchmark does not store the whole
KKT matrix densely. It constructs synthetic block-tridiagonal SPD systems and
solves them with a block Cholesky/Thomas-style factorization:

```text
D_0  E_1^T
E_1  D_1  E_2^T
     E_2  D_2  ...
          ...
```

This can be read as a regularized normal-equation or Schur-complement view of
an OCP Newton step. It is still not MiniSolver Riccati and not a GPU backend; it
is a route probe for explicit sparse/block factorization.

## Reproduction

Configure:

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
cmake --build .build_cuda_bench --target cuda_block_tridiag_factor_bench -j
.build_cuda_bench/cuda_block_tridiag_factor_bench
```

## Measurement

Machine:

- GPU: NVIDIA GeForce RTX 5080
- Metric: explicit block-tridiagonal Cholesky/solve
- Host/device transfers: excluded
- CPU baselines: sequential CPU and threaded CPU

Observed results:

| Block dim | Horizon N | Batch | Repeats | MiB | CPU us | CPU threaded us | GPU us | GPU speedup vs best CPU | Solution error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 16 | 1 | 100 | 0.00 | 1.92 | 1.86 | 219.34 | 0.01x | 3.47e-18 |
| 4 | 64 | 256 | 20 | 3.97 | 1926.05 | 1654.61 | 1698.51 | 0.97x | 1.04e-17 |
| 4 | 256 | 1024 | 5 | 63.88 | 32499.24 | 3725.45 | 7823.57 | 0.48x | 1.04e-17 |
| 8 | 64 | 256 | 10 | 15.88 | 8613.52 | 1708.60 | 7643.32 | 0.22x | 1.04e-17 |
| 12 | 64 | 64 | 10 | 8.93 | 4869.17 | 1615.70 | 18438.57 | 0.09x | 1.04e-17 |
| 12 | 128 | 64 | 5 | 17.93 | 10090.37 | 1833.15 | 36986.73 | 0.05x | 1.04e-17 |

## Interpretation

The explicit block-sparse route is much better aligned with OCP structure than
the dense full-KKT route, and the correctness signal is strong. However, this
simple GPU kernel still assigns one CUDA thread to a full block-tridiagonal
system. It therefore exposes little intra-system parallelism and loses to the
threaded CPU baseline for the measured shapes.

The closest case is `block_dim=4, N=64, batch=256`, where GPU reaches `0.97x`
of the best CPU baseline. Larger block sizes and longer horizons need more
parallelism within each system, not just more independent systems.

## Backend Implication

This benchmark supports a narrower conclusion than the dense full-KKT probe:

- explicit block-sparse structure is the right direction if the project pursues
  a full KKT route;
- a one-thread-per-system block factorization is not enough;
- a serious backend would need block-parallel kernels, cyclic reduction, or a
  GPU sparse/block solver with factorization reuse.

For MiniSolver's near-term GPU work, the stronger route remains batched
structured Riccati/KKT kernels plus device-side packet assembly.
