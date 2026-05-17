# MiniSolver Tools

The `tools/` directory contains repository-local benchmark, tuning, and replay
utilities for MiniSolver itself. These tools are not intended to be a
cross-solver benchmark suite.

## Benchmark Suite

* `benchmark_suite/benchmark_suite.cpp`: end-to-end MiniSolver configuration
  benchmark using the car tutorial model. This is a MiniSolver configuration
  benchmark, not a cross-solver comparison.
* `benchmark_suite/BENCHMARK_GUIDE.md`: solver archetype and configuration
  selection guide for the benchmark suite.

## Microbenchmarks

* `matrix_kernel_bench/`: fixed-size matrix kernel benchmark.
* `fast_inverse_bench/`: isolated `fast_inverse()` benchmark and notes.
* `warm_start_bench.cpp`: barrier and primal-dual warm-start behavior.
* `merit_armijo_bench.cpp`: merit line-search Armijo behavior.
* `implicit_sparse_riccati_bench.cpp`: sparse Riccati path for implicit
  integrators.
* `implicit_lu_bench.cpp`: implicit integrator LU reuse experiments.
* `line_search_rollout_bench.cpp`: line-search rollout policy timing.
* `barrier_fusion_bench.cpp`: barrier derivative fusion timing.
* `block_copy_bench.cpp`: fixed-size block-copy timing.

## CUDA Exploratory Benchmarks

CUDA benchmarks are opt-in and are not part of the default build:

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native
```

These targets are standalone route probes. They do not enable
`Backend::GPU_MPX` or `Backend::GPU_PCR`, which remain unsupported until a real
backend integration gate is met.

* `parallel_scan_gpu_bench.cu`: MPX/PCR-style affine prefix scan.
* `cuda_scalar_riccati_scan_bench.cu`: scalar Riccati recurrence as an
  MPX/PCR-style scan.
* `cuda_block_lft_scan_bench.cu`: block linear-fractional-transform scan near a
  block-Riccati operator composition route.
* `cuda_batched_factor_bench.cu`: batched small dense Cholesky with sequential
  CPU, threaded CPU, simple GPU, and cooperative GPU baselines.
* `cuda_batched_scalar_riccati_bench.cu`: many independent scalar Riccati
  horizons.
* `cuda_batched_lqr_riccati_bench.cu`: batched barrier-affine block Riccati
  recursion with synthetic defect RHS and mixed hard/L1/L2 recovery.
* `cuda_generated_packet_upload_bench.cu`: generated-model packet eval/pack and
  host-to-device upload timing for existing example generated models.

See `docs/matrix/gpu-route-triage.md` before interpreting these as solver
backend evidence.

## Debug And Tuning Utilities

* `auto_tuner.cpp`: Monte-Carlo search over MiniSolver configuration choices
  using the car tutorial generated model.
* `replay_solver.cpp`: replay solver snapshots for debugging. It
  currently uses the car tutorial generated model.
* `plot_trajectory.py`: plot trajectory CSV outputs from examples.

## Usage Notes

Most tools are intentionally small single-file programs. Build them through the
top-level CMake project when a target exists, or compile the specific tool
manually while keeping compiler flags and matrix backend consistent with the
experiment being measured.

Tools that include `examples/*/generated/*.h` are model-specific utilities. Keep
model-specific benchmark logic near the example or call it out explicitly in the
tool documentation.

Performance-sensitive changes should include one of:

* A focused microbenchmark if the change targets a local hot path.
* An end-to-end benchmark if the change affects solver behavior.
* A correctness test if the change affects solver semantics.
