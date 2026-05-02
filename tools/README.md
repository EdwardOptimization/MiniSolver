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

## Debug And Tuning Utilities

* `auto_tuner.cpp`: Monte-Carlo search over MiniSolver configuration choices
  using the car tutorial generated model.
* `replay_solver.cpp`: replay serialized solver snapshots for debugging. It
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
