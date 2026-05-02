# MiniSolver Examples

The `examples/` directory contains runnable MiniSolver models and local
demonstrations. These examples are part of the MiniSolver repository itself; they
are intended for onboarding, smoke testing, and solver-internal benchmark
experiments.

Examples should stay small and model-focused. They may include local benchmark
scripts when the benchmark demonstrates the specific generated model or codegen
path used by that example. Reusable solver microbenchmarks and configuration
sweeps belong under `tools/`.

## Cases

### `01_car_tutorial`

Kinematic bicycle obstacle-avoidance tutorial.

Primary files:

* `generate_model.py`: generates `generated/car_model.h` with MiniModel.
* `main.cpp`: solves one obstacle-avoidance scenario and writes
  `trajectory.csv`.
* `run_custom_benchmark.sh`: compiles the shared MiniSolver benchmark suite
  against MiniMatrix and Eigen backends using this tutorial model.

Typical flow:

```bash
cd examples/01_car_tutorial
python3 generate_model.py
cmake -S . -B build
cmake --build build -j
./build/car_demo
```

### `02_advanced_bicycle`

Extended bicycle model used for fused-vs-standard Riccati experiments.

Primary files:

* `generate_advanced_model.py`: generates `generated/bicycleextmodel.h`.
* `advanced_benchmark.cpp`: compares solver configurations on the extended
  bicycle case.
* `advanced_debug.cpp`: debug-oriented executable for inspecting one scenario.
* `run_benchmark.sh`: compares fused and standard Riccati-generated code. The
  script temporarily regenerates the model with fused Riccati disabled, then
  restores the default generated header before exiting.

Typical flow:

```bash
cd examples/02_advanced_bicycle
python3 generate_advanced_model.py
cmake -S . -B build
cmake --build build -j
./build/advanced_debug
./build/advanced_benchmark
```

## Example vs Benchmark Boundary

Example-local benchmark scripts are allowed when they demonstrate behavior of a
specific example model. General MiniSolver microbenchmarks and reusable
configuration benchmarks belong under `tools/`.

Example-local scripts may use benchmark-only compiler flags such as
`-march=native` and `-ffast-math`. Do not treat those scripts as portable
embedded build recipes.

Generated headers in `examples/*/generated/` are checked in so C++ examples can
be built without running Python first. If MiniModel changes, regenerate the
headers and commit the generated diff with the corresponding codegen change.
