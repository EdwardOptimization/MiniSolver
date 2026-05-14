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
python3 examples/01_car_tutorial/generate_model.py
cmake -S . -B .build
cmake --build .build --target car_demo -j
./.build/examples/01_car_tutorial/car_demo
```

### `02_advanced_bicycle`

Extended bicycle model used for fused-vs-standard Riccati experiments.

Primary files:

* `generate_advanced_model.py`: generates `generated/bicycleextmodel.h`.
* `advanced_benchmark.cpp`: compares solver configurations on the extended
  bicycle case.
* `advanced_debug.cpp`: debug-oriented executable for inspecting one scenario.
* `run_benchmark.sh`: runs the `no_quad/quad x fused/standard` local benchmark
  matrix. The script temporarily regenerates the model for each mode, then
  restores the default generated header before exiting.

Typical flow:

```bash
python3 examples/02_advanced_bicycle/generate_advanced_model.py
cmake -S . -B .build
cmake --build .build --target advanced_debug advanced_benchmark -j
./.build/examples/02_advanced_bicycle/advanced_debug
./.build/examples/02_advanced_bicycle/advanced_benchmark

# Optional local benchmark matrix. This regenerates tracked generated headers
# during the run and restores the default header before exit.
examples/02_advanced_bicycle/run_benchmark.sh
```

### `03_model_update_callback`

Minimal inline model showing the expert `set_model_update_callback()` hook. The
callback updates a reference parameter before presolve and before each iteration
model evaluation.

Typical flow:

```bash
cmake -S . -B .build
cmake --build .build --target model_update_callback_demo -j
./.build/examples/03_model_update_callback/model_update_callback_demo
```

### `04_dcol_two_cars`

Two opposing cars using the expert model-update callback as a bilevel modeling
hook. Both cars are optimized in one joint MiniModel problem. By default, the
callback uses an example-local analytic support-function oracle for each horizon
knot. A generated one-stage inner MiniSolver is still available as a reference
oracle with `--inner-minisolver`. Both oracles represent the same DCOL-inspired
rectangle uniform-scaling problem:

```text
minimize alpha
subject to p1 in alpha * Rect1(q1)
           p2 in alpha * Rect2(q2)
           ||p1 - p2||_2 <= alpha * 1m
           alpha >= 0
```

The analytic oracle maximizes the rectangle support-function ratio over the
separating normal and uses the envelope theorem to provide the local gradient of
`alpha(q1, q2)`. It explicitly evaluates non-smooth support-feature switch
angles and analytic stationary points inside smooth intervals. The inner
MiniSolver reference path obtains the same local packet from a small generated
problem and its dual variables. The outer generated model consumes the packet as
the local clearance constraint `1 - alpha_local <= 0`, so the solver core remains
geometry-agnostic.

This is an example-local `DCol-lite` construction, not a full implementation of the
DCOL paper's convex primitive optimizer
([arXiv:2207.00669](https://arxiv.org/abs/2207.00669)).

Typical flow:

```bash
python3 examples/04_dcol_two_cars/generate_model.py
cmake -S . -B .build
cmake --build .build --target dcol_two_cars_demo -j
./.build/examples/04_dcol_two_cars/dcol_two_cars_demo
./.build/examples/04_dcol_two_cars/dcol_two_cars_demo --inner-minisolver
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
