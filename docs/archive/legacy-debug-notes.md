# Legacy Debug Notes

This file preserves historical debugging notes that used to live in the root
`DEBUG_README.md`. It is not the authoritative project guide.

Use these current entry points first:

- [Documentation index](../README.md)
- [Roadmap](../ROADMAP.md)
- [Solver refactor plan](../architecture/solver-refactor-plan.md)
- [Testing matrix](../testing/testing-matrix.md)
- [Agent harness](../architecture/agent-harness.md)

## Historical Design Notes

### Python DSL And Generated C++

MiniSolver uses the Python `MiniModel.py` DSL to generate C++ model code rather
than relying on C++ operator overloading or runtime AD. SymPy computes symbolic
Jacobians and Hessians offline, then emits scalar C++ code that is easy to audit
and optimize.

### Fixed-Size Solve Storage

The solve path is designed around fixed-size trajectory buffers and compile-time
dimensions. The active/candidate trajectory buffers allow line search to build a
trial step without heap allocation, then accept by swapping buffers.

The zero-malloc guarantee applies to the configured `solve()` path tested by
`test_memory`. Debug features such as snapshots, profiling, and host logging are
not hard real-time capture mechanisms.

### Matrix Backend Abstraction

The backend abstraction lives under `include/minisolver/matrix/`.

- `USE_EIGEN` uses Eigen for the default desktop backend.
- `USE_CUSTOM_MATRIX` uses MiniSolver's fixed-size MiniMatrix backend.

The custom backend is intended to reduce dependency weight, but full embedded
productization is tracked separately in the review ledgers and architecture
docs.

### Fused Riccati Kernels

Generated models can provide fused Riccati kernels that bake model sparsity into
the generated C++ instructions instead of storing a generic sparse matrix format.
The core solver must still fall back to generic Riccati when the runtime
integrator does not match the generated model.

## Historical Postmortems

### Parameter Synchronization

Symptom: a solve converges for one step, then candidate costs or constraints
become zero or nonsensical.

Root cause: candidate trajectory buffers did not preserve user-set parameters
when line search swapped active and candidate states.

Current guardrail: candidate preparation must preserve parameters and any future
trajectory scratch buffer needs an explicit parameter policy.

### Vanishing Obstacle Gradients

Symptom: an obstacle-avoidance solve stalls when initialized exactly inside a
circular obstacle.

Root cause: squared-distance constraints such as `R^2 - (x^2 + y^2) <= 0` have a
zero gradient at the obstacle center.

Current takeaway: geometry-aware modeling should prefer signed-distance or
gauge-distance forms when possible. Solver-core code should remain
geometry-agnostic and consume only generated constraint packets.

### Restoration Dual Consistency

Symptom: feasibility restoration appears to succeed, then the main IPM loop
immediately diverges.

Root cause: restoration solves a different recovery problem, so its dual
variables are not automatically valid for the original OCP constraints.

Current guardrail: restoration/slack reset rebuild L1 and hard slack-dual pairs
through shared central-path projection helpers.

### Feasible Stagnation

Symptom: a rollout-feasible but high-cost warm start takes tiny steps and hits
the iteration limit.

Typical causes:

- Filter globalization has little infeasibility to trade against cost decrease.
- Barrier updates may become too aggressive for the current nonlinear region.
- The initial feasible trajectory can be far from the local optimum.

Current tools:

- Use `TerminationProfile` and `SolverInfo` to distinguish quality from stop
  reason.
- Try conservative barrier/globalization settings when debugging.
- Use snapshots to replay the exact pre-solve state before changing algorithms.

## Debugging Tools

### Logging

Compile-time logging is controlled by `MINISOLVER_LOG_LEVEL`. Host logging is
not an embedded fixed-buffer logger; embedded-safe logging remains a separate
productization task.

### Snapshots

Use solver snapshots for reproducibility:

```cpp
auto pre_solve = SolverSnapshotIO<CarModel, 100>::capture_snapshot(solver);
SolverStatus status = solver.solve();
SolverSnapshotIO<CarModel, 100>::save_failure_snapshot(
    "failure.msnap", pre_solve, status);
```

Then replay locally with the replay tool:

```bash
./replay_solver failure.msnap
```

### Auto Tuning

`auto_tuner` is a developer tool for exploring solver configuration choices.
It is not a substitute for scenario-specific benchmark evidence.
