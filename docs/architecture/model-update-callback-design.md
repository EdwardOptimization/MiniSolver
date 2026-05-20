# Model Update Callback Design

Status: first implementation landed as a function-pointer hook on `MiniSolver`.

Owner: MiniSolver runtime API.

## Purpose

`set_model_update_callback()` is an expert hook for nonlinear NMPC cases where
model data must be refreshed inside a solve, for example:

- moving reference parameters;
- external environment parameters;
- nonlinear model data computed outside the generated model;
- iterative linearization aids that are represented as ordinary solver
  parameters.

This is not a progress logger, a plugin framework, or a general user-interrupt
mechanism.

## API Shape

Each `MiniSolver<Model, MAX_N>` specialization exposes:

```cpp
using ModelUpdateCallback = ApiStatus (*)(MiniSolver& solver, void* user);

ApiStatus set_model_update_callback(ModelUpdateCallback callback, void* user = nullptr);
void clear_model_update_callback();
```

The callback receives the solver reference because the intended use is to call
existing solver setters, especially `set_parameter()` / `set_global_parameter()`
and, for advanced cases, `set_initial_state()` or `set_dt()`. This is an expert
hook, not a sandbox: users are responsible for preserving solver invariants.

No `std::function` is used. The callback mechanism itself performs no heap
allocation.

## Call Order

The callback is invoked:

1. once before presolve, with `get_iteration_count() == 0`;
2. before every solve-loop iteration model evaluation.

The presolve call is intentional: slack and dual initialization can evaluate
model constraints, so callback-updated parameters must be visible before
presolve initializes those variables.

If users do not want to update on the presolve call, they can check
`solver.get_iteration_count() == 0` in their callback and return immediately.

## Contract

Recommended callback operations:

- update stage or global parameters;
- refresh reference trajectories;
- update initial state if the model formulation intentionally requires it;
- update `dt` when the model treats it as a finite user-defined scale.

Operations that should not be used from the callback:

- `solve()` recursively;
- `set_config()`;
- `reset()`;
- `resize_horizon()`.

MiniSolver rejects these structural operations while a callback is active. If a
callback attempts them, the current solve exits as `INVALID_INPUT` and the
structural mutation is not applied. Execution plans must be rebuilt at
construction, `set_config()`, snapshot restore, or solve-entry build
boundaries, not mid-iteration.

If the callback returns any status other than `ApiStatus::OK`, the solve exits
with `SolverStatus::INVALID_INPUT`.

## Solver Progress Monitors

Installing a model-update callback disables residual-stagnation and
cost-stagnation loop exits. The callback may change references, parameters,
constraints, or local model approximations before each iteration, so residuals
and objective costs are not assumed comparable across iterations.

It also disables the pre-direction `ACCEPTABLE_NMPC` warm-start shortcut. A
callback may update only the objective or reference while leaving the previous
trajectory primal-feasible; MiniSolver therefore still attempts a direction solve
instead of zero-step accepting the old trajectory. The accepted-step
`ACCEPTABLE_NMPC` primal-feasible exit remains available after a direction and
globalization step have been attempted.

This is conservative. Callback-driven problems may run until another exit
condition, such as strict convergence, `ACCEPTABLE_NMPC` primal feasibility,
fixed-iteration RTI, line-search failure, or the iteration budget. Users who need
progress-based stopping with callback-updated models should implement that logic
inside their callback or surrounding control loop using problem-specific
comparability assumptions.

## Boundaries

This hook deliberately does not expose a public strategy/plugin object. Users
still choose built-in solver behavior through `SolverConfig`; the callback is
only for updating model data through existing solver APIs.

Callback code itself is user code. MiniSolver only guarantees that the dispatch
mechanism does not allocate; user callbacks must avoid allocation if they need a
hard real-time solve loop.
