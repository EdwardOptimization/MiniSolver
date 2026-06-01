# Solver Route

Module ID: `MOD-SOLVER-ROUTE`

Status: draft

Files:

- `include/minisolver/solver/solver.h`

Owner layer:

- Public solver API and canonical solve orchestration.

## Purpose

Own `MiniSolver`, public setters/getters, model-update callback handling,
component rebuild, presolve, solve loop orchestration, postsolve, diagnostics,
and top-level failure/status projection.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| `Model` static interface | Handwritten or generated model | Compile-time dimensions and required static functions exist. |
| `SolverConfig` | Constructor or `set_config()` | Validated before use. |
| Initial trajectory data | Public setters, warm start, callback | Mutations must respect callback structural restrictions. |
| Algorithm phase results | Linear solver, line search, termination kernels | Results are fixed-size status payloads. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| `SolverStatus` from `solve()` | User/tests | Final postsolve-classified status. |
| `SolverInfo` | User/tests/logging | Final diagnostics and loop status. |
| Updated active trajectory | User getters and next solve | Accepted iterate. |
| `ApiStatus` | Public mutators | Input and structural mutation results. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| `Trajectory` | Solver lifetime | Active/candidate buffers. |
| `SolverContext` | Solver lifetime | Algorithmic and diagnostic state. |
| `dt_traj` | Solver lifetime | Per-interval time steps. |
| Linear solver and line-search strategy | Rebuilt when config plan is dirty | Internal implementation choice. |
| Name maps | Solver lifetime | State/control/parameter lookup. |
| Callback state | Solver lifetime | Model update hook and user pointer. |

## Public API Surface

- `MiniSolver<Model, MAX_N>`
- Constructor, `solve()`, `reset()`, `resize_horizon()`, `set_config()`
- State/control/parameter/time-step setters and getters
- `set_model_update_callback()`
- `get_info()`, `get_iteration_count()`

## Internal Contracts

- The solve route should stay the canonical phase sequence.
- Side concerns belong in phase entry points or fixed-size result structs.
- Postsolve is the final authority for public status classification.
- Callback may update model data, not structural solver state.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no on default solve paths
- Notes: profiling/logging may allocate only when enabled and excluded from
  zero-allocation claims.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Invalid public mutator input | `ApiStatus` | Does not require `solve()`. |
| Unsupported backend | `LINEAR_SOLVE_FAILED` path through linear solve | GPU is reserved but not implemented. |
| Nonfinite residual/direction/line-search scalar | `SolverStatus::NUMERICAL_ERROR` | Boundary checks should be explicit. |
| Loop early exit | `loop_status` plus `termination_reason` | Postsolve may upgrade/downgrade final status. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_solver.cpp` | Public solve behavior. |
| `tests/test_status.cpp` | Status and reason semantics. |
| `tests/test_termination.cpp` | Termination route behavior. |
| `tests/test_memory.cpp` | Zero-allocation solve paths. |
| `tests/test_solver_snapshot.cpp` | Snapshot access through friend boundary. |

## Known Gaps

- Full solve-loop behavior is covered by
  [`../contracts/solve-loop-contract.md`](../contracts/solve-loop-contract.md),
  [`../contracts/status-semantics-contract.md`](../contracts/status-semantics-contract.md),
  [`../contracts/postsolve-contract.md`](../contracts/postsolve-contract.md),
  and
  [`../contracts/solver-info-contract.md`](../contracts/solver-info-contract.md).
  Public mutators are covered where they belong, primarily config, parameter,
  callback, and warm-start contracts.
