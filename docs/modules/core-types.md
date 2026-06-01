# Core Types And Solver Context

Module ID: `MOD-CORE-TYPES`

Status: draft

Files:

- `include/minisolver/core/types.h`
- `include/minisolver/core/solver_context.h`
- `include/minisolver/core/solver_plan.h`
- `include/minisolver/core/gpu_types.h`

Owner layer:

- Fixed-size solver data carriers and diagnostic/result types.

## Purpose

Own the solver's shared enums, status strings, public diagnostic info,
per-knot state/matrix storage, phase result structs, and build-plan state.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Template dimensions `NX`, `NU`, `NC`, `NP` | Model type | Compile-time constants. |
| Algorithm phase metrics | Solver route and kernels | Values are written by the owning phase. |
| Status/reason values | Solver route, postsolve, failure paths | Must be valid enum values before public projection. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| `SolverStatus`, `TerminationReason`, `ApiStatus` | Public API, tests, logs | Stable result vocabulary. |
| `SolverInfo` | User diagnostics and tests | Final public solve summary. |
| `KnotState` | Trajectory double buffering | Scalar/vector state copied across candidates. |
| `KnotMatrices` | Model evaluation and Riccati | Large derivative/work matrices recomputed per iteration. |
| `SolverContext` phase structs | Solver route | Internal phase state and result payloads. |
| `SolverPlanInfo` | Solver component rebuild | Resolved config/build choices. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| `KnotState` | Per knot, active/candidate trajectory | Double-buffered. |
| `KnotMatrices` | Per knot | Not copied by lightweight candidate preparation. |
| `SolverContext` | Per solver instance | Algorithmic, residual, direction, globalization, termination, and info state. |
| `SolverBuildState` | Per solver instance | Tracks dirty build state and resolved plan. |

## Public API Surface

- Status enums and string conversion helpers.
- `SolverInfo`
- `KnotPoint`
- `LineSearchResult`

## Internal Contracts

- `KnotState` participates in candidate copies and swaps.
- `KnotMatrices` are recomputed and should not be required to survive
  lightweight candidate swaps.
- Phase diagnostics should travel through fixed-size result/state structs, not
  mutable side channels.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: data carriers must stay fixed-size and template-resolved.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Unknown status enum in string conversion | Returns `"UNKNOWN"` | Diagnostic only. |
| Invalid phase metric | Owning phase maps to `NUMERICAL_ERROR` or phase-specific failure | This module carries values but does not classify them. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_status.cpp` | Status/reason projection. |
| `tests/test_memory.cpp` | Fixed-size solve-time allocation claims. |
| `tests/test_solver.cpp` | Public solver info behavior. |

## Known Gaps

- Solver status, public info projection, solve-loop phase state, and
  zero-allocation data-carrier requirements are covered by
  [`../contracts/status-semantics-contract.md`](../contracts/status-semantics-contract.md),
  [`../contracts/solver-info-contract.md`](../contracts/solver-info-contract.md),
  [`../contracts/solve-loop-contract.md`](../contracts/solve-loop-contract.md),
  and
  [`../contracts/memory-allocation-contract.md`](../contracts/memory-allocation-contract.md).
  Add a dedicated double-buffer contract only if future changes alter
  `KnotState`/`KnotMatrices` copy ownership.
