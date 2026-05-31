# Core Trajectory

Module ID: `MOD-CORE-TRAJ`

Status: draft

Files:

- `include/minisolver/core/trajectory.h`

Owner layer:

- Active/candidate trajectory storage and swapping.

## Purpose

Own fixed-capacity trajectory buffers, horizon resizing, candidate preparation,
and active/candidate swaps.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Initial horizon | `MiniSolver` constructor | Clamped to `[0, MAX_N]`. |
| New horizon | `MiniSolver::resize_horizon()` | Validated by solver before resize. |
| Active knot state | Solver route and algorithms | Source for candidate preparation. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Active trajectory | Solver phases and public setters/getters | Current iterate. |
| Candidate trajectory | Line search, SOC, backups | Trial iterate or scratch copy. |
| Swapped active/candidate pointers | Solver route | Accepted candidate becomes active. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| `memory_A`, `memory_B` | Solver lifetime | Two fixed arrays of `MAX_N + 1` knots. |
| `active_ptr`, `candidate_ptr` | Solver lifetime | Pointer swap only. |
| `N` | Solver lifetime | Current valid horizon. |

## Public API Surface

- `Trajectory::active()`
- `Trajectory::candidate()`
- `Trajectory::prepare_candidate()`
- `Trajectory::prepare_candidate_full()`
- `Trajectory::swap()`
- `Trajectory::resize()`

## Internal Contracts

- `prepare_candidate()` copies only `KnotState`.
- `prepare_candidate_full()` copies full `KnotPoint`, including matrices.
- `swap()` must not allocate and must not copy buffers.
- Newly exposed knots after grow are reinitialized in both buffers.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: arrays are owned by value; swap is pointer-only.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Invalid resize inside `Trajectory` | No-op | Public validation is owned by `MiniSolver::resize_horizon()`. |
| Stale candidate matrices after lightweight copy | Not a failure | Callers must recompute matrices or request full copy. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_solver.cpp` | Horizon and solve behavior. |
| `tests/test_line_search.cpp` | Candidate preparation through globalization. |
| `tests/test_memory.cpp` | Allocation behavior. |

## Known Gaps

- No dedicated trajectory ownership contract yet.
