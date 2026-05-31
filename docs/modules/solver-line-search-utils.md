# Solver Line-Search Utilities

Module ID: `MOD-SOLVER-LSUTIL`

Status: draft

Files:

- `include/minisolver/solver/line_search_utils.h`

Owner layer:

- Shared fraction-to-boundary helper semantics.

## Purpose

Own the reusable fraction-to-boundary rule for slacks, duals, and L1 soft
variables used by line search and any direction-alpha logic that needs the same
interior-point boundary policy.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Trajectory | Active or scratch trajectory | Contains current variables and search directions. |
| Horizon `N` | Solver route | Valid for the trajectory. |
| `tau` | Config | Usually in `(0, 1]`. |
| `SolverConfig` | Config | Supplies floors for L1 soft dual boundary. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| `alpha` | Line search or solver route | Maximum boundary-safe step fraction. |

## Owned State

No owned state.

## Public API Surface

- `fraction_to_boundary_rule<TrajVector, ModelType>()` internal helper.

## Internal Contracts

- Hard rows guard `s` and `lam`.
- L1/mixed rows additionally guard `soft_s` and implicit soft dual gap.
- The helper must not duplicate unrelated merit/filter logic.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: loops over fixed-size knots and constraint rows.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Boundary alpha collapses | Caller maps to line-search failure or tiny-step path | Utility only computes alpha. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_line_search.cpp` | Fraction-to-boundary through line-search behavior. |
| `tests/test_soft_constraints.cpp` | L1/mixed soft boundary behavior. |

## Known Gaps

- Fraction-to-boundary contract should be shared by line-search and solver route
  if additional alpha helpers exist.
