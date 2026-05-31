# Algorithm Line Search

Module ID: `MOD-ALG-LS`

Status: draft

Files:

- `include/minisolver/algorithms/line_search.h`

Owner layer:

- Globalization strategy implementations.

## Purpose

Own no-line-search, merit, and filter line-search strategies, candidate
construction, fraction-to-boundary use, merit/filter scalar calculation,
accepted-step refresh, SOC integration points, and barrier-update history reset.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Active trajectory and directions | Riccati solve | Direction fields are current. |
| Linear solver | Solver route | Used for SOC if enabled. |
| `dt_traj` | Solver route | Valid per interval. |
| `mu`, `reg`, config | Solver context/config | Current algorithmic values. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Accepted active trajectory | Solver route | Candidate swapped into active buffer. |
| `LineSearchResult` | Solver route | Alpha, status, and SOC flags. |
| Strategy history | Strategy instance | Merit/filter internal state. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| Merit penalty/filter entries | Strategy lifetime | Reset on construction, explicit reset, or barrier update where needed. |

## Public API Surface

- `LineSearchStrategy<Model, MAX_N>`
- `NoLineSearch`
- `MeritLineSearch`
- Filter line-search implementation in the same header.

## Internal Contracts

- Candidate evaluation must refresh model packets at accepted/trial points.
- Non-finite merit, filter, or derivative scalars should surface as numerical
  failure, not silent rejection loops.
- Barrier update invalidates comparable merit/filter history when required.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no on default configured paths
- Notes: strategy history must be pre-sized or fixed where allocation claims
  apply.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Tiny boundary alpha | `LineSearchResult` with zero alpha; solver classifies | May become tiny-step or line-search failure. |
| Non-finite merit/filter scalar | `NUMERICAL_ERROR` path | Boundary semantic. |
| Exhausted backtracking | Line-search failure path | Solver route maps final status/reason. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_line_search.cpp` | Merit/filter/no-line-search behavior. |
| `tests/test_soft_constraints.cpp` | Soft penalties and line-search candidates. |
| `tests/test_memory.cpp` | Allocation-sensitive solve paths. |

## Known Gaps

- Separate contracts for merit and filter globalization are not assigned yet.
