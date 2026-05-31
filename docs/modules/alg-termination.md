# Algorithm Termination

Module ID: `MOD-ALG-TERM`

Status: draft

Files:

- `include/minisolver/algorithms/termination.h`
- `include/minisolver/algorithms/residual_stagnation_monitor.h`

Owner layer:

- Loop-level convergence, early-stop predicates, and stagnation detection.

## Purpose

Own strict KKT convergence checks, acceptable NMPC primal shortcut predicate,
fixed-iteration profile detection, tiny-step classification, cost stagnation,
and residual stagnation monitoring.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| `TerminationSnapshot` | Solver route/postsolve | Freshness depends on caller phase. |
| Residual/cost history | Solver context | Updated by solve loop. |
| `SolverConfig` | Config | Tolerances and profile are valid. |
| Callback/profile context | Solver route | Controls residual-stagnation enablement. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Boolean convergence/feasible decisions | Solver route | Whether loop may exit. |
| `SolverStatus` classification | Solver route/postsolve | Candidate quality from current residuals. |
| Residual stagnation result | Solver route | Insufficient-progress loop exit. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| `ResidualStagnationMonitor` window/best state | Per solve | Reset each solve. |

## Public API Surface

- `detail::TerminationKernel`
- `detail::TerminationSnapshot`
- `detail::ResidualStagnationMonitor`

## Internal Contracts

- `OPTIMAL` requires strict KKT: linear ok, primal, dual, and true
  complementarity.
- `FEASIBLE` is primal-acceptable and may be finalized by postsolve.
- Residual stagnation is a loop-level insufficient-progress signal, not an
  infeasibility certificate.
- Tiny-step stagnation does not independently claim strict convergence.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: residual stagnation uses fixed-size/lightweight state.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Max iterations | `MAX_ITER`/`MAX_ITERATIONS` | Solver route owns projection. |
| Cost stagnation | `INSUFFICIENT_PROGRESS`/`COST_STAGNATION` | Requires feasible-enough residual context. |
| Residual stagnation | `INSUFFICIENT_PROGRESS`/`RESIDUAL_STAGNATION` | Final status still postsolve-classified. |
| Tiny feasible step | `FEASIBLE` candidate | Postsolve may upgrade to `OPTIMAL`. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_termination.cpp` | Termination profile and stagnation behavior. |
| `tests/test_status.cpp` | Reason projection. |
| `docs/architecture/termination-design.md` | Current design rationale. |

## Known Gaps

- Termination contract IDs are not assigned yet.
