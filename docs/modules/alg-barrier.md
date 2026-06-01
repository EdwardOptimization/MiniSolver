# Algorithm Barrier Update

Module ID: `MOD-ALG-BARRIER`

Status: draft

Files:

- `include/minisolver/algorithms/barrier_update.h`

Owner layer:

- Barrier parameter update policy.

## Purpose

Own monotone, adaptive, and Mehrotra barrier update formulas, including
aggressive/non-aggressive centering target policy.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Current `mu` | Solver context | Positive finite under valid solver state. |
| Complementarity residuals/gaps | Residual evaluation | Fresh for current iteration. |
| `mu_aff`, `alpha_aff` | Mehrotra predictor | Meaningful only in corrector paths. |
| `SolverConfig` | Config | Barrier strategy and tuning values validated. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Updated/target `mu` | Solver context and line search | Barrier parameter for next solve phase. |

## Owned State

No owned state.

## Public API Surface

- `detail::BarrierUpdateKernel`

## Internal Contracts

- `mu` is algorithmic state, not an optimality certificate by itself.
- Strict convergence uses complementarity residual, not just `mu`.
- Barrier decrease may invalidate globalization history.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Invalid inputs | Caller boundary should classify | Kernel assumes valid algorithmic state. |
| No decrease | Not a failure | May affect stagnation or loop budget later. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_barrier_residual_contract.cpp` | Barrier/complementarity behavior. |
| `tests/test_termination.cpp` | Interaction with termination. |

## Known Gaps

- Current barrier-mu contracts live in
  [`../contracts/barrier-mu-contract.md`](../contracts/barrier-mu-contract.md).
  Remaining tuning or replay evidence should be tracked in
  [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md).
