# Algorithm Initialization

Module ID: `MOD-ALG-INIT`

Status: draft

Files:

- `include/minisolver/algorithms/initialization.h`

Owner layer:

- Primal-dual initialization and warm-start algorithmic scalar selection.

## Purpose

Own cold/reuse initialization decisions, hard/L1/L2/mixed central-path slack and
dual initialization, average complementarity gap calculation, and warm-start
selection for `mu` and regularization.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Current knot residuals and soft weights | Model evaluation | Weights refreshed before per-row initialization. |
| `SolverConfig` | Solver route | Floors, initialization mode, and warm-start modes are valid. |
| Previous `mu`/`reg` | Solver context | May be clamped before reuse. |
| Stored primal-dual validity | Solver route | Determines whether reuse is allowed. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Initialized `s`, `lam`, `soft_s` | Riccati and line search | Interior primal-dual starting point. |
| Selected `mu` | Solver context | Barrier value for solve. |
| Selected `reg` | Solver context | Regularization value for linear solve. |

## Owned State

No owned state; mutates caller-owned knot and solver scalar state.

## Public API Surface

- `detail::InitializationKernel`
- `detail::WarmStartKernel`

## Internal Contracts

- Soft weights are updated before constraint primal-dual initialization.
- Strict L1, L2, mixed, and hard formulas are selected through shared semantics.
- Non-finite or invalid reused `mu/reg` is clamped at the warm-start boundary.

## Hot-Path And Allocation Policy

- Hot path: setup path before solve/iteration
- Solve-time allocation allowed: no
- Notes: row loops must stay fixed-size.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Invalid reused `mu` | Falls back to `config.mu_init` | Boundary clamp, not solver failure. |
| Invalid reused `reg` | Falls back or clamps by warm-start mode | Boundary clamp. |
| Degenerate L1 equation | Uses coefficient floor fallback | Avoids division instability. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_barrier_residual_contract.cpp` | Barrier/complementarity behavior. |
| `tests/test_soft_constraints.cpp` | Soft initialization paths. |
| `tests/test_termination.cpp` | Warm-start related termination behavior. |

## Known Gaps

- Current initialization and mixed soft central-path contracts live in
  [`../contracts/initialization-contract.md`](../contracts/initialization-contract.md)
  and
  [`../contracts/soft-constraints-contract.md`](../contracts/soft-constraints-contract.md).
  Remaining evidence gaps should stay in the coverage matrix, not this module
  summary.
