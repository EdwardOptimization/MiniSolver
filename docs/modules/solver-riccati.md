# Solver Riccati And Linear Solve

Module ID: `MOD-SOLVER-RICCATI`

Status: draft

Files:

- `include/minisolver/solver/riccati.h`
- `include/minisolver/solver/kkt_assembler.h`
- `include/minisolver/algorithms/riccati_solver.h`
- `include/minisolver/algorithms/linear_solver.h`
- `include/minisolver/algorithms/linear_solve_result.h`

Owner layer:

- KKT/Riccati direction solve and linear-solve result reporting.

## Purpose

Own barrier derivative assembly, Riccati backward/forward solve, dual/slack
direction recovery, affine/corrector support, SOC direction solve, optional
dynamics-defect refinement, and unsupported backend reporting.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Active trajectory matrices and vectors | Model evaluation and initialization | Derivatives are current for the active iterate. |
| `mu`, `reg`, inertia strategy | Solver context/config | Positive finite algorithmic values. |
| Soft/hard constraint metadata | Core semantics/model traits | Structural metadata is compile-time. |
| Optional affine/SOC trajectory | Mehrotra/SOC callers | Contains consistent predictor or SOC RHS data. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| `dx`, `du`, `ds`, `dlam`, `dsoft_s` | Line search and residual checks | Search direction. |
| `K`, `d` | Direction refinement and rollout use | Feedback gain and feedforward term. |
| `Q_bar`, `R_bar`, `H_bar`, `q_bar`, `r_bar` | Solver diagnostics and residual evaluation | Barrier-modified system. |
| `LinearSolveResult` | Solver route | Success/failure and degraded-step diagnostics. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| `RiccatiWorkspace` | Linear solver lifetime | Fixed-size scratch and factorization workspace. |
| `RiccatiSolver::workspace` | Solver lifetime | Reused to avoid allocation. |

## Public API Surface

- `LinearSolver<TrajArray>` internal polymorphic base.
- `LinearSolveResult`
- `RiccatiSolver<TrajArray, Model>`

## Internal Contracts

- CPU serial Riccati is the implemented backend.
- GPU backends must fail explicitly until implemented.
- Barrier derivative formulas must use shared constraint semantics.
- Linear-solve failure must return a structured result, not partially hidden
  side state.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: workspace is persistent; scratch dual-residual evaluation should use
  local fixed-size workspace.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Factorization/linear solve failure | `LINEAR_SOLVE_FAILED` | Solver route owns retry and status projection. |
| Degraded/frozen step | `degraded_step` diagnostics | Not necessarily terminal. |
| Unsupported GPU backend | Failed `LinearSolveResult` | No silent CPU fallback. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_riccati.cpp` | Riccati solve behavior. |
| `tests/test_implicit_sparse_riccati.cpp` | Generated/fused Riccati paths. |
| `tests/test_soft_constraints.cpp` | Soft row KKT derivatives. |
| `tests/test_memory.cpp` | Allocation behavior. |

## Known Gaps

- Riccati contract IDs for sigma convention and soft derivatives are not
  assigned yet.
