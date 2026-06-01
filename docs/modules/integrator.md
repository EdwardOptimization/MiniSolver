# Integrator

Module ID: `MOD-INTEGRATOR`

Status: draft

Files:

- `include/minisolver/integrator/implicit_integrator.h`
- `include/minisolver/integrator/newton_solver.h`
- `include/minisolver/integrator/numerical_jacobian.h`

Owner layer:

- Generic integration helpers and implicit dynamics support.

## Purpose

Own implicit Euler, implicit midpoint, Gauss-Legendre two-stage integration,
Newton solves, analytical/numerical continuous Jacobian dispatch, and invalid
dynamics marking on failed implicit solves.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| `x`, `u`, `p`, `dt` | Knot and solver time-step data | Model-unit values. |
| `IntegratorType` | Config or generated model metadata | Must be an implicit type for `ImplicitIntegrator`. |
| `NewtonConfig` | Solver config | Validated at config boundary. |
| Model continuous dynamics/Jacobians | Model static interface | Dynamics required; Jacobians optional. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| `f_resid` | Riccati/residuals | Integrated next state. |
| `A`, `B` | Riccati | Discrete dynamics Jacobians. |
| Newton convergence flag | Integrator caller | Whether implicit solve succeeded. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| `NewtonSolver` scratch | Newton solver instance | Stores residual/Jacobian/delta and optional last solution. |

## Public API Surface

- `ImplicitIntegrator<Model>`
- `NewtonSolver<T, N>`
- `compute_numerical_jacobian<Model, T>()`

## Internal Contracts

- Implicit integrator marks dynamics/Jacobians as NaN on failed implicit solve.
- Analytical continuous Jacobians are preferred when provided.
- Numerical Jacobian is a finite-difference fallback for model dynamics.

## Hot-Path And Allocation Policy

- Hot path: yes for implicit models
- Solve-time allocation allowed: no
- Notes: Newton workspaces are fixed-size.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Newton failure | Mark `f_resid`, `A`, `B` invalid | Solver numerical boundary should catch non-finite packets. |
| Unsupported implicit type | Throws invalid argument | Internal misuse. |
| Singular Newton Jacobian | Optional diagonal damping, then failure | Controlled by `NewtonConfig`. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_integrator.cpp` | Integrator and Newton behavior. |
| `tests/minimodel/test_implicit_patterns.py` | Generated implicit model patterns. |
| `tests/test_implicit_sparse_riccati.cpp` | Implicit generated/Riccati interaction. |

## Known Gaps

- Integrator behavior is covered by
  [`../contracts/explicit-integrator-contract.md`](../contracts/explicit-integrator-contract.md),
  [`../contracts/implicit-integrator-contract.md`](../contracts/implicit-integrator-contract.md),
  and
  [`../contracts/numerical-jacobian-contract.md`](../contracts/numerical-jacobian-contract.md).
