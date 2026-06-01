# Algorithm Model Evaluation

Module ID: `MOD-ALG-EVAL`

Status: draft

Files:

- `include/minisolver/algorithms/model_evaluation.h`

Owner layer:

- Model packet evaluation and scaling application.

## Purpose

Own stage/terminal model evaluation dispatch, optional true/QP/SOC constraint
packets, objective and row scaling, generated/implicit dynamics dispatch, and
per-stage soft weight refresh before constraint evaluation.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| `KnotPoint` state/control/parameters | Solver route/callback/trajectory | Values are model inputs. |
| Model static functions | Handwritten or generated model | Optional functions detected with traits. |
| `SolverConfig` and `dt` | Solver route | Integrator/scaling choices are valid. |
| Terminal flag | Solver route | Selects stage vs terminal packets. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Cost, gradients, Hessians | Riccati and merit/filter | Objective packet. |
| Dynamics residual and Jacobians | Riccati and residual checks | Discrete dynamics packet. |
| Constraint residuals/Jacobians | Initialization, Riccati, residuals | QP/true/SOC packets. |
| Scaling fields | Residual reporting and candidates | Active objective/row scales. |
| Soft weights | Soft-constraint kernels | Per-knot L1/L2 weights. |

## Owned State

No owned persistent state; mutates caller-owned `KnotPoint`.

## Public API Surface

Internal helpers in `minisolver::detail`, including stage evaluation,
constraint evaluation, scaling application, and residual helpers.

## Internal Contracts

- Generated model owns symbolic packet content.
- Model evaluation owns when packets are refreshed.
- Scaling changes internal residual/cost packets only; user-facing variables
  remain in model units.
- NaN/Inf scale reductions must preserve non-finite evidence instead of silently
  neutralizing it.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: optional packet detection is compile-time.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Non-finite model packet | Later residual/direction/line-search boundary returns `NUMERICAL_ERROR` | Model evaluation does not over-check every packet. |
| Non-finite scale norm | Preserved in scale value | Downstream boundary reports numerical failure. |
| Missing optional true constraints | Falls back to QP constraints | Trait-controlled behavior. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_scaling_regressions.cpp` | Scaling and non-finite preservation. |
| `tests/test_soft_constraints.cpp` | Soft weight refresh. |
| `tests/test_asset_regressions.cpp` | Generated model packet behavior. |
| `tests/test_integrator.cpp` | Dynamics/integrator dispatch. |

## Known Gaps

- Packet ownership and scaling contracts live in
  [`../contracts/constraint-packets-contract.md`](../contracts/constraint-packets-contract.md),
  [`../contracts/generated-packets-contract.md`](../contracts/generated-packets-contract.md),
  [`../contracts/scaling-contract.md`](../contracts/scaling-contract.md), and
  [`../contracts/row-scaling-contract.md`](../contracts/row-scaling-contract.md).
