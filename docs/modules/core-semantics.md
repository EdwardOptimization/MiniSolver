# Core Constraint And Model Semantics

Module ID: `MOD-CORE-SEMANTICS`

Status: draft

Files:

- `include/minisolver/core/constraint_semantics.h`
- `include/minisolver/core/model_traits.h`

Owner layer:

- Generic structural traits and constraint semantics shared by solver kernels.

## Purpose

Own compile-time model trait detection, scaling activation predicates, soft/hard
constraint classification helpers, soft weight update dispatch, and shared
soft-constraint algebra helpers.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Model static metadata | Handwritten model or generated MiniModel header | Structural flags are compile-time. |
| `KnotPoint` weights/slacks/duals | Model evaluation and initialization | Runtime values may vary per knot. |
| `SolverConfig` | Solver route | Floors and activation policy come from config. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| `hard_constraint_row()` | Riccati, initialization, line search, residuals | Structural hard row predicate. |
| `active_l1_soft_constraint()` | Soft-constraint kernels | Runtime L1 active predicate. |
| `active_l2_soft_constraint()` | Soft-constraint kernels | Runtime L2 active predicate. |
| `effective_l2_soft_weight()` | L2/zero-weight relaxation paths | Floor-protected L2 weight. |
| Scaling predicates | Model evaluation and solver plan | Whether scaling behavior is enabled. |

## Owned State

This module owns no state. It owns reusable predicates and algebra.

## Public API Surface

The helpers live in `minisolver::detail` and are internal implementation API.

## Internal Contracts

- Model/codegen owns structural metadata such as `constraint_has_l1`,
  `constraint_has_l2`, `any_l1_constraints`, and `any_l2_constraints`.
- Solver kernels should not duplicate soft row algebra.
- Finite/validity policy belongs at semantic boundaries, not inside every hot
  predicate.

## Hot-Path And Allocation Policy

- Hot path: yes
- Solve-time allocation allowed: no
- Notes: predicates must remain simple and compile-time prunable where possible.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Missing aggregate soft flags with row flags | Compile-time `static_assert` | Current development code does not preserve old generated headers. |
| Zero/tiny L2 weight | Uses effective floor | Represents weak relaxation, not invalid input. |
| Inactive soft row | Classified by shared helpers | Behavior contract still needs IDs. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_soft_constraints.cpp` | L1/L2/mixed and zero-weight semantics. |
| `tests/test_scaling_regressions.cpp` | Scaling activation side effects. |
| `tests/minimodel/test_constraints.py` | Generated structural metadata. |

## Known Gaps

- Soft/hard row mode contract IDs are not assigned yet.
- Runtime soft weight semantics should be captured in `soft-constraints-contract.md`.
