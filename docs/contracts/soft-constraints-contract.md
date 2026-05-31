# Soft Constraints Contract

Status: draft

Owner modules:

- `MOD-CORE-SEMANTICS`
- `MOD-ALG-INIT`
- `MOD-SOLVER-RICCATI`
- `MOD-ALG-LS`

Related modules:

- `MOD-MODEL-CODEGEN`
- `MOD-ALG-EVAL`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

L1, L2, mixed L1+L2, zero/tiny weight semantics, shared relaxation, and
generated per-knot soft weights.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `SOFT-001` | Structural soft mode is declared by model/codegen metadata, not inferred by C++ from weight values alone. | `covered` |
| `SOFT-002` | Pure L1 uses one `soft_s` relaxation with penalty `w1*soft_s` and implicit dual `w1 - lam`. | `covered` |
| `SOFT-003` | Pure L2 uses penalty curvature through `lam/w` and no separate L1 soft-dual box. | `covered` |
| `SOFT-004` | Mixed L1+L2 uses one shared `soft_s`; it is not equivalent to independently running pure L1 and pure L2 branches. | `covered` |
| `SOFT-005` | Mixed implicit soft dual is `w1 + w2*soft_s - lam`. | `covered` |
| `SOFT-006` | Declared soft rows with zero/tiny runtime L2 effect use weak effective L2 relaxation rather than becoming hard rows. | `covered` |
| `SOFT-007` | Runtime soft weights are updated per knot before model evaluation/initialization paths that need them. | `covered` |
| `SOFT-008` | Soft-weight validity policy is model/codegen-owned unless a solver invariant requires a core check. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | `constraint_has_l1/l2`, `any_l1/l2`, `l1_weight`, `l2_weight`, `soft_s`, `s`, `lam` |
| Outputs | Soft residuals, KKT derivatives, line-search penalties, unscaled diagnostics |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Zero L2 weight on declared soft row | Use effective floor/weak relaxation. |
| Inactive L1 due to tiny weight | Follow documented soft semantics; do not silently reinterpret user intent without contract update. |
| Non-finite soft residual | Numerical failure at residual/postsolve boundary. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `SOFT-001` | `tests/minimodel/test_constraints.py::test_numeric_zero_soft_weight_keeps_soft_structure`, `tests/minimodel/test_constraints.py::test_numeric_zero_mixed_soft_weight_keeps_same_row_structure`, `tests/test_soft_constraints.cpp::SoftConstraintTest.ZeroL1WeightInitializesAsRegularizedSoftRow`, `tests/test_soft_constraints.cpp::SoftConstraintTest.ZeroL2WeightInitializesAsRegularizedSoftRow` | `covered` |
| `SOFT-002` | `tests/test_soft_constraints.cpp::SoftConstraintTest.L1_Convergence`, `tests/test_soft_constraints.cpp::SoftConstraintTest.L1BarrierDerivativeUsesSharedSoftDualFloor`, `tests/test_soft_constraints.cpp::ComparisonTest.L1_SoftConstraint` | `covered` |
| `SOFT-003` | `tests/test_soft_constraints.cpp::SoftConstraintTest.L2_Convergence`, `tests/test_soft_constraints.cpp::SoftConstraintTest.TinyL2WeightUsesEffectiveFloor`, `tests/test_soft_constraints.cpp::ComparisonTest.L2_SoftConstraint` | `covered` |
| `SOFT-004` | `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2InitializationUsesCombinedSoftDual`, `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2_Convergence`, `tests/test_soft_constraints.cpp::ComparisonTest.MixedL1L2_SoftConstraint` | `covered` |
| `SOFT-005` | `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2BarrierDerivativeUsesBothWeights`, `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2DualRecoveryUsesBothWeights` | `covered` |
| `SOFT-006` | `tests/test_soft_constraints.cpp::SoftConstraintTest.ZeroL2WeightInitializesAsRegularizedSoftRow`, `tests/test_soft_constraints.cpp::SoftConstraintTest.ZeroL1WeightInitializesAsRegularizedSoftRow`, `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2WithBothZeroUsesRegularizedSoftRow`, `tests/test_soft_constraints.cpp::SoftConstraintTest.TinyL2WeightUsesEffectiveFloor` | `covered` |
| `SOFT-007` | `tests/minimodel/test_constraints.py::test_soft_constraint_parameter_weight_packet_updates_knot`, `tests/test_soft_constraints.cpp::SoftConstraintTest.L1RuntimeParameterWeightAffectsSolve` | `covered` |
| `SOFT-008` | `docs/architecture/solver-development-principles.md` | `covered` |

## Open Gaps

- Soft-constraint contract IDs are covered by current unit/codegen tests and
  the model/codegen ownership principle.
