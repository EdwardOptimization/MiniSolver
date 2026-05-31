# Initialization Contract

Status: draft

Owner modules:

- `MOD-ALG-INIT`
- `MOD-CORE-SEMANTICS`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Slack, dual, soft slack, and warm-start scalar initialization for hard, L1, L2,
and mixed soft constraint rows.

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | `kp.g_val`, soft weights, `mu`, `SolverConfig`, initialization mode |
| Outputs | `kp.s`, `kp.lam`, `kp.soft_s`, selected `mu`, selected `reg` |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `INIT-001` | Constraint initialization refreshes per-knot soft weights before row formulas use them. | `covered` |
| `INIT-002` | Hard rows initialize positive `s` and `lam` on the central path scale. | `covered` |
| `INIT-003` | Pure L1 rows initialize `g + s - soft_s = 0`, `s*lam = mu`, and `soft_s*z = mu` where `z = w1 - lam`. | `covered` |
| `INIT-004` | Pure L2 rows initialize `g + s - lam/w = 0` and `s*lam = mu` using the effective L2 weight. | `covered` |
| `INIT-005` | Mixed L1+L2 rows use one shared `soft_s` and `z = w1 + w2*soft_s - lam`. | `covered` |
| `INIT-006` | Initialization applies barrier and L1 dual floors to avoid invalid interior states. | `covered` |
| `INIT-007` | Warm-start average complementarity uses only valid positive complementarity pairs. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Degenerate L1 quadratic | Use coefficient-degeneracy fallback. |
| Tiny/zero L2 soft weight | Use effective L2 weight floor. |
| Invalid previous warm-start scalar | Clamp or reset at warm-start boundary. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `INIT-001` | `tests/test_soft_constraints.cpp::InitializationTest.ConstraintInitializationRefreshesRuntimeSoftWeight` | `covered` |
| `INIT-002` | `tests/test_soft_constraints.cpp::InitializationTest.HardConstraintInitializesPositiveCentralPath` | `covered` |
| `INIT-003` | `tests/test_soft_constraints.cpp::SoftConstraintTest.L1TinyWeightInitializationStaysFinite`, `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2WithZeroL2UsesL1Path` | `covered` |
| `INIT-004` | `tests/test_soft_constraints.cpp::SoftConstraintTest.ZeroL2WeightInitializesAsRegularizedSoftRow`, `tests/test_soft_constraints.cpp::SoftConstraintTest.TinyL2WeightUsesEffectiveFloor` | `covered` |
| `INIT-005` | `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2InitializationUsesCombinedSoftDual` | `covered` |
| `INIT-006` | `tests/test_soft_constraints.cpp::SoftConstraintTest.L1TinyWeightInitializationStaysFinite`, `tests/test_soft_constraints.cpp::SoftConstraintTest.TinyInactiveL1WeightUsesRegularizedSoftRow`, `tests/test_soft_constraints.cpp::SoftConstraintTest.TinyL2WeightUsesEffectiveFloor` | `covered` |
| `INIT-007` | `tests/test_soft_constraints.cpp::SoftConstraintTest.L1NegativeSoftDualDoesNotReduceAverageComplementarity` | `covered` |

## Open Gaps

- Initialization contract IDs are covered by direct initialization and
  warm-start complementarity tests.
