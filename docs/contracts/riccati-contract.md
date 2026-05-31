# Riccati Contract

Status: draft

Owner modules:

- `MOD-SOLVER-RICCATI`
- `MOD-CORE-SEMANTICS`

Related modules:

- `MOD-MATRIX`
- `MOD-ALG-EVAL`
- `MOD-ALG-INIT`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

KKT inputs, barrier derivative assembly, sigma convention, soft row derivatives,
backward/forward Riccati solve, dual recovery, SOC solve, and unsupported
backend failure.

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Evaluated trajectory packets, `mu`, `reg`, inertia strategy, config, optional affine/SOC data |
| Outputs | Search directions, feedback gains, barrier-modified packets, `LinearSolveResult` |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `RIC-001` | Riccati solve consumes current model packets and writes directions without allocating. | `covered` |
| `RIC-002` | Hard-row assembled diagonal contribution uses the standard `lam/s` convention. | `covered` |
| `RIC-003` | Pure L1 soft rows use shared `soft_s` and implicit dual `w1 - lam`. | `covered` |
| `RIC-004` | Pure L2 soft rows use `1 / (s/lam + 1/w)` with effective L2 weight. | `covered` |
| `RIC-005` | Mixed L1+L2 rows use one shared relaxation with implicit dual `w1 + w2*soft_s - lam`. | `covered` |
| `RIC-006` | Dual/slack direction recovery must match the same hard/L1/L2/mixed row formulas used in derivative assembly. | `covered` |
| `RIC-007` | Unsupported GPU backends fail explicitly and do not silently run CPU. | `covered` |
| `RIC-008` | `LinearSolveResult` carries degraded-step diagnostics with the solve result. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Factorization failure | Return failed `LinearSolveResult`. |
| Unsupported backend | Return failed `LinearSolveResult` and log diagnostic. |
| Degraded/frozen direction | Return successful result with degraded diagnostics when valid. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `RIC-001` | `tests/test_riccati.cpp::RiccatiTest.TrivialLQR`, `tests/test_memory.cpp::MemoryTest.ZeroMalloc_Compliance_Test`, `tests/test_memory.cpp::MemoryTest.ZeroMalloc_ConfigMatrixSolve` | `covered` |
| `RIC-002` | `tests/test_soft_constraints.cpp::SoftConstraintTest.HardBarrierDerivativeUsesLamOverS` | `covered` |
| `RIC-003` | `tests/test_soft_constraints.cpp::SoftConstraintTest.L1BarrierDerivativeUsesSharedSoftDualFloor`, `tests/test_soft_constraints.cpp::SoftConstraintTest.PureL1DualRecoveryUsesImplicitSoftDual` | `covered` |
| `RIC-004` | `tests/test_soft_constraints.cpp::SoftConstraintTest.PureL2BarrierDerivativeUsesEffectiveWeight`, `tests/test_soft_constraints.cpp::SoftConstraintTest.PureL2DualRecoveryUsesEffectiveWeight`, zero/tiny L2 tests | `covered` |
| `RIC-005` | `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2BarrierDerivativeUsesBothWeights` | `covered` |
| `RIC-006` | `tests/test_soft_constraints.cpp::SoftConstraintTest.HardDualRecoveryMatchesBarrierDerivativeFormula`, `tests/test_soft_constraints.cpp::SoftConstraintTest.PureL1DualRecoveryUsesImplicitSoftDual`, `tests/test_soft_constraints.cpp::SoftConstraintTest.PureL2DualRecoveryUsesEffectiveWeight`, `tests/test_soft_constraints.cpp::SoftConstraintTest.MixedL1L2DualRecoveryUsesBothWeights` | `covered` |
| `RIC-007` | `tests/test_features.cpp::FeaturesTest.GPUBackendUnsupportedFailsExplicitly`, `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigPreservesConstructorBackend` | `covered` |
| `RIC-008` | `tests/test_riccati.cpp::RiccatiTest.NonSPDQuuFreezesControlDimsInsteadOfFailing`, `tests/test_status.cpp::StatusTest.SolverInfoReportsDegradedRiccatiStep` | `covered` |

## Open Gaps

- No open P0 Riccati formula coverage gaps. Future backend-specific Riccati
  implementations should add backend rows instead of weakening these CPU
  formula tests.
