# Barrier Mu Contract

Status: draft

Owner modules:

- `MOD-ALG-BARRIER`
- `MOD-ALG-INIT`
- `MOD-ALG-TERM`

Related modules:

- `MOD-SOLVER-ROUTE`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Barrier parameter ownership.
- Monotone/adaptive/Mehrotra update semantics.
- Relationship between `mu`, complementarity gap, and centrality residuals.
- Warm-start `mu` selection.

Out of scope:

- New barrier strategies.
- Exact production tuning defaults.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Current `mu` | Solver context | Positive finite under valid state. |
| Average complementarity gap | Residual evaluation | Uses valid positive pairs. |
| Barrier centrality residual | Residual evaluation | Fresh for current iterate. |
| `SolverConfig` | Config | Barrier strategy and tuning values validated. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Updated `mu` | Barrier value for next iteration. | Solver route/Riccati |
| Mehrotra target `mu` | Corrector target. | Solver route/Riccati |
| `SolverInfo::mu` | Final diagnostic. | User/tests |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `SolveState::mu` | `MOD-CORE-TYPES` | Solver algorithmic state. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `BARR-001` | `mu` is algorithmic barrier state, not a standalone convergence certificate. | `covered` |
| `BARR-002` | Monotone strategy decreases `mu` only when centrality residual is ready. | `covered` |
| `BARR-003` | Adaptive strategy targets a safety-scaled average complementarity gap while not increasing `mu`. | `covered` |
| `BARR-004` | Mehrotra strategy computes target `mu` from affine complementarity and centering policy. | `covered` |
| `BARR-005` | `mu` must not be decreased below `mu_final`. | `covered` |
| `BARR-006` | Barrier decrease must notify globalization strategies that depend on barrier history. | `covered` |
| `BARR-007` | Warm-start `mu` reuse must be clamped to valid config bounds. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Invalid reused `mu` | Fallback/clamp in warm-start selection | Not a solve failure. |
| No `mu` decrease | Continue or stagnation later | Not immediate failure. |
| Non-finite barrier residual | `NUMERICAL_ERROR` at residual boundary | Not hidden by barrier update. |

## Numeric And Performance Constraints

- Barrier update must be allocation-free and scalar-only.
- `mu` update should not trigger broad solver state mutation beyond owned
  globalization reset hooks.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `BARR-001` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesTrueComplementarityGapSnapshot`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesKktComplementarityNotBarrierTarget`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceRejectsLargeTrueComplementarityAtMuFinal` | `covered` |
| `BARR-002` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.MonotoneBarrierDecreaseWaitsForCentralityReadiness` | `covered` |
| `BARR-003` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.AdaptiveBarrierTargetsSafeAverageGapWithoutIncreasing` | `covered` |
| `BARR-004` | `tests/test_bugfixes.cpp::BugfixTest.MehrotraMuAffIncludesL1SoftPair`, `tests/test_bugfixes.cpp::BugfixTest.MehrotraAffineMuUsesSeparatePrimalAndDualStepLengths` | `covered` |
| `BARR-005` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.BarrierUpdateDoesNotDecreaseBelowMuFinal` | `covered` |
| `BARR-006` | `tests/test_bugfixes.cpp::BugfixTest.FilterLineSearchClearsFilterOnBarrierUpdate`, `tests/test_bugfixes.cpp::BugfixTest.MeritLineSearchResetsNuOnBarrierUpdate` | `covered` |
| `BARR-007` | `tests/test_config_regressions.cpp::ConfigRegressionTest.DefaultWarmStartResetsBarrierAndRegularization`, `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartCanReusePreviousBarrier`, `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartCanUseComplementarityGapBarrier`, `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartInvalidPrimalDualFallsBackToMuInit` | `covered` |

## Open Gaps

- No open P0 barrier update coverage gaps. Production tuning of strategy
  constants remains benchmark/replay work.
