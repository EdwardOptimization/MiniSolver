# Warm Start Contract

Status: draft

Owner modules:

- `MOD-ALG-INIT`
- `MOD-SOLVER-ROUTE`
- `MOD-CORE-TYPES`

Related modules:

- `MOD-ALG-TERM`
- `MOD-ALG-BARRIER`
- `MOD-CORE-TRAJ`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Initialization modes.
- Primal-dual reuse validity.
- Barrier and regularization reuse policy.
- Warm-start interaction with acceptable NMPC shortcut.

Out of scope:

- User-level initial guess quality recommendations.
- External shifting strategies for MPC.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Existing trajectory | `MOD-CORE-TRAJ` | May contain valid or invalid primal-dual state. |
| `InitializationMode` | Config | Valid enum. |
| Warm-start `mu/reg` modes | Config | Valid enum. |
| Previous `mu/reg` | Solver context | Clamped before reuse. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Rebuilt or reused `s/lam/soft_s` | Interior primal-dual state. | Riccati/line search |
| Selected `mu` and `reg` | Algorithmic state. | Solver loop |
| `primal_dual_reused_this_solve` | Whether reuse happened. | Termination shortcuts |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| Primal-dual reuse flag | `MOD-CORE-TYPES` | Per solve. |
| Previous `mu/reg` | `MOD-CORE-TYPES` | Solver algorithmic state. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `WARM-001` | `COLD_START` rebuilds primal-dual barrier state for the current problem data. | `covered` |
| `WARM-002` | `REUSE_PRIMAL` reuses primal guess but rebuilds slack/dual/barrier state. | `covered` |
| `WARM-003` | `REUSE_PRIMAL_DUAL` may reuse slack/dual only when stored primal-dual state is valid. | `covered` |
| `WARM-004` | Invalid or non-positive reused `mu` falls back to config bounds/defaults. | `covered` |
| `WARM-005` | Invalid or non-positive reused `reg` falls back or clamps according to regularization mode. | `covered` |
| `WARM-006` | `FROM_COMPLEMENTARITY_GAP` computes `mu` from positive valid complementarity pairs. | `covered` |
| `WARM-007` | Acceptable NMPC zero-step shortcut may only use a reused primal-dual warm start and no callback. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Invalid stored primal-dual guess | Fallback to primal reuse/rebuild | Not a public solver failure. |
| Invalid previous `mu/reg` | Clamp/fallback | Boundary sanitation. |
| Callback present | Disable zero-step shortcut | Avoid stale model-data success. |

## Numeric And Performance Constraints

- Warm-start checks should be done at solve setup, not repeatedly inside hot
  row predicates.
- Reinitialization must stay allocation-free.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `WARM-001` | `tests/test_solver.cpp`, `tests/test_soft_constraints.cpp`, `tests/test_config_regressions.cpp::ConfigRegressionTest.DefaultWarmStartResetsBarrierAndRegularization` | `covered` |
| `WARM-002` | `tests/test_solver.cpp`, `tests/test_config_regressions.cpp::ConfigRegressionTest.DefaultWarmStartResetsBarrierAndRegularization` | `covered` |
| `WARM-003` | `tests/test_solver.cpp`, `tests/test_termination.cpp`, `tests/test_replay_corpus.cpp::ReplayCorpusTest.WarmStartSoftConstraintNeighboringFrameSurvivesActiveSetChange` | `covered` |
| `WARM-004` | `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartCanReusePreviousBarrier`, `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartCanUseComplementarityGapBarrier`, `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartInvalidPrimalDualFallsBackToMuInit` | `covered` |
| `WARM-005` | `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartRegularizationModesAreExplicit` | `covered` |
| `WARM-006` | `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartCanUseComplementarityGapBarrier`, `tests/test_config_regressions.cpp::ConfigRegressionTest.WarmStartComplementarityGapIncludesL1SoftPair`, `tests/test_barrier_residual_contract.cpp` | `covered` |
| `WARM-007` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcCallbackDoesNotSkipDirectionSolve` | `covered` |

## Open Gaps

- No open P1 evidence gaps. External user-side shifting strategies remain out
  of scope for this solver-core contract.
