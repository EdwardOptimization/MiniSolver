# Termination Contract

Status: draft

Owner modules:

- `MOD-ALG-TERM`
- `MOD-SOLVER-ROUTE`

Related modules:

- `MOD-ALG-BARRIER`
- `MOD-ALG-EVAL`
- `MOD-CORE-TYPES`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Strict KKT convergence predicate.
- `ACCEPTABLE_NMPC` primal-feasible shortcuts.
- `RTI_FIXED_ITERATION` loop-budget behavior.
- Cost and residual stagnation.
- Tiny-step stagnation classification.

Out of scope:

- Final public status classification after postsolve.
- Scale-aware physical-unit acceptance policy.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| `TerminationSnapshot` | Solver route | Freshness depends on caller phase. |
| Residual history | Solver context | Updated by solve loop. |
| Cost and `mu` history | Solver context/barrier update | Used only for stagnation. |
| `SolverConfig` | Config | Tolerances and profile are validated. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Convergence boolean | Strict KKT candidate. | Solver route/postsolve |
| Feasible boolean/status | Primal-acceptable candidate. | Solver route/postsolve |
| Stagnation decision | Loop-level insufficient progress. | Solver route |
| Fixed-iteration predicate | Budget-mode loop exit. | Solver route |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `ResidualStagnationMonitor` | `MOD-ALG-TERM` | Per solve. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `TERM-001` | Strict convergence requires `linear_ok`, `primal_inf <= tol_con`, `dual_inf <= tol_dual`, and `complementarity_inf <= tol_mu`. | `covered` |
| `TERM-002` | Strict convergence must not use `mu` alone as an optimality certificate. | `covered` |
| `TERM-003` | `ACCEPTABLE_NMPC` may produce primal-only `FEASIBLE` exits but must not claim stale strict KKT convergence. | `covered` |
| `TERM-004` | Warm-start zero-step primal shortcut is disabled when a model-update callback is installed. | `covered` |
| `TERM-005` | Accepted-step primal shortcut must refresh primal residual before returning feasible. | `covered` |
| `TERM-006` | `RTI_FIXED_ITERATION` is a loop-budget policy and cannot mask fatal failures. | `covered` |
| `TERM-007` | Residual stagnation returns loop-level insufficient progress and leaves final quality to postsolve. | `covered` |
| `TERM-008` | Cost stagnation requires feasible-enough primal residual and insufficient cost progress. | `covered` |
| `TERM-009` | Tiny-step stagnation may return `FEASIBLE` when primal residual is within `tol_con`, but must not independently claim `OPTIMAL`. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Residual stagnation | `INSUFFICIENT_PROGRESS` / `RESIDUAL_STAGNATION` | Loop-level only. |
| Cost stagnation | `INSUFFICIENT_PROGRESS` / `COST_STAGNATION` | Loop-level only. |
| Max iteration budget | `MAX_ITER` / `MAX_ITERATIONS` | Unless fixed-iteration profile applies. |
| Fixed iteration reached | `UNSOLVED` or current quality / `FIXED_ITERATION` before postsolve | Fatal failures have precedence. |
| Tiny step not primal feasible | `STEP_TOO_SMALL` or unresolved loop failure | Solver route owns exact projection. |

## Numeric And Performance Constraints

- Termination predicates must be lightweight and allocation-free.
- Expensive residual refresh is owned by solver/postsolve, not by tiny predicate
  helpers.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `TERM-001` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesTrueComplementarityGapSnapshot`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveProvesStrictOptimalWhenFreshKktResidualsPass` | `covered` |
| `TERM-002` | `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceUsesKktComplementarityNotBarrierTarget`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.ConvergenceRejectsLargeTrueComplementarityAtMuFinal`, `docs/architecture/termination-design.md` | `covered` |
| `TERM-003` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcPrimalFeasibleSkipsDirectionFailure`, `tests/test_termination.cpp::TerminationTest.AcceptableNmpcInvalidReuseGuessDoesNotSkipDirectionSolve`, `tests/test_barrier_residual_contract.cpp::BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal` | `covered` |
| `TERM-004` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcCallbackDoesNotSkipDirectionSolve` | `covered` |
| `TERM-005` | `tests/test_termination.cpp::TerminationTest.AcceptableNmpcAcceptedStepRefreshesPrimalResidual` | `covered` |
| `TERM-006` | `tests/test_termination.cpp::TerminationTest.RtiFixedIterationDoesNotMaskLinearSolveFailure`, `tests/test_status.cpp::StatusTest.RtiFixedIterationProfileStopsAfterOneIteration` | `covered` |
| `TERM-007` | `tests/test_termination.cpp::TerminationTest.ResidualStagnationMonitorRequiresConfiguredWindow`, `tests/test_termination.cpp::TerminationTest.ResidualStagnationMonitorHonorsMinIterations`, `tests/test_termination.cpp::TerminationTest.ResidualStagnationMonitorResetsOnFeasibleMuDecrease`, `tests/test_termination.cpp::TerminationTest.ResidualStagnationLoopStatusWaitsForPostsolveFeasibleQuality` | `covered` |
| `TERM-008` | `tests/test_features.cpp::FeaturesTest.CostStagnationTermination`, `tests/test_features.cpp::FeaturesTest.CostStagnationSkipsModelUpdateCallbacks`, `tests/test_bugfixes.cpp::BugfixTest.CostStagnationNotGatedOnMuFinal` | `covered` |
| `TERM-009` | `tests/test_termination.cpp::TerminationTest.TinyStepStagnationDoesNotClaimStrictOptimality` | `covered` |

## Open Gaps

- Residual-stagnation default tuning still belongs to replay/benchmark work.
