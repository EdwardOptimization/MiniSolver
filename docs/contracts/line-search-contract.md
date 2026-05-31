# Line Search Contract

Status: draft

Owner modules:

- `MOD-ALG-LS`
- `MOD-SOLVER-LSUTIL`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-SOLVER-ROUTE`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Candidate construction, fraction-to-boundary alpha ownership, accepted-step
refresh, no-line-search mode, merit/filter shared behavior, and line-search
failure status handoff.

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Active trajectory, search directions, `mu`, `reg`, `dt_traj`, `SolverConfig` |
| Outputs | Accepted trajectory swap, `LineSearchResult`, SOC flags |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `LS-001` | Line search candidates are derived from the active trajectory plus alpha-scaled directions. | `covered` |
| `LS-002` | Fraction-to-boundary protects `s`, `lam`, and L1/mixed `soft_s`/implicit soft dual. | `covered` |
| `LS-003` | Accepted candidates must refresh model packets before becoming the active iterate. | `covered` |
| `LS-004` | No-line-search mode still applies fraction-to-boundary and candidate refresh. | `covered` |
| `LS-005` | Rollout mode keeps `x0` fixed, updates controls/slacks/duals, and re-integrates states. | `covered` |
| `LS-006` | Tiny alpha or exhausted backtracking returns a structured line-search result for solver classification. | `covered` |
| `LS-007` | Line search must not allocate on zero-allocation solve paths. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Alpha collapses below usable threshold | Return failed/tiny `LineSearchResult`; solver route classifies. |
| Non-finite line-search scalar | Return numerical failure. |
| Accepted but only primal feasible | Solver route/postsolve classifies final quality. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `LS-001` | `tests/test_line_search.cpp::LineSearchTest.FilterAcceptance`, `tests/test_line_search.cpp::LineSearchTest.MeritArmijoDoesNotBuildFiniteDifferenceProbe`, `tests/test_line_search.cpp::LineSearchTest.NoLineSearchRefreshesAcceptedPointEvaluations` | `covered` |
| `LS-002` | `tests/test_line_search.cpp::LineSearchTest.FractionToBoundaryProtectsMixedSoftInterior`, `tests/test_soft_constraints.cpp` | `covered` |
| `LS-003` | `tests/test_line_search.cpp::LineSearchTest.FilterAcceptanceUsesTrueResidualNotQpResidual`, `tests/test_line_search.cpp::LineSearchTest.MeritAcceptanceUsesTrueResidualNotQpResidual`, `tests/test_line_search.cpp::LineSearchTest.NoLineSearchRefreshesAcceptedPointEvaluations` | `covered` |
| `LS-004` | `tests/test_line_search.cpp::LineSearchTest.NoLineSearchAppliesFractionToBoundary`, `tests/test_line_search.cpp::LineSearchTest.NoLineSearchRefreshesAcceptedPointEvaluations` | `covered` |
| `LS-005` | `tests/test_line_search.cpp::LineSearchTest.MeritRolloutProducesConsistentStates`, `tests/test_line_search.cpp::LineSearchTest.NoLineSearchRolloutProducesConsistentStates`, `tests/test_line_search.cpp::LineSearchTest.FilterSocSkippedInRolloutMode` | `covered` |
| `LS-006` | `tests/test_line_search.cpp::LineSearchTest.NoLineSearchTinyAlphaReturnsZeroStepResult`, `tests/test_line_search.cpp::LineSearchTest.FilterRejectsTrialAboveThetaMax`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteDphiPropagatesToSolverStatus` | `covered` |
| `LS-007` | `tests/test_memory.cpp::MemoryTest.ZeroMalloc_FilterSOC_Path` | `covered` |

## Open Gaps

- No open P0 line-search unit gaps.
