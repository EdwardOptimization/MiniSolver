# Filter Line Search Contract

Status: draft

Owner modules:

- `MOD-ALG-LS`

Related modules:

- `MOD-ALG-BARRIER`
- `MOD-ALG-EVAL`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Filter history, switching condition, `theta`/`phi` semantics, barrier-update
reset, acceptance/rejection, and numerical failure handling.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `FILTER-001` | Filter entries store comparable infeasibility/objective measures for the current barrier regime. | `covered` |
| `FILTER-002` | Barrier decreases reset filter history when old entries are no longer comparable. | `covered` |
| `FILTER-003` | Switching condition decides whether filter or Armijo-like objective decrease is required. | `covered` |
| `FILTER-004` | `theta`, `phi`, and switching directional derivative values must be finite before filter acceptance uses them. | `covered` |
| `FILTER-005` | Accepted filter candidates must refresh model packets before trajectory swap. | `covered` |
| `FILTER-006` | Exhausted filter backtracking returns a structured failure for solver classification. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Active/candidate trajectory, filter entries, `mu`, config |
| Outputs | Accepted alpha, updated filter history, failure/numerical status |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Non-finite `theta`, `phi`, or switching derivative | `NUMERICAL_ERROR` path. |
| Candidate dominated and no acceptable alpha | Line-search failure path. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `FILTER-001` | `tests/test_line_search.cpp::LineSearchTest.FilterHistoryWrapsAtFixedCapacity`, `tests/test_line_search.cpp::LineSearchTest.FilterHTypeAcceptanceStillAugmentsFilter` | `covered` |
| `FILTER-002` | `tests/test_line_search.cpp::LineSearchTest.FilterBarrierUpdateClearsHistory` | `covered` |
| `FILTER-003` | `tests/test_line_search.cpp::LineSearchTest.FilterFTypeUsesArmijoAndDoesNotAugmentFilter`, `tests/test_line_search.cpp::LineSearchTest.FilterHTypeAcceptanceStillAugmentsFilter` | `covered` |
| `FILTER-004` | `tests/test_line_search.cpp::LineSearchTest.FilterNonFiniteInitialMetricsReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.FilterNonFiniteDphiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.FilterNonFiniteTrialMetricsReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.FilterSocNonFiniteMetricsReturnsNumericalError` | `covered` |
| `FILTER-005` | `tests/test_line_search.cpp::LineSearchTest.FilterAcceptanceUsesTrueResidualNotQpResidual`, `tests/test_line_search.cpp::LineSearchTest.FilterRejectsTrialAboveThetaMax` | `covered` |
| `FILTER-006` | `tests/test_line_search.cpp::LineSearchTest.FilterRejectsTrialAboveThetaMax` | `covered` |

## Open Gaps

- Difficult nonlinear-case benchmark/replay evidence remains useful before
  tuning default filter policy, but the current P1 contract semantics have
  focused tests.
