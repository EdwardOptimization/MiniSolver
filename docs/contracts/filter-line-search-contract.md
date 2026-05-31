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
| `FILTER-001` | Filter entries store comparable infeasibility/objective measures for the current barrier regime. | `partial` |
| `FILTER-002` | Barrier decreases reset filter history when old entries are no longer comparable. | `partial` |
| `FILTER-003` | Switching condition decides whether filter or Armijo-like objective decrease is required. | `partial` |
| `FILTER-004` | `theta` and `phi` values must be finite before filter acceptance uses them. | `partial` |
| `FILTER-005` | Accepted filter candidates must refresh model packets before trajectory swap. | `partial` |
| `FILTER-006` | Exhausted filter backtracking returns a structured failure for solver classification. | `partial` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Active/candidate trajectory, filter entries, `mu`, config |
| Outputs | Accepted alpha, updated filter history, failure/numerical status |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Non-finite `theta` or `phi` | `NUMERICAL_ERROR` path. |
| Candidate dominated and no acceptable alpha | Line-search failure path. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `FILTER-001` | `tests/test_line_search.cpp` | `partial` |
| `FILTER-002` | `tests/test_line_search.cpp` | `partial` |
| `FILTER-003` | `tests/test_line_search.cpp` | `partial` |
| `FILTER-004` | `tests/test_line_search.cpp` | `partial` |
| `FILTER-005` | `tests/test_line_search.cpp` | `partial` |
| `FILTER-006` | `tests/test_line_search.cpp` | `partial` |

## Open Gaps

- Need benchmark/replay evidence for filter behavior on difficult nonlinear
  cases.
