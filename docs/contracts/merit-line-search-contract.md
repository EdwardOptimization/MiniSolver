# Merit Line Search Contract

Status: draft

Owner modules:

- `MOD-ALG-LS`

Related modules:

- `MOD-CORE-SEMANTICS`
- `MOD-ALG-EVAL`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Merit value, directional derivative, Armijo sufficient decrease, soft penalties,
barrier terms, and non-finite scalar handling for merit globalization.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `MERIT-001` | Merit value includes objective, barrier terms, dynamics defects, and constraint residual penalties. | `covered` |
| `MERIT-002` | Merit directional derivative uses current cost gradients, directions, barrier terms, and constraint directions. | `covered` |
| `MERIT-003` | Armijo acceptance must use finite `phi` and finite `dphi`. | `covered` |
| `MERIT-004` | L1 and mixed soft rows contribute L1/mixed penalty and soft barrier terms through shared semantics. | `covered` |
| `MERIT-005` | L2 and zero-weight relaxed rows use effective L2 weight consistently. | `covered` |
| `MERIT-006` | Non-finite merit value or derivative produces numerical failure, not silent rejection. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Active/candidate trajectory, directions, `mu`, config, merit penalty |
| Outputs | Acceptance/rejection, alpha, numerical failure status |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| `phi` or `dphi` non-finite | `NUMERICAL_ERROR` path. |
| Armijo fails for all backtracking steps | Line-search failure path. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `MERIT-001` | `tests/test_line_search.cpp::LineSearchTest.MeritFunctionBacktracking`, `tests/test_line_search.cpp::LineSearchTest.MeritRolloutProducesConsistentStates`, `tests/test_line_search.cpp::LineSearchTest.MeritAcceptanceUsesTrueResidualNotQpResidual` | `covered` |
| `MERIT-002` | `tests/test_line_search.cpp::LineSearchTest.MeritArmijoDoesNotBuildFiniteDifferenceProbe`, `tests/test_line_search.cpp::LineSearchTest.MeritAcceptanceUsesTrueResidualNotQpResidual` | `covered` |
| `MERIT-003` | `tests/test_line_search.cpp::LineSearchTest.MeritArmijoDoesNotBuildFiniteDifferenceProbe`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteInitialPhiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteTrialPhiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteDphiReturnsNumericalError` | `covered` |
| `MERIT-004` | `tests/test_soft_constraints.cpp::ComparisonTest.MixedL1L2SoftConstraintWithMeritLineSearch`, `tests/test_line_search.cpp` merit barrier/Armijo tests | `covered` |
| `MERIT-005` | `tests/test_soft_constraints.cpp::ComparisonTest.L2SoftConstraintWithMeritLineSearch`, `tests/test_soft_constraints.cpp::SoftConstraintTest.ZeroL2WeightInitializesAsRegularizedSoftRow`, `tests/test_soft_constraints.cpp::SoftConstraintTest.TinyL2WeightUsesEffectiveFloor` | `covered` |
| `MERIT-006` | `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteDphiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteInitialPhiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteTrialPhiReturnsNumericalError`, `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteDphiPropagatesToSolverStatus` | `covered` |

## Open Gaps

- No open P0 merit line-search coverage gaps. Difficult globalization
  performance/tuning remains replay and benchmark evidence, not unit coverage.
