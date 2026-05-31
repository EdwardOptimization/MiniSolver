# Numerical Jacobian Contract

Status: draft

Owner modules:

- `MOD-INTEGRATOR`
- `MOD-MATRIX`

Related modules:

- `MOD-MODEL-CODEGEN`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Finite-difference continuous Jacobian fallback used by implicit integration.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `INT-040` | Numerical Jacobian uses centered finite differences. | `partial` |
| `INT-041` | Perturbation size scales with `max(1, abs(value))`. | `partial` |
| `INT-042` | State and control perturbations restore the original value after each column. | `partial` |
| `INT-043` | Numerical Jacobian fallback requires model continuous dynamics. | `partial` |
| `INT-044` | Numerical Jacobian is a fallback, not preferred over analytical continuous Jacobians. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Continuous dynamics returns non-finite values | Downstream implicit packet becomes non-finite and solver numeric boundary classifies. |
| Missing continuous dynamics | Compile failure for fallback path. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `INT-040` | `tests/test_integrator.cpp` | `partial` |
| `INT-041` | `tests/test_integrator.cpp` | `partial` |
| `INT-042` | `tests/test_integrator.cpp` | `partial` |
| `INT-043` | build tests using fallback | `partial` |
| `INT-044` | `tests/test_integrator.cpp` | `partial` |

## Open Gaps

- Need direct tolerance/accuracy tests if fallback becomes performance-critical.
