# Implicit Integrator Contract

Status: draft

Owner modules:

- `MOD-INTEGRATOR`
- `MOD-ALG-EVAL`

Related modules:

- `MOD-MODEL-CODEGEN`
- `MOD-MATRIX`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Implicit Euler, implicit midpoint, Gauss-Legendre two-stage integration, Newton
solve behavior, Jacobian inversion, and invalid dynamics marking.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `INT-020` | Implicit integrator only accepts implicit integrator types. | `partial` |
| `INT-021` | Newton solve uses analytical continuous Jacobians when available and numerical Jacobians otherwise. | `partial` |
| `INT-022` | Failed Newton solve marks dynamics and Jacobian packets invalid. | `partial` |
| `INT-023` | Failed Jacobian inversion marks Jacobian packets invalid. | `partial` |
| `INT-024` | Implicit integrator writes `f_resid`, `A`, and `B` for Riccati/model evaluation. | `partial` |
| `INT-025` | Newton regularization is a fallback for singular Jacobians, not unconditional damping. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Newton failure | Mark packet NaN; solver numeric boundary handles solve failure. |
| Unsupported type | Throw invalid argument for internal misuse. |
| Singular Jacobian with no successful damping | Return failure/mark invalid. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `INT-020` | `tests/test_integrator.cpp` | `partial` |
| `INT-021` | `tests/test_integrator.cpp` | `partial` |
| `INT-022` | `tests/test_integrator.cpp` | `partial` |
| `INT-023` | `tests/test_integrator.cpp` | `partial` |
| `INT-024` | `tests/test_integrator.cpp`, `tests/test_implicit_sparse_riccati.cpp` | `partial` |
| `INT-025` | `tests/test_integrator.cpp` | `partial` |

## Open Gaps

- Need replay evidence for stiff implicit models if they become release-critical.
