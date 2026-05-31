# Explicit Integrator Contract

Status: draft

Owner modules:

- `MOD-MODEL-CODEGEN`
- `MOD-ALG-EVAL`

Related modules:

- `MOD-INTEGRATOR`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Explicit/discrete dynamics packet generation and dispatch for Euler, RK2, RK4,
and direct discrete dynamics.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `INT-001` | Explicit integrator choice determines generated/evaluated discrete dynamics packet. | `partial` |
| `INT-002` | Generated explicit dynamics writes `f_resid`, `A`, and `B` consistently. | `partial` |
| `INT-003` | Direct discrete dynamics bypasses continuous integration formulas. | `partial` |
| `INT-004` | Generated integrator metadata records the integrator used for generated packets. | `partial` |
| `INT-005` | Explicit integrator packet behavior must be covered by generated asset or MiniModel tests. | `partial` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Unsupported explicit integrator pattern in MiniModel | Python codegen error. |
| Non-finite generated dynamics packet | Solver numeric boundary classifies. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `INT-001` | `tests/minimodel/test_dynamics_dsl.py`, `tests/test_integrator.cpp` | `partial` |
| `INT-002` | generated asset tests | `partial` |
| `INT-003` | `tests/minimodel/test_dynamics_dsl.py` | `partial` |
| `INT-004` | `tests/test_implicit_sparse_riccati.cpp`, generated header tests | `partial` |
| `INT-005` | `tests/test_asset_regressions.cpp` | `partial` |

## Open Gaps

- Need explicit docs for generated explicit integrator formulas.
