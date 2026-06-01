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

Explicit/discrete dynamics packet generation and dispatch for explicit Euler,
RK2 explicit midpoint, classic RK4, and direct discrete dynamics.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `INT-001` | Explicit integrator choice determines generated/evaluated discrete dynamics packet. | `covered` |
| `INT-002` | Generated explicit dynamics writes `f_resid`, `A`, and `B` consistently. | `covered` |
| `INT-003` | Direct discrete dynamics bypasses continuous integration formulas. | `covered` |
| `INT-004` | Generated integrator metadata records the integrator used for generated packets. | `covered` |
| `INT-005` | Explicit integrator packet behavior must be covered by generated asset or MiniModel tests. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Unsupported explicit integrator pattern in MiniModel | Python codegen error. |
| Non-finite generated dynamics packet | Solver numeric boundary classifies. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `INT-001` | `tests/minimodel/test_dynamics_dsl.py`, `tests/test_integrator.cpp::IntegratorTest.AccuracyComparison` | `covered` |
| `INT-002` | `tests/minimodel/test_dynamics_dsl.py::test_dot_subject_to_generates_continuous_dynamics`, `tests/test_asset_regressions.cpp` | `covered` |
| `INT-003` | `tests/minimodel/test_dynamics_dsl.py::test_next_subject_to_generates_discrete_dynamics_map`, `tests/minimodel/test_dynamics_dsl.py::test_next_model_rejects_non_discrete_runtime_integrators` | `covered` |
| `INT-004` | `tests/minimodel/test_dynamics_dsl.py::test_generate_rejects_integrator_mode_mismatch`, `tests/minimodel/test_dynamics_dsl.py::test_model_fingerprint_changes_between_dot_and_next_modes`, `tests/test_implicit_sparse_riccati.cpp` | `covered` |
| `INT-005` | `tests/test_asset_regressions.cpp`, `tests/minimodel/test_dynamics_dsl.py` | `covered` |

## Open Gaps

- No open P1 evidence gaps. Formula-level documentation can still be expanded
  before release-critical integrator claims.
