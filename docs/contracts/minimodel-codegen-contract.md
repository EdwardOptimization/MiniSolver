# MiniModel Codegen Contract

Status: draft

Owner modules:

- `MOD-MODEL-CODEGEN`

Related modules:

- `MOD-CORE-SEMANTICS`
- `MOD-ALG-EVAL`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Python DSL validation, symbolic ownership, generated C++ model shape, and
MiniModel-owned modeling semantics.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `CODEGEN-001` | State, control, and parameter identifiers must be valid, unique C++ identifiers. | `covered` |
| `CODEGEN-002` | Dynamics equations must match supported `Dot`/`Next` or discrete patterns. | `covered` |
| `CODEGEN-003` | Residual weights must be numeric/symbolic expressions that do not require unsupported solver-core semantics. | `covered` |
| `CODEGEN-004` | Soft constraint weight expressions may depend on parameters but not state/control decision variables. | `covered` |
| `CODEGEN-005` | MiniModel owns modeling semantics such as loss type, structural soft flags, and generated updater shape. | `covered` |
| `CODEGEN-006` | Core solver must not add extra modeling-policy validation unless it protects a generic solver invariant. | `covered` |
| `CODEGEN-007` | Generated metadata must include aggregate soft flags when row soft flags are emitted. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Python symbolic model declarations and generation options |
| Outputs | Generated C++ header, metadata, packets, fingerprint |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Invalid DSL usage | Python exception before C++ generation. |
| Unsupported symbolic dependency | Python exception. |
| Generated runtime non-finite packet | Solver numeric boundary classifies. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `CODEGEN-001` | `tests/minimodel/test_identifiers.py`, `tests/minimodel/test_model_safety.py::test_model_name_validation_rejects_invalid_cpp_type_names` | `covered` |
| `CODEGEN-002` | `tests/minimodel/test_dynamics_dsl.py`, `tests/minimodel/test_implicit_patterns.py` | `covered` |
| `CODEGEN-003` | `tests/minimodel/test_residual_costs.py` | `covered` |
| `CODEGEN-004` | `tests/minimodel/test_model_safety.py::test_soft_constraint_weight_validation_is_explicit`, `tests/minimodel/test_constraints.py` | `covered` |
| `CODEGEN-005` | `tests/minimodel/test_constraints.py`, `tests/test_soft_constraints.cpp` | `covered` |
| `CODEGEN-006` | `tests/minimodel/test_model_safety.py::test_soft_constraint_weight_validation_is_explicit`, `docs/architecture/solver-development-principles.md` | `covered` |
| `CODEGEN-007` | `tests/minimodel/test_constraints.py::test_soft_constraint_parameter_weight_packet_updates_knot` | `covered` |

## Open Gaps

- No open P1 evidence gaps. New generated packet categories should add shape
  tests with the feature.
