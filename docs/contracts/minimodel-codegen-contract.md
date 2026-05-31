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
| `CODEGEN-001` | State, control, and parameter identifiers must be valid, unique C++ identifiers. | `partial` |
| `CODEGEN-002` | Dynamics equations must match supported `Dot`/`Next` or discrete patterns. | `partial` |
| `CODEGEN-003` | Residual weights must be numeric/symbolic expressions that do not require unsupported solver-core semantics. | `partial` |
| `CODEGEN-004` | Soft constraint weight expressions may depend on parameters but not state/control decision variables. | `partial` |
| `CODEGEN-005` | MiniModel owns modeling semantics such as loss type, structural soft flags, and generated updater shape. | `partial` |
| `CODEGEN-006` | Core solver must not add extra modeling-policy validation unless it protects a generic solver invariant. | `partial` |
| `CODEGEN-007` | Generated metadata must include aggregate soft flags when row soft flags are emitted. | `partial` |

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
| `CODEGEN-001` | `tests/minimodel/test_identifiers.py` | `partial` |
| `CODEGEN-002` | `tests/minimodel/test_dynamics_dsl.py` | `partial` |
| `CODEGEN-003` | `tests/minimodel/test_residual_costs.py` | `partial` |
| `CODEGEN-004` | `tests/minimodel/test_constraints.py` | `partial` |
| `CODEGEN-005` | `tests/minimodel/test_constraints.py`, `tests/test_soft_constraints.cpp` | `partial` |
| `CODEGEN-006` | `docs/architecture/solver-development-principles.md` | `partial` |
| `CODEGEN-007` | `tests/minimodel/test_constraints.py` | `partial` |

## Open Gaps

- Need generated-code shape snapshots for all packet categories.
