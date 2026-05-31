# Generated Packets Contract

Status: draft

Owner modules:

- `MOD-MODEL-CODEGEN`
- `MOD-ALG-EVAL`

Related modules:

- `MOD-MATRIX`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Generated cost, dynamics, constraint, soft-weight, and fused Riccati packet
ownership, including clear vs full-overwrite rules.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `CODEGEN-020` | Generated functions own the packets they write and must leave them internally consistent. | `partial` |
| `CODEGEN-021` | A packet that is provably fully overwritten should not emit redundant clears. | `partial` |
| `CODEGEN-022` | A packet that is not fully overwritten must be cleared or initialized before sparse assignments. | `partial` |
| `CODEGEN-023` | L1 and L2 soft weight updates may be emitted separately when that avoids redundant zero/reset work. | `partial` |
| `CODEGEN-024` | Generated terminal packets must not rely on terminal control values unless explicitly projected. | `partial` |
| `CODEGEN-025` | Generated fused Riccati packets must be guarded by integrator compatibility metadata. | `partial` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Symbolic expressions, sparsity patterns, generation options |
| Outputs | C++ packet assignment code and metadata |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Not fully assigned packet without clear | Codegen/test failure. |
| Unsupported terminal control dependency | Warning or generated projection according to MiniModel policy. |
| Incompatible fused integrator | Solver avoids fused path. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `CODEGEN-020` | `tests/test_asset_regressions.cpp` | `partial` |
| `CODEGEN-021` | `tests/minimodel/*.py`, benchmark codegen inspections | `partial` |
| `CODEGEN-022` | `tests/minimodel/*.py` | `partial` |
| `CODEGEN-023` | `tests/minimodel/test_constraints.py`, benchmark generated models | `partial` |
| `CODEGEN-024` | `tests/minimodel/test_terminal.py` | `partial` |
| `CODEGEN-025` | `tests/test_implicit_sparse_riccati.cpp` | `partial` |

## Open Gaps

- Need explicit generated header diff/shape tests for clear elision.
