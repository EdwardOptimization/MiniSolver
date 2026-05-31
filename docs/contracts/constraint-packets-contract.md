# Constraint Packets Contract

Status: draft

Owner modules:

- `MOD-ALG-EVAL`
- `MOD-MODEL-CODEGEN`

Related modules:

- `MOD-CORE-SEMANTICS`
- `MOD-SOLVER-RICCATI`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Stage/terminal QP constraints, true constraints, SOC constraints, Jacobian
packets, generated packet overwrite/clear behavior, and fallback behavior when
optional packets are missing.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `CON-020` | QP constraint packets provide the residual/Jacobian used by the linearized KKT system. | `partial` |
| `CON-021` | True constraint packets provide the residual used by internal primal metrics when present. | `partial` |
| `CON-022` | Terminal constraint packets are selected at the terminal knot when provided. | `partial` |
| `CON-023` | SOC constraint packets evaluate trial nonlinear residuals for second-order correction. | `partial` |
| `CON-024` | Missing optional true constraints fall back to QP residuals. | `partial` |
| `CON-025` | Generated packets should clear only fields not provably fully overwritten. | `partial` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Model state/control/parameters, terminal flag, generated functions |
| Outputs | `g_val`, `g_true`, `g_unscaled`, `C`, `D`, SOC residual packets |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Missing optional packet | Use documented fallback. |
| Non-finite packet | Surface at numeric boundary. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `CON-020` | `tests/test_asset_regressions.cpp`, `tests/test_solver_quality.cpp` | `partial` |
| `CON-021` | `tests/test_scaling_regressions.cpp` | `partial` |
| `CON-022` | `tests/minimodel/test_terminal.py`, C++ terminal tests | `partial` |
| `CON-023` | `tests/test_line_search.cpp`, `tests/test_soft_constraints.cpp` | `partial` |
| `CON-024` | `tests/test_solver_quality.cpp` | `partial` |
| `CON-025` | `tests/minimodel/*.py`, generated asset tests | `partial` |

## Open Gaps

- Need explicit generated packet full-overwrite regression matrix.
