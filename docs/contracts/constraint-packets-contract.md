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
| `CON-020` | QP constraint packets provide the residual/Jacobian used by the linearized KKT system. | `covered` |
| `CON-021` | True constraint packets provide the residual used by internal primal metrics when present. | `covered` |
| `CON-022` | Terminal constraint packets are selected at the terminal knot when provided. | `covered` |
| `CON-023` | SOC constraint packets evaluate trial nonlinear residuals for second-order correction. | `covered` |
| `CON-024` | Missing optional true constraints fall back to QP residuals. | `covered` |
| `CON-025` | Generated packets should clear only fields not provably fully overwritten. | `covered` |

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
| `CON-020` | `tests/minimodel/test_constraints.py::test_quad_boundary_projection_splits_qp_and_true_residuals`, `tests/test_asset_regressions.cpp`, `tests/test_solver_quality.cpp` | `covered` |
| `CON-021` | `tests/test_line_search.cpp::LineSearchTest.FilterAcceptanceUsesTrueResidualNotQpResidual`, `tests/test_scaling_regressions.cpp` | `covered` |
| `CON-022` | `tests/minimodel/test_terminal.py::test_generated_terminal_stage_uses_x_only_projection`, `tests/minimodel/test_constraints.py::test_stage_only_constraint_zeros_terminal_row` | `covered` |
| `CON-023` | `tests/test_line_search.cpp::LineSearchTest.FilterSocUsesModelSocConstraintOverride`, `tests/test_replay_corpus.cpp::ReplayCorpusTest.SocNonlinearObstaclePathAttemptsAndAcceptsCorrection` | `covered` |
| `CON-024` | `tests/test_solver_quality.cpp`, `tests/test_line_search.cpp::LineSearchTest.FilterAcceptanceUsesTrueResidualNotQpResidual` | `covered` |
| `CON-025` | `tests/minimodel/test_residual_costs.py::test_sparse_generated_packets_zero_first_then_assign_nonzero`, `tests/minimodel/test_residual_costs.py::test_full_generated_packets_skip_clear`, `tests/minimodel/test_constraints.py::test_l1_only_soft_weight_update_does_not_clear_l2_packet` | `covered` |

## Open Gaps

- No open P1 evidence gaps. Future packet ownership optimizations should add
  new generated shape tests with the codegen change.
