# Parameter Contract

Status: draft

Owner modules:

- `MOD-SOLVER-ROUTE`
- `MOD-MODEL-CODEGEN`

Related modules:

- `MOD-ALG-EVAL`
- `MOD-CORE-TYPES`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Per-knot parameter storage, name lookup, generated parameter usage, and parameter
trust boundary.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `PARAM-001` | Parameters are stored per knot in `KnotState::p`. | `covered` |
| `PARAM-002` | Parameter setters validate horizon/stage/name/index/size and value finiteness before mutation. | `covered` |
| `PARAM-003` | Generated model code may use parameters in cost, constraints, dynamics, and soft weights. | `covered` |
| `PARAM-004` | Solver core treats parameterized modeling semantics as model/codegen-owned. | `covered` |
| `PARAM-005` | Parameter mutations through callbacks affect subsequent model evaluation before solver decisions use packets. | `covered` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Public parameter setters, callbacks, generated expressions |
| Outputs | Updated per-knot `p`, generated packet values |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Unknown parameter name | `ApiStatus::UnknownName`. |
| Invalid stage/index/size | Matching `ApiStatus` rejection. |
| Non-finite value passed to a public parameter setter | `ApiStatus::NonFiniteValue`; stored parameters are not mutated. |
| Finite parameter value produces a non-finite generated model packet | Numeric boundary classifies solve failure. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `PARAM-001` | `tests/test_config_regressions.cpp::ConfigRegressionTest.CheckedScalarGettersReportInvalidAccess`, `tests/test_solver_snapshot.cpp` | `covered` |
| `PARAM-002` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ApiSettersReturnExplicitStatusAndDoNotMutate` | `covered` |
| `PARAM-003` | `tests/minimodel/test_residual_costs.py::test_add_residual_accepts_parameter_vector_weight`, `tests/minimodel/test_residual_costs.py::test_add_residual_accepts_parameter_reference`, `tests/minimodel/test_constraints.py::test_soft_constraint_parameter_weight_packet_updates_knot`, `tests/minimodel/test_constraints.py::test_generated_soc_refreshes_parameterized_soft_weights` | `covered` |
| `PARAM-004` | `tests/minimodel/test_model_safety.py::test_soft_constraint_weight_validation_is_explicit`, `docs/architecture/solver-development-principles.md` | `covered` |
| `PARAM-005` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforeFirstEvaluation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackRunsBeforePresolveSlackInitialization` | `covered` |

## Open Gaps

- No open P1 evidence gaps. Future parameter-owned packet optimizations should
  add matching overwrite/update tests with the behavior change.
