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
| `PARAM-001` | Parameters are stored per knot in `KnotState::p`. | `partial` |
| `PARAM-002` | Parameter setters validate horizon/stage/name/index/size and value finiteness before mutation. | `partial` |
| `PARAM-003` | Generated model code may use parameters in cost, constraints, dynamics, and soft weights. | `partial` |
| `PARAM-004` | Solver core treats parameterized modeling semantics as model/codegen-owned. | `partial` |
| `PARAM-005` | Parameter mutations through callbacks affect subsequent model evaluation before solver decisions use packets. | `partial` |

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
| `PARAM-001` | solver setter tests | `partial` |
| `PARAM-002` | `tests/test_solver.cpp`, `tests/test_config_regressions.cpp` | `partial` |
| `PARAM-003` | `tests/minimodel/test_constraints.py`, generated C++ tests | `partial` |
| `PARAM-004` | design docs and code review | `partial` |
| `PARAM-005` | callback tests/examples | `partial` |

## Open Gaps

- Broaden parameterized generated-model solve regressions before changing
  parameter/codegen semantics. Current soft-weight paths have focused coverage;
  future parameter-owned packets should get matching overwrite/update tests when
  their ownership is optimized.
