# Config API Contract

Status: draft

Owner modules:

- `MOD-CORE-CONFIG`
- `MOD-SOLVER-ROUTE`
- `MOD-RUNTIME`

Related modules:

- `MOD-DEBUG-SNAPSHOT`
- `MOD-CORE-TYPES`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

In scope:

- Config validation.
- `set_config()` mutation rules.
- Backend preservation.
- Snapshot config registry dependency.
- Callback structural mutation restriction for config and topology mutators.

Out of scope:

- New public config fields.
- User-facing presets.

## Inputs

| Input | Owner | Validity assumptions |
| --- | --- | --- |
| Constructor backend | User | Explicit source of backend truth. |
| `SolverConfig` | User/snapshot | May be invalid until validated. |
| Snapshot config fields | Snapshot I/O | Must match config registry. |

## Outputs

| Output | Meaning | Consumer |
| --- | --- | --- |
| Accepted `SolverConfig` | New solver configuration. | Solver route |
| `ApiStatus` | Rejection reason. | User/snapshot |
| Dirty build state | Components must rebuild before use. | Solver route |

## Owned State

| State | Owner | Lifetime |
| --- | --- | --- |
| `SolverConfig` | `MOD-SOLVER-ROUTE` | Solver lifetime. |
| Config field registry | `MOD-CORE-CONFIG` | Compile-time. |

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `SOLVE/API-001` | Constructor rejects invalid config before solver execution. | `covered` |
| `SOLVE/API-002` | `set_config()` validates candidate config before mutating current config. | `covered` |
| `SOLVE/API-003` | `set_config()` preserves the constructor backend unless backend mutation is explicitly supported. | `covered` |
| `SOLVE/API-004` | Accepted config changes mark solver components dirty for rebuild. | `covered` |
| `SOLVE/API-005` | Public structural mutators are rejected while model-update callback is executing. | `covered` |
| `SOLVE/API-006` | Config enum validation rejects invalid raw enum values. | `covered` |
| `SOLVE/API-007` | Snapshot config serialization uses the central field registry. | `covered` |

## Failure Semantics

| Failure | Required status/reason | Notes |
| --- | --- | --- |
| Invalid enum/range | `ApiStatus::InvalidArgument` | No partial mutation. |
| Non-finite config value | `ApiStatus::NonFiniteValue` | Config boundary. |
| Constructor invalid config | `std::invalid_argument` | Construction failure. |
| Callback structural mutation | `ApiStatus::InvalidArgument` or callback contract violation | Protects solve route. |

## Numeric And Performance Constraints

- Config validation is not a hot loop.
- Config-derived strategy resolution should happen at rebuild boundaries.

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `SOLVE/API-001` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ConstructorRejectsInvalidConfig` | `covered` |
| `SOLVE/API-002` | `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigRejectsInvalidConfigWithoutMutation` | `covered` |
| `SOLVE/API-003` | `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigPreservesBackendInvariant` | `covered` |
| `SOLVE/API-004` | `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigDefersPlanRebuildUntilSolve` | `covered` |
| `SOLVE/API-005` | `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackSetConfigIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackResizeHorizonIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackSetCallbackIsRejectedWithoutMutation`, `tests/test_config_regressions.cpp::ConfigRegressionTest.ModelUpdateCallbackClearCallbackIsRejectedWithoutMutation` | `covered` |
| `SOLVE/API-006` | `tests/test_config_regressions.cpp::ConfigRegressionTest.SetConfigRejectsInvalidEnumValues` | `covered` |
| `SOLVE/API-007` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.SnapshotPreservesAllConfigFields` | `covered` |

## Open Gaps

- Config API contract IDs are covered by current tests.
