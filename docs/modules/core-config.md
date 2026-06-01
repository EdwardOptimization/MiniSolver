# Core Config

Module ID: `MOD-CORE-CONFIG`

Status: draft

Files:

- `include/minisolver/core/solver_options.h`
- `include/minisolver/core/config_fields.h`
- `include/minisolver/core/config_validation.h`

Owner layer:

- Public configuration and config validation.

## Purpose

Own the user-facing solver options, enum policy choices, mechanical config field
registry, and validation rules used before a config becomes solver execution
state.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| `SolverConfig` | User, constructor, `MiniSolver::set_config()` | Values may be invalid until validated. |
| `NewtonConfig` | Nested config for implicit integration | Numeric fields must be finite before use. |
| Backend, line-search, scaling, warm-start, termination enums | User config | Raw enum values may be invalid after casts or binary load. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Validated `SolverConfig` | `MiniSolver`, algorithms, snapshot load | Configuration accepted for execution. |
| `ApiStatus` | Constructor/setter/snapshot callers | Validation result for rejected config. |
| `MINISOLVER_CONFIG_FIELDS` | Snapshot I/O, layout/equality tests | Mechanical registry of serializable config fields. |

## Owned State

This module owns no runtime state. It owns defaults and validation policy.

## Public API Surface

- `SolverConfig`
- config enums such as `LineSearchType`, `TerminationProfile`,
  `BarrierStrategy`, `Backend`, and scaling modes.
- `detail::validate_solver_config()`

## Internal Contracts

- Defaults live in `solver_options.h`.
- Mechanical iteration over fields lives in `config_fields.h`.
- Validation lives in `config_validation.h`.
- Config is resolved into build/runtime state by solver construction or
  `set_config()`, not repeatedly in hot loops.

## Hot-Path And Allocation Policy

- Hot path: no
- Solve-time allocation allowed: no direct allocation here
- Notes: validation should happen at config boundaries.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Invalid enum | `ApiStatus::InvalidArgument` | Prevents undefined enum execution. |
| Non-finite numeric config | `ApiStatus::NonFiniteValue` | Config boundary finite check. |
| Invalid ranges | `ApiStatus::InvalidArgument` | Includes iteration counts, tolerances, scale bounds, and line-search settings. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_config_regressions.cpp` | Config validation and preservation behavior. |
| `tests/test_solver_snapshot.cpp` | Config field registry and snapshot validation. |

## Known Gaps

- Config validation and field-registry behavior is covered by
  [`../contracts/config-api-contract.md`](../contracts/config-api-contract.md)
  and
  [`../contracts/build-config-contract.md`](../contracts/build-config-contract.md).
  Config-to-plan implementation details should be added there if future plan
  projection semantics become externally relevant.
