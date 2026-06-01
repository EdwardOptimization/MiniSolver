# Runtime, Logging, Backend, And Build

Module ID: `MOD-RUNTIME`

Status: draft

Files:

- `include/minisolver/core/logger.h`
- `include/minisolver/backend/backend_interface.h`
- `CMakeLists.txt`
- build options and backend-related compile definitions

Owner layer:

- Runtime logging/profiling boundaries, backend selection interface, and build
  configuration.

## Purpose

Own logger callback configuration, logging macros, backend dispatch boundary,
CMake backend selection, dependency fetching, build flags, CUDA opt-in gating,
test/example/tool build switches, and matrix backend compile definitions.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| `LoggerConfig` | User/test | Callback may be null. |
| `MINISOLVER_LOG_LEVEL` | Compile definition | Controls macro expansion. |
| CMake options | Build user/CI | Select dependencies, matrix backend, fast math, native arch, tests/tools. |
| `Backend` config | Solver config | CPU serial implemented; GPU reserved. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Log messages | User/test callback or stdout/stderr | Diagnostic output. |
| Compile definitions | Headers | Select matrix/log/backend behavior. |
| Backend dispatch entry point | Future GPU implementation | Reserved bridge. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| Global `LoggerConfig` | Process lifetime | Set through `set_logger_config()`. |

## Public API Surface

- `LoggerConfig`
- `set_logger_config()`, `get_logger_config()`
- `MLOG_*` macros
- CMake options documented in `CMakeLists.txt`

## Internal Contracts

- GPU backend must not silently fall back to CPU.
- Logging/profiling overhead must be excluded from zero-allocation/performance
  claims unless explicitly enabled and measured.
- Build flags such as fast math are benchmark-sensitive and need evidence.

## Hot-Path And Allocation Policy

- Hot path: logging macros may appear in hot code but should compile away or be
  disabled in performance builds.
- Solve-time allocation allowed: no for default no-profiling/no-debug claims.
- Notes: enabled logging uses streams and may allocate.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Unsupported GPU backend | Linear solve failure path | Runtime/backend boundary. |
| Missing dependencies | CMake configure failure | Build-time failure. |
| Logger callback failure | Not handled | Callback is user-owned. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_logger.cpp` | Logger API behavior. |
| `tests/test_memory.cpp` | Allocation-sensitive runtime paths. |
| CI/build commands | CMake option behavior. |

## Known Gaps

- Logging/profiling, backend policy, and memory exclusions are covered by
  [`../contracts/logging-profiling-contract.md`](../contracts/logging-profiling-contract.md),
  [`../contracts/backend-contract.md`](../contracts/backend-contract.md),
  and
  [`../contracts/memory-allocation-contract.md`](../contracts/memory-allocation-contract.md).
