# API Error And Logger Boundary Design

Last updated: 2026-05-03

Status: API setter/getter status phases and host logger backend implemented;
embedded no-stream logger profile remains to be implemented.

Owner: MiniSolver public API and observability boundary.

Related review items: N-API-1, N-OBS-2, N-EMBED-1, N-OBS-1.

## Problem

MiniSolver still has two related release-boundary issues:

- Some user-facing setters silently return on invalid input, while name-based
  setters only warn and then return.
- The logging macro layer writes directly to `std::cout` / `std::cerr`, uses
  `std::endl`, and hardcodes ANSI escape sequences.

These are not core algorithm bugs, but they weaken the solver contract:

- invalid user input can be missed;
- embedded users cannot redirect or compile out output cleanly;
- future diagnostics would keep adding ad-hoc warning paths;
- snapshot I/O and solver runtime logging remain conceptually mixed unless
  the boundary is explicit.

## Non-Goals

- Do not add a public OOP plugin framework.
- Do not expose logger or strategy objects as solver hot-path extension points.
- Do not refactor snapshot in this pass. Snapshot remains a separate module
  that needs its own format and I/O cleanup.
- Do not change solver numerical behavior while changing API error reporting.
- Do not add exceptions to hot solve paths.

## Current MiniSolver Surface

Current setter behavior is mixed:

- `resize_horizon()` logs and returns when `new_n` is out of range.
- `set_initial_state(vector)` silently returns on size mismatch.
- indexed setters silently return on invalid stage or index.
- string setters warn on unknown names.
- scalar getters often return `0.0` for invalid inputs.
- vector getters return an empty vector for invalid stages or names.

Current logger behavior is also mixed:

- `MLOG_ERROR/WARN/INFO/DEBUG` write directly to C++ streams.
- warnings and debug use ANSI colors unconditionally.
- `std::endl` forces flushes.
- `MINISOLVER_LOG_LEVEL` can compile out messages, but enabled logging is not
  redirectable and is not embedded-safe.

## Mature-System Signals

The design below is based on local source inspection of mature solvers under
`/tmp`.

### acados

acados exposes integer status codes such as `ACADOS_SUCCESS`,
`ACADOS_NAN_DETECTED`, `ACADOS_MAXITER`, `ACADOS_MINSTEP`,
`ACADOS_QP_FAILURE`, and `ACADOS_INFEASIBLE`, with `status_to_string()`.
Major operations return or store status instead of silently succeeding.

### OSQP

OSQP setup and settings-update APIs return explicit error codes after validating
data and settings. Verbosity and profiling are settings. Code generation also
has explicit printing/profiling switches, which separates embedded builds from
host-debug builds.

### Clarabel

Clarabel keeps structured settings and solver info, and supports configurable
print targets: stdout, file, buffer, arbitrary stream, or sink. The important
lesson for MiniSolver is not the exact Rust API, but the separation between
solver state and output destination.

### Ipopt

Ipopt returns `ApplicationReturnStatus` from initialize/optimize operations and
routes output through a `Journalist` abstraction with levels and categories.
The important lesson is centralized output routing, not adding an Ipopt-style
public output framework to MiniSolver.

## Design Principles

1. Users configure solver behavior through `SolverConfig` and existing setter
   families. Do not expose public strategy/plugin objects.
2. Invalid user input should return an explicit status. Warnings are secondary.
3. Setters may be ignored by callers, but the return value must be available for
   production code.
4. Logging must be centralized and redirectable. Solver code should not write to
   streams directly.
5. Logging and profiling are observability features. They must have documented
   zero-malloc boundaries.
6. Snapshot I/O is out of scope. Do not use snapshot as the logging model.

## API Error Contract

Add a small status enum, for example:

```cpp
enum class ApiStatus {
    OK = 0,
    InvalidHorizon,
    InvalidStage,
    InvalidIndex,
    UnknownName,
    SizeMismatch,
    NonFiniteValue,
    TerminalControl,
    InvalidArgument
};
```

Also add:

```cpp
const char* api_status_to_string(ApiStatus status);
bool api_status_ok(ApiStatus status);
```

### Setter Rule

Change user-facing setters from `void` to `ApiStatus`.

This is intentionally cleaner than adding duplicate `try_set_*` APIs:

- existing call sites that ignore the return value still compile;
- production users can check the return value;
- no temporal side channel such as `last_api_status()` is needed;
- no duplicate setter matrix has to be maintained.

Examples:

```cpp
ApiStatus resize_horizon(int new_n);
ApiStatus set_initial_state(const std::vector<double>& x0);
ApiStatus set_initial_state(const std::string& name, double value);
ApiStatus set_parameter(int stage, int idx, double value);
ApiStatus set_parameter(int stage, const std::string& name, double value);
ApiStatus set_control_guess(int stage, int idx, double value);
ApiStatus set_dt(const std::vector<double>& dts);
```

Invalid inputs must not mutate solver state.

### Getter Rule

Do not broaden the first pass by changing every getter. Instead:

- keep current convenience getters for source compatibility;
- add checked scalar getters where ambiguity matters:

```cpp
ApiStatus get_parameter(int stage, int idx, double& out) const;
ApiStatus get_state(int stage, int idx, double& out) const;
ApiStatus get_control(int stage, int idx, double& out) const;
ApiStatus get_slack(int stage, int idx, double& out) const;
ApiStatus get_dual(int stage, int idx, double& out) const;
```

After users migrate, the silent `0.0` scalar getters can be deprecated or
documented as convenience-only.

### Constructor And Config Boundary

Constructor/config validation should be explicit and early:

- invalid `initial_N` should not silently clamp in release-quality API;
- invalid `SolverConfig` values should be rejected at construction,
  `set_config()`, or solve pre-build boundary;
- hot solve loops should not add defensive checks for invalid config states that
  should have been rejected earlier.

This follows the same rule as the solver strategy work: resolve expensive or
semantic choices at construction/config-build boundaries, not repeatedly inside
hot loops.

## Logger Contract

Replace direct stream writes inside `MLOG_*` with one centralized logging
backend.

Recommended minimum interface:

```cpp
enum class LogLevel {
    Error,
    Warn,
    Info,
    Debug
};

using LogCallback = void (*)(LogLevel level, const char* message, void* user);

struct LoggerConfig {
    LogCallback callback = nullptr;
    void* user = nullptr;
    bool enable_color = false;
};
```

The default host backend can still write to stdout/stderr. Embedded users should
be able to select one of:

- compile-time no-op logging with `MINISOLVER_LOG_LEVEL=0`;
- a fixed-buffer callback;
- a platform callback.

The default logger should not hardcode ANSI color. Color belongs to host-debug
configuration, not solver core.

### Stream-Style Macro Transition

The current macro syntax is convenient:

```cpp
MLOG_WARN("Unknown state " << name);
```

The transition can keep this syntax initially by formatting into a local stream
only when the log level is enabled. However, the zero-malloc contract must say:

- `MINISOLVER_LOG_LEVEL=0` compiles logging out;
- host logging may allocate or flush;
- embedded zero-malloc solve requires logging disabled or a fixed-buffer logger.

Longer term, a fixed-buffer formatter can replace stream formatting for embedded
profiles.

## Implementation Order

### Phase 1: Status Type And Setter Batch

Add `ApiStatus` and convert setters in one or two behavior-scoped commits.

Required red tests:

- unknown state/control/parameter name returns `UnknownName`;
- invalid stage/index returns `InvalidStage` or `InvalidIndex`;
- terminal control setter returns `TerminalControl` and does not mutate state;
- wrong-size initial state vector returns `SizeMismatch`;
- invalid `dt` vector behavior is explicit and tested.

Validation:

- `ctest` full pass;
- README examples still compile if they ignore setter return values.

### Phase 2: Checked Scalar Getters

Add checked scalar getter overloads without deleting convenience getters.

Required tests:

- invalid scalar getter reports error without returning a fake value through the
  checked path;
- existing convenience getters retain documented behavior.

### Phase 3: Logger Backend

Route `MLOG_*` through a central backend and remove hardcoded ANSI output from
the default path.

Required tests:

- a test callback captures a warning without writing to stdout/stderr;
- `MINISOLVER_LOG_LEVEL=0` compiles log calls out for the tested path;
- snapshot direct I/O remains unchanged and documented as deferred.

### Phase 4: Release Documentation

Update README and testing matrix:

- state that solver behavior is configured through `SolverConfig`;
- state that API setters return `ApiStatus`;
- document logger/zero-malloc boundary;
- keep snapshot as a known separate cleanup item.

## Architecture Review

Rejected options:

- `last_api_status()` side channel: creates temporal coupling.
- Duplicate `try_set_*` API for every setter: doubles the public surface.
- Public logger/plugin object framework: unnecessary for current use cases.
- Exceptions for all setters: poor fit for embedded and C-style generated use.

Chosen route:

- setters return explicit `ApiStatus`;
- checked getters are added where ambiguity matters;
- logging goes through one internal backend selected by compile-time level and
  optional callback;
- users still configure solver algorithms through `SolverConfig`.

## Status

This document is both the design contract and implementation ledger.

Implemented:

- Phase 1 setter status returns: `ApiStatus` is available, high-level setters
  return explicit status, and invalid setter inputs do not mutate solver state.
- Phase 2 checked scalar getters: state, control, parameter, slack, and dual
  checked scalar getters report invalid access through `ApiStatus` without
  overwriting the output reference.
- Phase 2b constructor/config validation: constructors reject invalid horizon
  and invalid `SolverConfig`; `set_config()` validates a candidate config before
  mutating solver state and preserves the constructor-selected backend.
- Phase 3 host logger backend: `MLOG_*` routes through a centralized backend
  with optional callback capture, default ANSI color is disabled, and host
  output uses newline writes instead of `std::endl`.

Still deferred:

- Embedded fixed-buffer/no-stream logger profile.
- Phase 4 release documentation update after the API/logger behavior stabilizes.

Future implementation must follow the evidence-driven workflow: red tests first,
one behavior group per commit, full CTest before push.
