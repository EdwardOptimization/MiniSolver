# Snapshot And Replay Design

MiniSolver snapshot I/O is a debug and replay facility, not a general-purpose
serialization framework. It exists to capture enough state to reproduce a solver
failure or compare solver behavior across commits.

## Scope

Snapshot I/O may save and restore:

- `SolverConfig`
- runtime metadata such as status, iterations, `mu`, `reg`, and total cost
- horizon length and `dt` trajectory
- primal, slack, soft-slack, and dual trajectory values

Snapshot I/O does not promise:

- backward compatibility with older snapshot formats
- schema migration
- cross-language interchange
- hard-real-time capture without allocation
- compatibility between different model equations that happen to share dimensions

The supported header is:

```cpp
#include "minisolver/debug/solver_snapshot.h"
```

Older `serializer` names are retired. New code should use `SolverSnapshotIO`.

## Failure Capture Pattern

For debugging intermittent failures, capture before `solve()` and persist only if
the solve fails:

```cpp
auto pre_solve = SolverSnapshotIO<Model, MAX_N>::capture_snapshot(solver);
SolverStatus status = solver.solve();

if (status != SolverStatus::OPTIMAL && status != SolverStatus::FEASIBLE) {
    SolverSnapshotIO<Model, MAX_N>::save_failure_snapshot(
        "failure.msnap", pre_solve, status);
}
```

`capture_snapshot()` is an allocating debug snapshot. Do not call it from a
hard real-time control loop unless allocation is acceptable in that context.

## Config Codec Contract

Snapshot config read/write uses the single field table in
`solver_snapshot.h`. When adding a `SolverConfig` field, update that table and
the snapshot config round-trip test in the same commit.

Snapshot load validates the restored config before mutating the target solver.
Invalid config, invalid enum values, bad dimensions, model mismatch, truncated
files, and trailing bytes are rejected atomically.

## Model Fingerprint Contract

Generated MiniModel C++ headers provide:

```cpp
static constexpr std::uint64_t model_fingerprint = ...;
```

Snapshot replay prefers this generated fingerprint. It changes when the symbolic
model changes, so replay rejects snapshots from stale generated models even if
dimensions and names still match.

Handwritten models that need reliable replay should define the same member:

```cpp
struct MyModel {
    static constexpr int NX = 4;
    static constexpr int NU = 2;
    static constexpr int NC = 3;
    static constexpr int NP = 1;

    static constexpr std::uint64_t model_fingerprint = 0x123456789abcdef0ull;

    // model callbacks...
};
```

If a handwritten model does not define `model_fingerprint`, snapshot replay falls
back to metadata hashing: dimensions, variable names, constraint metadata, and
integrator metadata. That fallback is useful for basic protection, but it cannot
prove that dynamics, cost, or constraint equations are unchanged.

For replay corpora and bug reports, prefer explicit fingerprints for handwritten
models.

