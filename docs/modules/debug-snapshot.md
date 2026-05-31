# Debug Snapshot And Replay

Module ID: `MOD-DEBUG-SNAPSHOT`

Status: draft

Files:

- `include/minisolver/debug/solver_snapshot.h`
- Replay tools if present in future revisions.

Owner layer:

- Binary snapshot load/save and solver replay compatibility.

## Purpose

Own snapshot format versioning, config serialization, trajectory serialization,
model/dimension compatibility checks, backend load policy, non-finite rejection,
and atomic load behavior for reproducing solver states.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| `MiniSolver` instance | Snapshot save/load | Friend access is internal. |
| Snapshot file bytes | File system | May be malformed or from another model/version. |
| `SnapshotLoadOptions` | Caller | Selects backend and mismatch policies. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Snapshot file | Debug/replay users | Serialized solver state. |
| Loaded solver state | Solver instance | Config, horizon, trajectory, and algorithmic state. |
| `SnapshotResult` | Caller/tests | Load/save outcome. |

## Owned State

No persistent runtime state beyond local snapshot structs and file I/O.

## Public API Surface

- `SolverSnapshotIO<Model, MAX_N>`
- `SnapshotStatus`
- `SnapshotLoadOptions`
- `SnapshotResult`

## Internal Contracts

- Snapshot format changes require version handling.
- Model fingerprint/dimension mismatch should fail before mutating solver state.
- Invalid snapshots should not partially corrupt the current solver.
- Config fields must stay aligned with `MINISOLVER_CONFIG_FIELDS`.

## Hot-Path And Allocation Policy

- Hot path: no
- Solve-time allocation allowed: not applicable
- Notes: snapshot uses `std::vector` and file I/O by design.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| File open/read/write failure | `SnapshotStatus` | Does not map to `SolverStatus`. |
| Unsupported version | `UnsupportedVersion` | Format boundary. |
| Invalid config in snapshot | `InvalidConfig` plus `ApiStatus` | Reuses config validation. |
| Non-finite data | `NonFiniteData` | Snapshot boundary rejects corrupted data. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/test_solver_snapshot.cpp` | Snapshot load/save and corruption cases. |
| `tests/test_replay_corpus.cpp` | Replay-oriented behavior. |

## Known Gaps

- Snapshot/replay contract IDs are not assigned yet.
