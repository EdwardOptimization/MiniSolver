# Snapshot Replay Contract

Status: draft

Owner modules:

- `MOD-DEBUG-SNAPSHOT`
- `MOD-CORE-CONFIG`
- `MOD-SOLVER-ROUTE`

Related modules:

- `MOD-MODEL-CODEGEN`
- `MOD-TESTING`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Snapshot format, atomic load failure, config serialization, model fingerprint,
backend policy, and replay reproducibility boundary.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `SNAP-001` | Snapshot files include magic, format version, dimensions, config, horizon, dt, status, algorithmic scalars, and trajectory data. | `covered` |
| `SNAP-002` | Unsupported format versions are rejected. | `covered` |
| `SNAP-003` | Dimension and model fingerprint mismatches are rejected according to load options. | `covered` |
| `SNAP-004` | Invalid serialized config is rejected using normal config validation. | `covered` |
| `SNAP-005` | Non-finite serialized trajectory data is rejected. | `covered` |
| `SNAP-006` | Failed loads must not partially corrupt the current solver. | `covered` |
| `SNAP-007` | Backend load policy decides whether to keep constructed backend, use snapshot backend, or override. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Truncated/trailing bytes | Snapshot status error according to load options. |
| Invalid snapshot data | Snapshot status error; no partial load. |
| Model mismatch | `ModelMismatch` when rejection is enabled. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `SNAP-001` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.CaptureAndSaveAndLoad`, `tests/test_solver_snapshot.cpp::SolverSnapshotTest.FullRoundTrip` | `covered` |
| `SNAP-002` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.RejectsOldFormatMagic` | `covered` |
| `SNAP-003` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsSameDimensionDifferentModelFingerprint`, `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsSameMetadataDifferentGeneratedFingerprint` | `covered` |
| `SNAP-004` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsInvalidSnapshotConfigAtomically`, `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsInvalidSnapshotConfigEnumAtomically`, `tests/test_solver_snapshot.cpp::SolverSnapshotTest.SaveRejectsInvalidSnapshotConfig` | `covered` |
| `SNAP-005` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsNonFiniteTrajectoryDataAtomically` | `covered` |
| `SNAP-006` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsInvalidSnapshotConfigAtomically`, `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsInvalidSnapshotStatusAtomically`, `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadRejectsNonFiniteTrajectoryDataAtomically` | `covered` |
| `SNAP-007` | `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadKeepsConstructedBackendByDefault`, `tests/test_solver_snapshot.cpp::SolverSnapshotTest.LoadCanOverrideBackendExplicitly` | `covered` |

## Open Gaps

- Snapshot format and load failure contracts are covered. Replay corpus quality
  remains a benchmark/process concern rather than a snapshot I/O contract gap.
