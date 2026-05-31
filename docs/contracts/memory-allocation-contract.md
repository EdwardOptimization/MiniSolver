# Memory Allocation Contract

Status: draft

Owner modules:

- `MOD-TESTING`
- `MOD-SOLVER-ROUTE`
- `MOD-CORE-TRAJ`
- `MOD-SOLVER-RICCATI`
- `MOD-MATRIX`

Related modules:

- `MOD-RUNTIME`
- `MOD-MODEL-CODEGEN`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Solve-time zero-allocation claims, allowed exclusions, and evidence ownership.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `MEM-001` | Default solve path must not allocate after solver construction and required reserves. | `covered` |
| `MEM-002` | Line-search diagnostics that use dynamic storage must reserve before solve-time hot use. | `covered` |
| `MEM-003` | Trajectory, knot, Riccati, matrix, and Newton workspaces must be fixed-size for solve paths. | `covered` |
| `MEM-004` | Logging/profiling enabled paths may allocate and are excluded from default zero-allocation claims unless separately tested. | `covered` |
| `MEM-005` | Generated C++ model packets must not allocate in solve-time functions. | `covered` |
| `MEM-006` | Snapshot/replay and Python codegen are outside solve-time zero-allocation claims. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Allocation in claimed zero-allocation path | Test failure. |
| Allocation in excluded debug/profile path | Allowed only if documented. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `MEM-001` | `tests/test_memory.cpp::MemoryTest.ZeroMalloc_Compliance_Test`, `tests/test_memory.cpp::MemoryTest.DefaultConfigSolveDoesNotAllocate` | `covered` |
| `MEM-002` | `tests/test_memory.cpp::MemoryTest.ZeroMalloc_SolveAfterSetConfigDoesNotAllocate`, `tests/test_memory.cpp::MemoryTest.ZeroMalloc_FilterSOC_Path` | `covered` |
| `MEM-003` | `tests/test_memory.cpp::MemoryTest.ZeroMalloc_ConfigMatrixSolve`, `tests/test_memory.cpp::MemoryTest.ZeroMalloc_ImplicitIntegrator`, `tests/test_riccati.cpp`, fixed-size `KnotPoint`/`Trajectory`/`RiccatiWorkspace` inspection | `covered` |
| `MEM-004` | `docs/testing/testing-matrix.md`, `docs/architecture/api-logger-boundary-design.md`, `tests/test_memory.cpp` disables profiling/logging for zero-allocation claims | `covered` |
| `MEM-005` | `tests/test_memory.cpp::MemoryTest.ZeroMalloc_GeneratedBicycleConstraintModel` | `covered` |
| `MEM-006` | `docs/testing/testing-matrix.md`, `docs/architecture/snapshot-replay-design.md`, `tests/test_solver_snapshot.cpp` | `covered` |

## Open Gaps

- No open P0 memory-allocation contract gaps. Profiling/logging, snapshot I/O,
  replay corpus management, and Python codegen are documented exclusions unless
  a future fixed-buffer or hard-real-time profile adds direct allocation tests.
