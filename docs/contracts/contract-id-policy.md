# Contract ID Policy

Contract IDs are stable handles for behavior requirements. They let code
reviews, tests, replay assets, and benchmarks refer to the same solver promise
without depending on prose location.

## Format

Use:

```text
PREFIX-NNN
PREFIX/SUB-NNN
```

Examples:

```text
SOLVE-001
NUM-020
SCALE/CON-001
RIC/REFINE-001
```

Rules:

1. `PREFIX` names the behavior domain.
2. `SUB` is optional and should only be used when a domain has a stable
   subdomain.
3. `NNN` is a zero-padded integer.
4. IDs are never renumbered after review.
5. Retired IDs stay listed as retired; do not reuse them.
6. New requirements should use the next available ID in the owning file.

## Initial Prefixes

| Prefix | Domain |
| --- | --- |
| `SOLVE` | Solve route and phase orchestration. |
| `STATUS` | Final status, loop status, and termination reason projection. |
| `TERM` | Convergence, early stop, stagnation, and fixed-iteration semantics. |
| `POST` | Postsolve residual refresh and final classification. |
| `WARM` | Warm start and initial guess reuse. |
| `NUM` | Numeric boundary checks and residual calculations. |
| `SCALE` | Objective, problem, and row scaling. |
| `BARR` | Barrier parameter and centrality behavior. |
| `REG` | Regularization and linear-solve retry behavior. |
| `INIT` | Slack, dual, and soft-slack initialization. |
| `LS` | Line-search candidate and acceptance behavior. |
| `MERIT` | Merit function globalization. |
| `FILTER` | Filter globalization. |
| `RIC` | Riccati/KKT assembly and direction recovery. |
| `REST` | Restoration phase behavior. |
| `SOC` | Second-order correction behavior. |
| `CON` | Constraint packet and hard-constraint semantics. |
| `SOFT` | Soft-constraint semantics. |
| `MODEL` | Model interface and callback behavior. |
| `CODEGEN` | MiniModel and generated C++ behavior. |
| `PARAM` | Parameter ownership and update semantics. |
| `MEM` | Allocation and memory behavior. |
| `LOG` | Logging and profiling behavior. |
| `BACKEND` | Backend selection and unsupported backend behavior. |
| `MAT` | Matrix backend and kernel behavior. |
| `BUILD` | Build and CMake behavior. |
| `SNAP` | Snapshot and replay behavior. |
| `DIAG` | Diagnostics and `SolverInfo` projection. |
| `INT` | Integrator behavior. |

## Evidence Linkage

Every active ID should appear in
[`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)
with one of these statuses:

| Status | Meaning |
| --- | --- |
| `covered` | Direct test, benchmark, replay, or memory evidence exists. |
| `partial` | Some behavior is covered, but important variants or failures are missing. |
| `missing` | No direct evidence exists yet. |
| `deferred` | Evidence belongs in benchmark, replay, nightly, or future feature work. |
