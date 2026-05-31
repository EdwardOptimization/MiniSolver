# Contract-Driven Development Plan

MiniSolver has reached the point where review-driven bug discovery is no longer
enough. The next quality step is to make solver behavior explicit: every major
flow, module, numerical boundary, and model/codegen contract should have a
documented owner, inputs, outputs, invariants, failure semantics, and test
coverage status.

This plan defines the full rollout. It is intentionally broader than a minimal
documentation patch, but it should still be implemented in small docs/test
commits.

## Rollout Status

Status: initial rollout complete as of the first contract coverage closure pass.

Authoritative evidence:

- Module inventory exists and every module row links to a module document.
- Contract files exist for every behavior domain listed in this plan.
- Every contract ID appears in
  [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md).
- Current `P0` coverage matrix rows are `covered`.
- Remaining `partial` rows are `P1/P2` with owner modules and explicit deferred
  evidence paths.
- The completion audit is tracked in
  [`../testing/contract-rollout-completion-audit.md`](../testing/contract-rollout-completion-audit.md).

Future contract work should update the relevant contract file and coverage
matrix row in the same change as the behavior or test evidence.

## Goals

1. Move from ad-hoc review findings to contract-driven development.
2. Give each solver module a clear owner boundary and input/output shape.
3. Define behavior contracts independently from the current header layout.
4. Map every contract ID to tests, replay assets, memory checks, or benchmarks.
5. Make future feature work start from a contract or an explicit contract
   update.

## Non-Goals

- Do not mirror `include/minisolver/**` one-to-one under `docs/contracts/`.
- Do not create public plugin or strategy APIs just to make documentation neat.
- Do not require a Cartesian-product test matrix for every solver option.
- Do not rewrite solver code during the documentation bootstrap phase.
- Do not make snapshot/replay a stable external interchange format.

## Directory Plan

Create two complementary documentation layers.

`docs/modules/` describes implementation modules. It may be close to the code
layout because it is about ownership and local inputs/outputs.

`docs/contracts/` describes behavior. It should be organized by solver behavior,
failure semantics, and cross-module flows, not by header file.

Target structure:

```text
docs/contracts/
  README.md
  _template.md
  contract-id-policy.md

  solver/
    solve-loop-contract.md
    status-semantics-contract.md
    termination-contract.md
    postsolve-contract.md
    warm-start-contract.md
    config-api-contract.md

  numeric/
    finite-check-boundary-contract.md
    residual-contract.md
    scaling-contract.md
    barrier-mu-contract.md
    regularization-contract.md
    tolerance-contract.md

  algorithms/
    initialization-contract.md
    line-search-contract.md
    filter-line-search-contract.md
    merit-line-search-contract.md
    riccati-contract.md
    restoration-contract.md
    soc-contract.md
    direction-refinement-contract.md

  constraints/
    hard-constraints-contract.md
    soft-constraints-contract.md
    constraint-packets-contract.md
    row-scaling-contract.md

  model/
    model-interface-contract.md
    minimodel-codegen-contract.md
    generated-packets-contract.md
    model-callback-contract.md
    parameter-contract.md

  runtime/
    memory-allocation-contract.md
    logging-profiling-contract.md
    backend-contract.md
    matrix-backend-contract.md
    build-config-contract.md

  debug/
    solver-info-contract.md
    snapshot-replay-contract.md
    diagnostics-contract.md

  integrators/
    explicit-integrator-contract.md
    implicit-integrator-contract.md
    numerical-jacobian-contract.md

docs/modules/
  README.md
  _template.md
  module-inventory.md

  core-config.md
  core-types.md
  core-trajectory.md
  core-semantics.md
  solver-route.md
  solver-riccati.md
  solver-line-search-utils.md
  algorithms-initialization.md
  algorithms-line-search.md
  algorithms-model-evaluation.md
  algorithms-termination.md
  algorithms-barrier.md
  integrators.md
  matrix.md
  model-codegen.md
  debug-snapshot.md
  runtime.md
  testing.md

docs/testing/
  contract-coverage-matrix.md
```

## Contract ID Policy

Contract IDs are stable behavior IDs. They should survive header splits and
minor refactors.

Suggested prefixes:

| Prefix | Area |
| --- | --- |
| `SOLVE` | Solve route, phase order, canonical loop |
| `STATUS` | `SolverStatus`, `TerminationReason`, `SolverInfo` |
| `TERM` | Termination predicates and stagnation exits |
| `POST` | Postsolve refresh and final verdict |
| `WARM` | Warm start and primal-dual reuse |
| `NUM` | Finite checks, residuals, numerical boundaries |
| `SCALE` | Objective, constraint, and problem scaling |
| `BARR` | Barrier `mu`, complementarity, centrality |
| `REG` | Regularization and linear-solve attempts |
| `INIT` | Primal-dual initialization |
| `LS` | Shared line-search semantics |
| `MERIT` | Merit line search |
| `FILTER` | Filter line search |
| `RIC` | Riccati and KKT solve |
| `REST` | Feasibility restoration |
| `SOC` | Second-order correction |
| `CON` | Hard constraints and constraint packets |
| `SOFT` | L1/L2/mixed soft constraints |
| `MODEL` | C++ model interface and callback behavior |
| `CODEGEN` | MiniModel and generated C++ packets |
| `PARAM` | Parameter semantics |
| `MEM` | Runtime allocation and hard-real-time claims |
| `LOG` | Logging and profiling |
| `BACKEND` | Backend selection and unsupported backends |
| `MAT` | Matrix backend and kernels |
| `SNAP` | Snapshot and replay |
| `DIAG` | Diagnostics and solver reports |
| `INT` | Integrators and numerical Jacobians |

Each ID should appear in exactly one contract file and at least one coverage
matrix row.

## Module Document Template

Every module document should use the same small schema:

```text
Module ID
Files
Owner layer
Purpose
Non-goals
Inputs
Outputs
Owned state
Hot-path status
Allocation policy
Failure/status semantics
Public API surface
Related contracts
Current tests
Known gaps
```

## Contract Document Template

Every contract document should use this schema:

```text
Purpose
Related code
Related modules
Related tests
Inputs
Outputs
Invariants
Failure semantics
Hot-path policy
Required coverage
Non-goals
Known gaps
```

The `Inputs` section should classify each input by source and trust level:

```text
Source: config / public setter / generated model / callback / internal state
Checked where: config validation / setter / residual boundary / postsolve / debug only
Trust level: checked / trusted model input / internal invariant
```

## Coverage Matrix Schema

`docs/testing/contract-coverage-matrix.md` should be the authoritative test
ledger.

Columns:

```text
Contract ID
Contract file
Behavior
Owner module
Related code
Unit test
Component test
Integration test
Codegen test
Asset/replay test
Benchmark/memory evidence
Status
Priority
Last evidence commit
Notes
```

Allowed status values:

| Status | Meaning |
| --- | --- |
| `covered` | The contract has direct evidence in tests or benchmark/memory evidence. |
| `partial` | Some behavior is tested, but failure semantics or important variants are missing. |
| `missing` | No direct test currently protects the contract. |
| `deferred` | Coverage belongs in replay, benchmark, nightly, or future feature work. |

## Module Inventory Scope

The module inventory should start with these implementation modules.

| Module ID | Files | Purpose |
| --- | --- | --- |
| `MOD-CORE-CONFIG` | `core/solver_options.h`, `core/config_fields.h`, `core/config_validation.h` | User configuration, field registry, validation, construction-time solver policy. |
| `MOD-CORE-TYPES` | `core/types.h` | Status enums, result structs, knot state, solver info, and phase payload types. |
| `MOD-CORE-TRAJ` | `core/trajectory.h` | Active/candidate trajectory buffers and double-buffer swapping. |
| `MOD-CORE-SEMANTICS` | `core/constraint_semantics.h`, `core/model_traits.h` | Structural model traits, soft/hard row classification, scaling activation. |
| `MOD-SOLVER-ROUTE` | `solver/solver.h` | Public API, solve route, phase orchestration, postsolve, diagnostics projection. |
| `MOD-SOLVER-RICCATI` | `solver/riccati.h`, `solver/kkt_assembler.h`, `algorithms/riccati_solver.h`, `algorithms/linear_solver.h`, `algorithms/linear_solve_result.h` | KKT/Riccati solve, dual recovery, linear-solve failure and regularization integration. |
| `MOD-SOLVER-LSUTIL` | `solver/line_search_utils.h` | Fraction-to-boundary and shared line-search helper semantics. |
| `MOD-ALG-INIT` | `algorithms/initialization.h` | Slack, dual, and soft slack initialization on a central path. |
| `MOD-ALG-LS` | `algorithms/line_search.h` | Merit, filter, and no-line-search globalization. |
| `MOD-ALG-EVAL` | `algorithms/model_evaluation.h` | Model evaluation, cost/constraint packets, true/QP/SOC packets, scaling application. |
| `MOD-ALG-TERM` | `algorithms/termination.h`, `algorithms/residual_stagnation_monitor.h` | Convergence, feasible shortcuts, stagnation exits, loop-level reason selection. |
| `MOD-ALG-BARRIER` | `algorithms/barrier_update.h` | Barrier update policy and `mu` evolution. |
| `MOD-INTEGRATOR` | `integrator/*.h` | Explicit/implicit integration, Newton solve, numerical Jacobian support. |
| `MOD-MATRIX` | `matrix/*.h` | Matrix storage, kernels, policies, backend-independent math primitives. |
| `MOD-MODEL-CODEGEN` | `python/minisolver/MiniModel.py`, `python/minisolver/templates/model.h.in` | User modeling DSL, generated model packets, structural metadata and codegen hooks. |
| `MOD-DEBUG-SNAPSHOT` | `debug/solver_snapshot.h`, `tools/replay_solver.cpp` if present | Snapshot/replay debug I/O and failure reproduction scope. |
| `MOD-RUNTIME` | `core/logger.h`, `backend/backend_interface.h`, CMake/build options | Runtime logging, profiling, backend policy, unsupported backend failure. |
| `MOD-TESTING` | `tests/**`, `docs/testing/**` | Unit/component/integration/codegen/replay/asset/memory evidence. |

## Contract Scope

The first full contract set should include the following.

### Solver Contracts

| Contract | Primary IDs | Required topics |
| --- | --- | --- |
| `solve-loop-contract.md` | `SOLVE-001...` | Phase order, callback timing, buffer ownership, early exits, loop status. |
| `status-semantics-contract.md` | `STATUS-001...` | Final status vs loop status, reason projection, `SolverInfo` meaning. |
| `termination-contract.md` | `TERM-001...` | Strict KKT, acceptable NMPC, RTI fixed iteration, cost/residual stagnation. |
| `postsolve-contract.md` | `POST-001...` | Fresh residual refresh, final authority, stale loop status correction. |
| `warm-start-contract.md` | `WARM-001...` | Primal-dual reuse, callback restrictions, `mu/reg` selection. |
| `config-api-contract.md` | `SOLVE/API-001...` | Config validation, public setter mutation rules, callback mutation restrictions. |

### Numeric Contracts

| Contract | Primary IDs | Required topics |
| --- | --- | --- |
| `finite-check-boundary-contract.md` | `NUM-001...` | Where finite checks belong, trusted model inputs, residual/direction/line-search/postsolve boundaries. |
| `residual-contract.md` | `NUM-020...` | Primal/dual/complementarity metrics, NaN-safe reductions, scaled/unscaled reporting. |
| `scaling-contract.md` | `SCALE-001...` | Objective scaling, row scaling, problem scaling activation, NaN/Inf scale reduction semantics. |
| `barrier-mu-contract.md` | `BARR-001...` | `mu` ownership, complementarity average, centrality residuals, Mehrotra fallback. |
| `regularization-contract.md` | `REG-001...` | Linear-solve attempts, regularization escalation/downscale, degraded step diagnostics. |
| `tolerance-contract.md` | `NUM-050...` | `tol_con`, `tol_dual`, `tol_mu`, scaled vs unscaled semantics. |

### Algorithm Contracts

| Contract | Primary IDs | Required topics |
| --- | --- | --- |
| `initialization-contract.md` | `INIT-001...` | Slack/dual floors, L1/L2/mixed central path, warm-start rebuilding. |
| `line-search-contract.md` | `LS-001...` | Candidate construction, alpha ownership, finite scalar failure, accepted step semantics. |
| `merit-line-search-contract.md` | `MERIT-001...` | Merit value, Armijo derivative, `dphi/phi` nonfinite failure. |
| `filter-line-search-contract.md` | `FILTER-001...` | Filter history, switching condition, `theta/phi` semantics, reset on barrier update. |
| `riccati-contract.md` | `RIC-001...` | KKT inputs, sigma convention, soft row derivatives, direction outputs. |
| `restoration-contract.md` | `REST-001...` | Restoration entry, improvement requirement, status precedence. |
| `soc-contract.md` | `SOC-001...` | SOC candidate construction, scale/soft weight refresh, accept/reject semantics. |
| `direction-refinement-contract.md` | `RIC/REFINE-001...` | Optional refinement mode and allowed mutation. |

### Constraint Contracts

| Contract | Primary IDs | Required topics |
| --- | --- | --- |
| `hard-constraints-contract.md` | `CON-001...` | Hard row residuals, slack/dual KKT, active row scaling. |
| `soft-constraints-contract.md` | `SOFT-001...` | L1, L2, mixed L1+L2, zero/tiny weight semantics, floors, shared relaxation. |
| `constraint-packets-contract.md` | `CON-020...` | QP constraints, true constraints, terminal constraints, SOC constraints. |
| `row-scaling-contract.md` | `SCALE/CON-001...` | Row scale storage, reuse for candidates, unscaled residual transformation. |

### Model And Codegen Contracts

| Contract | Primary IDs | Required topics |
| --- | --- | --- |
| `model-interface-contract.md` | `MODEL-001...` | Required static fields/functions, handwritten model compatibility. |
| `minimodel-codegen-contract.md` | `CODEGEN-001...` | DSL ownership, symbolic checks, generated C++ structure. |
| `generated-packets-contract.md` | `CODEGEN-020...` | Full overwrite vs clear, cost/constraint/dynamics packet ownership. |
| `model-callback-contract.md` | `MODEL-CB-001...` | Callback timing, allowed data updates, forbidden structural mutation. |
| `parameter-contract.md` | `PARAM-001...` | Parameter trust level, setter policy, generated parameter usage. |

### Runtime, Debug, And Integrator Contracts

| Contract | Primary IDs | Required topics |
| --- | --- | --- |
| `memory-allocation-contract.md` | `MEM-001...` | Solve-time zero allocation, profiling/logging exclusions, test evidence. |
| `logging-profiling-contract.md` | `LOG-001...` | Logging levels, profiling overhead, real-time boundary. |
| `backend-contract.md` | `BACKEND-001...` | Backend selection, unsupported GPU behavior, backend preservation. |
| `matrix-backend-contract.md` | `MAT-001...` | Eigen/MiniMatrix parity, fixed-size kernels, policy boundaries. |
| `build-config-contract.md` | `BUILD-001...` | CMake options, custom matrix build, generated asset builds. |
| `solver-info-contract.md` | `DIAG-001...` | `SolverInfo` projection, timing/iteration/status fields. |
| `snapshot-replay-contract.md` | `SNAP-001...` | Current-format replay, model fingerprint, atomic load failure. |
| `diagnostics-contract.md` | `DIAG-020...` | Alpha log, SOC/restoration counters, regularization diagnostics. |
| `explicit-integrator-contract.md` | `INT-001...` | Explicit integration packet and Jacobian semantics. |
| `implicit-integrator-contract.md` | `INT-020...` | Newton solve, implicit schemes, Jacobian accuracy. |
| `numerical-jacobian-contract.md` | `INT-040...` | Finite-difference fallback and tolerances. |

## Execution Plan

### Phase 0: Plan And Skeleton

Commit shape: docs-only.

Tasks:

1. Add this plan.
2. Add `docs/contracts/README.md`.
3. Add `docs/contracts/_template.md`.
4. Add `docs/contracts/contract-id-policy.md`.
5. Add `docs/modules/README.md`.
6. Add `docs/modules/_template.md`.
7. Add empty `docs/modules/module-inventory.md`.
8. Add empty `docs/testing/contract-coverage-matrix.md`.
9. Link new docs from `docs/README.md`.

Validation:

```bash
git diff --check
```

### Phase 1: Module Inventory

Commit shape: docs-only, possibly split by module domain if large.

Tasks:

1. Fill `docs/modules/module-inventory.md`.
2. Add one module document per module listed above.
3. For each module, define inputs, outputs, owner layer, owned state, hot-path
   status, allocation policy, failure semantics, public API surface, related
   tests, and known gaps.
4. Do not invent new behavior contracts yet; only link likely future contract
   files.

Validation:

```bash
git diff --check
```

### Phase 2: Core Behavior Contracts

Commit shape: docs-only.

Tasks:

1. Write solver lifecycle contracts:
   - solve loop
   - status semantics
   - termination
   - postsolve
   - warm start
2. Write numeric boundary contracts:
   - finite checks
   - residuals
   - scaling
   - barrier `mu`
   - regularization
   - tolerances
3. Assign initial contract IDs.
4. Link each contract to owner modules.

Validation:

```bash
git diff --check
```

### Phase 3: Algorithm, Constraint, Model, Runtime Contracts

Commit shape: docs-only, split into several commits if needed.

Tasks:

1. Write line-search, Riccati, initialization, restoration, SOC, and direction
   refinement contracts.
2. Write hard/soft constraint and constraint packet contracts.
3. Write MiniModel/codegen/model callback/parameter contracts.
4. Write memory/backend/matrix/logging/snapshot/diagnostics/integrator
   contracts.

Validation:

```bash
git diff --check
```

### Phase 4: Coverage Matrix

Commit shape: docs-only.

Tasks:

1. Add every contract ID to `contract-coverage-matrix.md`.
2. Map each ID to current tests, replay assets, memory tests, or benchmark
   evidence.
3. Mark each row as `covered`, `partial`, `missing`, or `deferred`.
4. Assign P0/P1/P2 priority.
5. Record the last evidence commit where known.

Validation:

```bash
git diff --check
```

### Phase 5: P0 Test Closure

Commit shape: tests plus behavior fixes only when red tests prove a gap.

P0 examples:

- `NUM`: line-search `phi/theta` nonfinite values return `NUMERICAL_ERROR`.
- `STATUS`: every terminal status has a stable reason projection.
- `POST`: postsolve nonfinite residuals override stale loop success.
- `SOFT`: mixed L1+L2 paths use one shared relaxation consistently.
- `SCALE`: scaled/unscaled residuals preserve NaN/Inf and report correct units.
- `MEM`: zero allocation for default, merit, filter, and soft-constraint solve
  paths.

Validation:

```bash
cmake --build build -j16
ctest --test-dir build --output-on-failure
git diff --check
```

Run custom MiniMatrix build too before push when the touched contract affects
matrix/Riccati/backend behavior.

### Phase 6: P1/P2 Replay, Benchmark, And Nightly Closure

Commit shape: tests/assets/bench harness updates split from solver behavior.

P1 examples:

- Callback plus `ACCEPTABLE_NMPC` behavior.
- RTI fixed-iteration precedence.
- SOC scale/soft-weight refresh.
- Snapshot/replay drift when config/runtime state changes.
- MiniMatrix parity.
- Generated packet overwrite/clear coverage.

P2 examples:

- nmpc-bench smoke.
- Badly scaled replay corpus.
- Warm-start replay corpus.
- Race-cars replay regression corpus.
- Long-horizon memory/performance smoke.

## Required Development Rule After Rollout

After the contract framework exists, every behavior change should include:

1. A contract ID.
2. A contract update, or an explicit statement that the contract is unchanged.
3. A red test or evidence path tied to the contract ID.
4. A coverage matrix update.
5. Commit trailer evidence.

Suggested trailer:

```text
Harness: core5
Scope: solver-core
Contracts: NUM-004, LS-003
Evidence: ...
```

## Completion Criteria

The rollout is complete when:

1. Every module in the inventory has a module document.
2. Every major behavior domain has at least one contract file.
3. Every contract file has stable contract IDs.
4. Every contract ID appears in `contract-coverage-matrix.md`.
5. All P0 rows are `covered`.
6. P1 rows are either `covered` or have an owner and clear deferred evidence
   path.
7. Full Eigen `ctest` passes.
8. Full MiniMatrix `ctest` passes before push for matrix/Riccati/backend
   contract changes.
9. `test_memory` backs all zero-allocation claims.
10. Replay corpus covers at least termination, soft constraints, scaling,
    callbacks, snapshot/replay, and one real-ish generated model.

## Recommended Commit Order

1. `docs: add contract development plan`
2. `docs: add contract and module skeleton`
3. `docs: inventory solver modules`
4. `docs: add solver and numeric contracts`
5. `docs: add algorithm and constraint contracts`
6. `docs: add model codegen runtime and debug contracts`
7. `docs: add contract coverage matrix`
8. `tests: cover P0 numeric and line-search contracts`
9. `tests: cover P0 status postsolve and termination contracts`
10. `tests: cover P0 soft scaling and codegen contracts`

## Current Next Step

Commit the rollout in small groups following the recommended order above. After
that, normal MiniSolver changes should start from a contract ID or an explicit
contract update.
