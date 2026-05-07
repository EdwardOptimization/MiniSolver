# MiniSolver Testing Matrix

MiniSolver tests are organized around solver contracts, not around a full
Cartesian product of every option. The goal is to keep per-commit CI fast while
still locking the semantics that matter for NMPC correctness, zero-allocation
claims, and generated-model compatibility.

## Scope

In scope:

- Core math kernels: matrix backends, factorization, finite checks, autodiff.
- Solver contracts: primal/dual residuals, barrier updates, line search, SOC,
  implicit integration, restoration behavior, zero-malloc solve paths.
- Code generation: MiniModel identifiers, terminal projection, residual costs,
  constraint packets, implicit Riccati sparsity.
- Integration quality: reference solver path, default solver path, generated
  asset regressions, benchmark smoke.

Out of scope for this matrix:

- Full NMPC performance comparisons against acados/CasADi. Those belong in the
  separate `MiniSolver-Bench` repository.
- Cross-version snapshot schema migration. The current solver snapshot I/O is a
  replay/debug tool, not a stable public interchange format. Current-format
  round-trip behavior is tested, while old formats are rejected explicitly.

## Test Tiers

| Tier | Purpose | Examples | Required Frequency |
| --- | --- | --- | --- |
| Unit | Small deterministic contracts | matrix kernels, barrier update, termination, initialization | Every commit |
| Component | One solver subsystem with controlled inputs | line search, Riccati, implicit integrator, SOC | Every commit |
| Integration | Full `MiniSolver::solve()` on small models | reference/default agreement, soft constraints, memory tests | Every commit |
| Codegen | Python MiniModel generates compilable and correct C++ | residual costs, constraint packets, terminal projection | Every commit |
| Asset regression | Generated real-ish models against stored references | bicycle, double-integrator 3D | Every commit |
| Stress | Long iteration or capacity tests | filter ring-buffer wrap, zero-malloc variants | Every commit if fast; otherwise nightly |
| Benchmark | Runtime and accuracy comparisons | nmpc-bench, matrix microbenchmarks | Manual/nightly, not unit CI |

## Required Matrix

### Backends

| Backend | Coverage |
| --- | --- |
| Eigen | Full `ctest` in `.build` |
| MiniMatrix | Full `ctest` in `.build_custom` |
| CUDA/GPU | No required CI until the GPU backend is made functional |

### Solver Routes

| Route | Must Cover | Current Tests |
| --- | --- | --- |
| Reference path | Conservative IPM semantics without advanced heuristics | `test_solver_quality` |
| Default path | Recommended production defaults | `test_solver_quality`, `test_solver` |
| Filter line search | Acceptance, SOC, fixed-capacity history | `test_line_search` |
| Merit line search | Armijo/failure semantics | `test_line_search`, `test_bugfixes` |
| No line search | RTI-style full step and rollout semantics | `test_line_search` |
| Postsolve verdict | Fresh residuals override stale loop status | `test_barrier_residual_contract` |

### Model Features

| Feature | Must Cover | Current Tests |
| --- | --- | --- |
| Hand-written model | Core C++ API remains usable without codegen | most C++ tests |
| MiniModel generated model | Header compiles and exposes expected symbols | `tests/minimodel/*`, asset regressions |
| Terminal stage projection | Terminal cost/constraints do not depend on `u_N` semantics | `tests/minimodel/test_terminal.py` |
| True/QP/SOC constraint packets | Globalization uses true residuals; QP/SOC may use overrides | `tests/minimodel/test_constraints.py`, `test_line_search` |
| L1/L2 soft constraints | Initialization, convergence, dual/slack safety | `test_soft_constraints`, `test_bugfixes` |
| Residual costs | True Gauss-Newton Hessian and parameter weights | `tests/minimodel/test_residual_costs.py` |
| Implicit integrators | BE, midpoint, Gauss-Legendre accuracy and A/B Jacobians | `test_integrator`, `test_implicit_sparse_riccati` |

### Memory And Diagnostics

| Claim | Required Coverage |
| --- | --- |
| Zero dynamic allocation during production `solve()` | `test_memory` with profiling/logging disabled and multiple line-search modes |
| Profiling/logging | Explicitly not part of the zero-malloc guarantee unless converted to fixed buffers |
| Solver snapshot/replay | Round-trip config, trajectory, soft slacks, backend policy, model fingerprint, and atomic failure behavior. `SnapshotPreservesAllConfigFields` keeps every serialised `SolverConfig` field in the round-trip, anchored by `MakeNonDefaultConfig()` setting a non-default value for each field; `SnapshotPreservesV3FieldsExplicitly` covers the v3 additions (coordinate scaling, Riccati robust mode, restoration penalty knobs, full-KKT iterative refinement, RTI-lite) per field; `RejectsPreviousFormatVersion` keeps the format-version bump honest |
| Logger callback contract | `test_logger` covers callback capture, default config, and `silent_fallback` runtime behavior |
| Embedded no-stream logger profile | `test_logger_no_stream` compiles the same logger test with `MINISOLVER_DISABLE_STREAM_LOGGER` defined to keep `<iostream>` out and silence stream fallbacks |
| Embedded ARM cross-build | CI job `embedded-arm-cortex-m4` configures with `MINISOLVER_EMBEDDED_PROFILE=ON` and `cmake/toolchains/arm-cortex-m4.cmake`, builds `minisolver_embedded_smoke`, and `scripts/check_arm_size_budget.sh` enforces a 64 KiB budget on `text+data+bss` of the smoke object (measured via `arm-none-eabi-size`, *not* on-disk file size, so DWARF/symbol/relocation noise is excluded). Regressions in actual on-target footprint surface before tagging a release; budget changes must update both the script and this matrix entry in the same commit |
| Coordinate-scaling hint contract | `test_coordinate_scaling` covers default unity scales, by-index/by-name setter equivalence, validation rejects (invalid index, name, NaN, out-of-range), `reset_coordinate_scaling`, the `NONE` baseline-equality guarantee, the `USER_SUPPLIED` weighted dual-norm contract, and config validation of `coordinate_scale_{min,max}` |
| Warm-start mu/reg reuse contract | `test_warm_start_reuse` pins the end-to-end behaviour: `REUSE_PRIMAL_DUAL + REUSE_PREVIOUS_MU + DECAY_PREVIOUS_REG` must beat `COLD_START + RESET_TO_*` in cumulative outer iterations on a neighbouring tracking problem, decay never drives reg below `reg_min`, and the reuse modes transparently fall back when the stored primal-dual iterate is invalidated. `tools/warm_start_bench` reports the mean/worst iteration counts across 11 reuse strategy combinations as benchmark evidence |
| MiniMatrix vs Eigen kernel evidence | `tools/mini_matrix_vs_eigen_bench` measures GEMM and LDLT throughput on 4x4-12x12 fixed-size blocks under both backends. The README captures reference numbers and documents the policy: kernel changes (e.g. widening `MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK`, adding SIMD) require a measurable win in this table to land |
| API error contract | `test_config_regressions` and `test_status` cover `ApiStatus` returns for setters, checked scalar getters, and constructor validation |
| Line-search backtracking visibility | `test_line_search` asserts `SolverInfo::line_search_backtracking_count` accumulates on strongly nonlinear merit cases |
| Direction refinement modes | `test_direction_refinement` pins three contracts: (1) the default config keeps `DirectionRefinementMode::NONE` and reports zero refinement passes; (2) `FULL_KKT_ITERATIVE_REFINEMENT` on an unconstrained nonlinear problem records at least one pass per iteration and is bounded by `iterations * direction_refinement_max_passes`; (3) on a problem with active inequality bounds the iterative mode auto-degrades to one pass per iteration, matching `DYNAMICS_DEFECT_ROLLOUT` and preserving the OD-005 dual-consistency safety net. Validation rejects `direction_refinement_max_passes <= 0` and non-positive `direction_refinement_tol` |
| Riccati robustness diagnostics | `test_riccati_robust` pins the contract: (1) default config uses `RiccatiRobustMode::STANDARD`; (2) `SolverInfo::riccati_indefinite_blocks` and `SolverInfo::riccati_max_diagonal_perturbation` stay at zero on a clean SPD problem in both `STANDARD` and `INERTIA_AWARE_DIAGNOSTICS` modes; (3) `INERTIA_AWARE_DIAGNOSTICS` does not flip `degraded_step` when no inertia correction occurred; (4) `SolverInfo::reset()` clears both counters; (5) config validation accepts both modes and rejects unknown enum values. The contract recognises that the small-Nu freeze fallback ALWAYS sets `degraded_step` (via `LinearSolveResult::degraded_step`, the pre-existing N-DEG-1 path), independent of `RiccatiRobustMode`; the per-mode difference is whether the *non-freeze* fallback paths (general-path SPD retry, `SATURATION`, `IGNORE_SINGULAR`) also escalate. Behavioural coverage of the freeze path lives in `test_riccati`/`test_status`. Non-freeze fallback coverage is now in this file: `GeneralPathRegularizationFallbackBumpsCounterButLeavesDegradedFalse` exercises the real general-path REGULARIZATION retry on a NU=4 model with mildly indefinite R(0,0) and pins the per-call invariants (`riccati_indefinite_blocks > 0`, `riccati_max_diagonal_perturbation == regularization_step`, `degraded_step` and `degraded_riccati_freeze_count` stay zero). `StandardModeDoesNotEscalateNonFreezeInertiaEvents` / `InertiaAwareModeEscalatesNonFreezeInertiaEvents` / `FreezeFallbackEscalatesInBothModes` use a per-test `SolverInternalAccess` to drive `record_linear_solver_diagnostics_` with hand-crafted `LinearSolveResult` instances and pin the Solver-level escalation logic for both modes. Square-root and `FACTORIZATION_MODIFY` paths remain explicit non-goals (see OD-006 and `solver-capability-adoption-plan.md` P1) |
| Pareto-frontier filter pruning | `test_filter_pareto` and the rewritten `test_line_search.FilterHistoryParetoCollapsesMonotonicallyImprovingSequence` pin the contract: (1) on a strictly improving (theta, phi) sequence, the filter history collapses to a single Pareto entry; (2) `LineSearchResult::filter_entries_pruned` reports the number of dominated entries removed during insertion, `filter_redundant_inserts` reports a failed insert when the new entry is itself dominated, and `filter_size_after` mirrors the live frontier size; (3) `MeritLineSearch` leaves all three filter fields at zero; (4) `SolverInfo::filter_entries_pruned_total` / `filter_redundant_inserts_total` / `filter_max_history_size` accumulate across an entire solve and are cleared by `SolverInfo::reset()` |
| Restoration penalty rho contract | `test_restoration_penalty` pins: (1) defaults preserve the legacy hardcoded behaviour (`RestorationPenaltyMode::FIXED` with `restoration_rho_init = 1000.0`); (2) validation rejects non-positive `restoration_rho_init` / `restoration_rho_min` / `restoration_rho_max` / `restoration_rho_violation_floor`, `restoration_rho_max < restoration_rho_min`, and unknown `restoration_penalty_mode` enum values; (3) validation accepts `VIOLATION_ADAPTIVE`; (4) `SolverInfo::reset()` clears `restoration_rho_min_used`, `restoration_rho_max_used`, and `restoration_rho_adaptive_steps`. The adaptive scaling itself runs only when feasibility restoration is invoked, so end-to-end behavioural coverage is intentionally deferred until a benchmarked restoration regression appears |

## CI Commands

Per-commit local validation should run both configured build trees:

```bash
cmake --build .build -j$(nproc)
ctest --test-dir .build --output-on-failure

cmake --build .build_custom -j$(nproc)
ctest --test-dir .build_custom --output-on-failure
```

Focused development may run a narrower target first, but full Eigen and
MiniMatrix `ctest` should pass before commit or push.

## Current Follow-Up Items

These are testing obligations, not a request to delete old gap records. When a
gap is closed, keep the discovery in the dated gap/review document and add the
regression test that now protects it.

P0:

- Keep implicit integrator A/B Jacobian checks for all implicit schemes.
- Keep reference/default agreement on at least unconstrained QP, L1 soft, and
  implicit-integrator QP.

P1:

- Add stricter Armijo directional-derivative regression if cancellation is
  observed in a real solver case.
- Expand reference/default agreement as new public model features are added.

P2:

- Keep fixed-capacity line-search history tests lightweight.
- Move long horizon or slow stress coverage to nightly if it becomes expensive.
- Keep snapshot/replay tests current whenever `SolverConfig` or solver runtime
  state changes.
