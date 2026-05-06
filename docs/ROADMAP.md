# MiniSolver Roadmap

This file tracks **milestones** (not daily progress). The goal is to keep the project aligned, avoid overdesign, and make decisions reviewable.

## Completed Milestones

### 2025-12 (Bootstrap And Core Features)

- Core NMPC solver brought up with configurable integration, barrier/regularization strategy, and solver status reporting.
- Python/SymPy `MiniModel` workflow established for symbolic modeling and C++ header generation.
- Early benchmark tooling and auto-tuning utilities added, along with snapshot/replay style debugging helpers.
- Soft constraints and basic nonlinear constraints integrated (foundation for obstacle-style problems).
- `MiniMatrix` (custom fixed-size backend) introduced and iterated, including early sparsity/fused-Riccati performance work.
- CI introduced with basic checks and a style gate.

### 2026-02 (Knot Refactor)

- `KnotPoint` refactor to clarify per-stage data layout and reduce coupling.
- Test suite refactor to improve regression coverage and reduce duplication.

### 2026-04 (Stabilization After Refactors)

- KnotPoint / Trajectory refactor stabilized with regression coverage.
- Solver public API surface tightened; configuration is centralized in `SolverConfig` via `set_config()` / `get_config()`.
- Snapshot/replay I/O moved away from raw `sizeof(SolverConfig)` dumps; format is explicit and includes soft-constraint state where required.
- Matrix backend selection wired end-to-end (`USE_EIGEN` vs `USE_CUSTOM_MATRIX`), with CI/test coverage on both.
- Zero-malloc discipline strengthened: SOC path and line-search filter avoid heap allocation, with tests and ASan-friendly hooks.
- Numerical robustness fixes landed (e.g., `fast_inverse()` guarded by SPD checks) and a microbenchmark added to quantify overhead.

## Near-Term (Next 2-6 Weeks)

- Establish an external `nmpc-bench` repository:
- Compare MiniSolver vs acados vs CasADi using **C/C++** interfaces (no Python runner for fairness).
- Use a small set of canonical NMPC scenarios and record both accuracy (reference solution) and runtime.
- Define and enforce correctness gates: MiniSolver must match a high-precision reference before optimizing iteration/time.

## Remaining Solver Hardening Plan (2026-05)

This plan tracks the remaining solver-hardening work after the May 2026
review pass. Each item should follow the project rule:

1. Add or tighten a focused test/benchmark that exposes the behavior.
2. Make the smallest code change that fixes the behavior.
3. Compare before/after correctness and runtime when performance can be affected.

For issue-level discovery, evidence, and resolution status, use
[Review Issue Ledger: 2026-05-02](reviews/review-fix-plan-2026-05-02.md).
Historical review files are preserved in `docs/reviews/` as evolution records;
fixed findings should be marked with evidence instead of deleted.

### May 2026 Hardening Batch

| Priority | Item | Status | Validation |
| --- | --- | --- | --- |
| P0 | Reject duplicate, invalid, keyword, and codegen-reserved MiniModel names instead of silently renaming them. | Implemented | `test_minimodel_generation` |
| P0 | Treat terminal stage as an x-only stage: coupled x/u cost and constraints are evaluated with terminal controls projected to zero. | Implemented | `test_minimodel_generation`, line-search terminal-control regression |
| P0 | Keep default solve path allocation-safe by disabling profiling and iteration logging by default. | Implemented | `test_memory` default-config allocation test |
| P0 | Make special quadratic-constraint codegen safer: unique temporary names plus basic numeric domain checks for `rhs` and `Q`. | Implemented | `test_minimodel_generation` |
| P0 | Require `dynamics_continuous()` for implicit integrator dispatch in custom models. | Implemented | `test_integrator` |
| P0 | Replace proportional "Armijo" merit check with standard directional-derivative Armijo and keep a dedicated benchmark. | Implemented | `test_line_search`, `merit_armijo_bench` |
| P0 | Expand zero-malloc coverage beyond the default config. | Implemented | Global `operator new` instrumentation over `MERIT`, `FILTER`, `NONE`, rollout on/off, SOC on/off, and long `max_iters` scenarios |
| P1 | Make `postsolve()` evaluate dual residuals on scratch data instead of mutating the active trajectory. | Implemented | `PostsolveDoesNotMutateActiveDirections` |
| P1 | Broaden solve-time finite checks beyond the first direction element. | Implemented | `StepRejectsNaNDirectionBeyondFirstKnot` |
| P1 | Reconfirm L2 soft-constraint residual semantics end-to-end. | Implemented | L2 convergence, interface-vs-manual comparison, filter residual regression, `merit_armijo_bench` |
| P2 | Document the actual Hessian approximation semantics. | Implemented | `OBJECTIVE_HESSIAN_ONLY` name added; `GAUSS_NEWTON` remains a compatibility alias |
| P2 | Document direction-refinement semantics. | Implemented | README/config comments clarify current behavior is defect-rollout correction, not full KKT iterative refinement |

### Next Hardening Items

| Priority | Item | Reason | Required evidence before landing |
| --- | --- | --- | --- |
| P2 | Design piecewise/ppoly support before implementing it. | Piecewise support is a MiniModel/codegen contract and should not be patched into the solver hot path ad hoc. Concrete benchmark-case impact belongs to MiniSolver-Bench. | Design note, generated-model tests, and nmpc-bench before/after data. |
| P2 | Continue MiniMatrix hot-path optimization only after benchmark gates are in place. | Matrix optimizations can easily increase complexity without improving full solver time. | Microbenchmark vs Eigen plus full NMPC benchmark before/after data. |
| P2 | Defer full KKT iterative refinement until constrained benchmarks justify it. | Current direction refinement only fixes dynamics defects; full KKT refinement is useful but couples to slack/dual, soft constraints, Mehrotra, SOC, and line search. | Red test or benchmark showing stale constraint/dual directions limit convergence or accuracy, then design `DirectionRefinementMode::FULL_KKT_ITERATIVE_REFINEMENT`. |

### Commit Discipline

Use separate commits for separate behavioral units:

- Codegen safety and terminal-stage semantics.
- Zero-malloc defaults and allocation tests.
- Implicit-integrator dispatch hard-fail behavior.
- Line-search semantic fixes and Armijo benchmark evidence.
- Documentation-only clarifications.

Do not mix solver semantics, matrix-kernel tuning, and benchmark asset changes in
the same commit unless a test proves they are inseparable.

## Medium-Term (Next 2-3 Months)

- Capability adoption from mature solvers:
- Treat [`architecture/solver-capability-adoption-plan.md`](architecture/solver-capability-adoption-plan.md)
  as the active filter for solver-landscape research.
- Prioritize scaling/normalization, warm-start strategy, and structured
  diagnostics before deeper RTI-lite or Riccati-mode work.
- Add solver profiles (`Reference`, `Default`, `Speed`, `Robust`) as
  `SolverConfig` presets, not as a new solver hierarchy.
- Keep RTI-lite conservative: track linearization age and refresh whenever
  safety gates fail.

- Matrix backend hardening:
- Keep `MiniMatrix` as an **NMPC-specialized** backend; do not grow it into a general linear algebra library.
- Every new kernel must have:
- A `vs Eigen` correctness test.
- A microbenchmark if it is on the hot path.
- Keep matrix dispatch policy centralized in `MatrixPolicy`; platform-specific
  threshold overrides are acceptable only when supported by target benchmarks.
- Embedded readiness:
- Reduce compile-time/binary-size overhead where possible (especially when Eigen is disabled).
- Harden determinism and real-time constraints (no hidden allocations, stable iteration behavior under neighboring problems).

## Design Guardrails

- Occam's razor: no new public APIs without a concrete use-case, tests, and a performance/correctness justification.
- "Benchmark-driven": performance claims must be backed by reproducible benchmark artifacts, not anecdotes.
- "Correctness-first": always establish a reference solution before tuning heuristics/regularization/line-search behavior.

## Known Limitations (Informed Defer)

- **Eigen-like automatic matrix tuning table is deferred.**
  MiniMatrix currently uses conservative compile-time threshold policies instead
  of an operation/size/platform tuning table. Building a true tuning table would
  require operation-specific policies, target grouping, benchmark data
  generation, and selection rules. Reopen only after real deployments show that
  threshold overrides are not enough.

- **Filter Pareto-frontier history is deferred.**
  `FilterLineSearch` now has `theta_max`, f/h-type switching, f-type Armijo, and
  h-type-only filter augmentation. The remaining theory gap is that filter
  history is still a fixed-capacity ring buffer rather than a Pareto-frontier
  certificate. See [ADR 0002](adr/0002-filter-line-search-switching.md).

- **Floating initial state / MHE support is deferred.**
  MiniSolver's current core is an NMPC-oriented multiple-shooting SQP/IPM path:
  the initial state is treated as measured problem data and the Riccati
  direction uses `dx0 = 0`. That is the correct default for real-time control.
  Making `x0` an optimization variable is a different profile: moving-horizon
  estimation (MHE), smoothing, or offline estimation/planning. It should not be
  added as a loose `floating_x0` switch in the NMPC path.

  A future MHE profile should introduce explicit initial-state semantics, for
  example:

  ```text
  InitialStateMode::FIXED                // current NMPC default
  InitialStateMode::ESTIMATED_WITH_PRIOR // MHE / smoothing
  ```

  `ESTIMATED_WITH_PRIOR` requires an arrival/prior cost
  `0.5 * (x0 - x_prior)^T P0 * (x0 - x_prior)`, which means carrying at least
  one additional `NX x NX` prior information/weight matrix plus the prior state.
  It should also support stage-0-only bounds or constraints on the estimated
  initial state, such as physical state limits, map/geometry consistency, or a
  confidence-region constraint around the prior. These constraints are MHE
  problem semantics and should not be mixed with the current NMPC
  `set_initial_state()` hard-fixing API.

  This profile also changes the Riccati boundary condition, warm-start
  semantics, and status/diagnostic interpretation. Reopen only after a concrete
  MHE or noisy-state estimation case exists, with tests proving that optimizing
  `x0` improves estimation quality without hiding invalid NMPC measurements.
