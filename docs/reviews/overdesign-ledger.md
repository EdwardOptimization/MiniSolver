# Overdesign Ledger

This ledger records proposals that looked safer or more complete at first pass,
but needed a decision about whether they protect a real MiniSolver invariant or
unnecessarily constrain modeling freedom.

Use this file when a review suggests a broad validation rule, new config knob,
new public API, or theory feature without a concrete failing case.

## Decision Template

```text
ID:
Proposal:
Source:
Category: hard-invariant | algorithm-convention | modeling-choice | product-boundary
Decision: keep | modify | docs-only | defer | reject
Reason:
Evidence:
```

## Decisions

### OD-001: Positive `dt` Validation

Proposal: Reject `default_dt <= 0` and `set_dt(...) <= 0`.

Category: modeling-choice

Decision: reject

Reason: `dt` is model semantics, not a solver-core invariant. Users may encode
static stages, pseudo-time, normalized horizon parameters, reverse-time
constructions, or custom integrator scales. MiniSolver should reject non-finite
values, but should not globally require positive time steps.

Evidence: Current `SolverConfig::default_dt`, `set_dt(double)`, and
`set_dt(vector)` require finite values only. This is intentional.

### OD-002: Linear Solve Retry Naming

Proposal: Treat `inertia_max_retries = 0` as "no retry".

Category: hard-invariant

Decision: modify

Reason: The implementation loop used the field as a total attempt count, not an
extra retry count. A value of zero skipped the linear solve entirely. The field
was renamed to `linear_solve_max_attempts`, requires `>= 1`, and documents that
`1` means exactly one attempt with no retry.

Evidence: `445cb0c fix: tighten snapshot and linear solve config contracts`.
### OD-003: Mandatory Handwritten Model Fingerprint

Proposal: Require every handwritten model to define
`static constexpr std::uint64_t model_fingerprint`.

Category: product-boundary

Decision: docs-only

Reason: Generated MiniModel headers already provide a strong fingerprint.
Handwritten models should be encouraged to define one for reliable replay, but
forcing it would increase adoption friction and is not required for normal solve
correctness.

Evidence: `docs/architecture/snapshot-replay-design.md` documents the explicit
fingerprint contract and metadata fallback limitation.

### OD-004: Unscaled Feasibility Status Knob

Proposal: Add a config option such as
`require_unscaled_feasibility_for_feasible_status`.

Category: product-boundary

Decision: docs-only

Reason: `SolverStatus::FEASIBLE` uses the solver's internal scaled feasibility
metric by design. `SolverInfo::unscaled_primal_inf` already reports model-unit
feasibility. Adding a new status knob would increase status semantics complexity
before benchmarks prove the need.

Evidence: `docs/architecture/scaling-normalization-design.md` and
`docs/architecture/solver-status-semantics.md` document the scaled/unscaled
contract.

### OD-005: Auto-Disable Defect Rollout Refinement With Constraints

Proposal: Automatically disable dynamics-defect rollout refinement when active
inequality or soft constraints are present.

Category: algorithm-convention

Decision: docs-only

Reason: Defect rollout refinement is not full KKT iterative refinement and can
leave slack/dual directions less consistent in constrained cases. Automatically
disabling it is too strong because users may still find it useful for
dynamics-dominant NMPC problems. Prefer documentation and diagnostics until a
benchmark or failing case justifies a stronger policy.

Evidence: `docs/architecture/solver-refactor-plan.md` records full KKT
refinement as deferred. `tests/test_features.cpp::FeaturesTest.
DefectRolloutRefinementKeepsConstrainedDirectionConsistent` anchors the
runtime contract that with rollout refinement enabled and the upper control
bound strongly active for the majority of the horizon, the solver still
reaches OPTIMAL/FEASIBLE without violating bounds, dropping interior
slacks, or producing infinite/negative duals.

### OD-006: Full KKT Iterative Refinement

Proposal: Implement full KKT iterative refinement now.

Category: algorithm-convention

Decision: defer

Reason: Full KKT refinement couples to Mehrotra, SOC, line search, slack/dual
recovery, and linear solver residual contracts. It should be driven by a
constrained benchmark failure, not added as a speculative feature.

Evidence: Current `DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT` is
explicitly scoped as dynamics-defect rollout, not full KKT refinement.

### OD-007: Pareto Frontier Filter

Proposal: Replace the current bounded filter history with a full Pareto-frontier
filter implementation.

Category: algorithm-convention

Decision: defer

Reason: The current filter already covers the main f-type/h-type and switching
behavior needed for globalization hardening. Pareto-frontier management is a
separate theory pass and should require a concrete filter failure case.

Evidence: `docs/architecture/globalization-mehrotra-theory-plan.md` keeps deeper
filter variants as a later pass.

### OD-008: Embedded Fixed-Buffer Logger/Profile

Proposal: Convert diagnostics/logging/profiling to a full embedded fixed-buffer
profile now.

Category: product-boundary

Decision: defer

Reason: This is a release/embedded-product profile involving logging sinks,
iostream boundaries, exception policy, memory ownership, binary size, and target
toolchains. It should not block solver-core correctness hardening.

Evidence: `docs/architecture/api-logger-boundary-design.md` tracks the logger
boundary as a separate release-phase concern.

### OD-009: Blanket Positivity Validation For All Numeric Fields

Proposal: Require every tolerance, penalty, scale, and heuristic value to be
strictly positive.

Category: hard-invariant

Decision: modify

Reason: Positivity should be enforced only when a field is a divisor, floor,
barrier/globalization invariant, or control-flow bound. Some numeric values can
legitimately be zero to disable or neutralize a behavior. Avoid blanket
validation that turns modeling or tuning choices into solver-core constraints.

Evidence: Existing validation now targets specific invariant-bearing fields
such as line-search factors, barrier floors, `linear_solve_max_attempts`, and
core tolerances.

### OD-010: Snapshot Save-Side Strictness

Proposal: Allow `save_snapshot()` to write invalid forensic snapshots, while
`load_case()` rejects restoring them.

Category: product-boundary

Decision: reject

Reason: MiniSolver snapshot I/O is a strict replay/debug protocol, not a
forensic dump of arbitrary invalid memory. Public `save_snapshot()` should not
write files that `load_case()` would reject for invalid status or runtime
metadata. Corrupt-file tests should patch valid binary snapshots directly.

Evidence: `445cb0c fix: tighten snapshot and linear solve config contracts`.
