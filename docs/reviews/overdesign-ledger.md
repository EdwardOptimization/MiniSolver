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

Decision: modify (shipped as scoped dynamics-defect iterative variant)

Reason: Full KKT refinement still couples to Mehrotra, SOC, line search,
slack/dual recovery, and linear-solver residual contracts, so a primal-and-dual
iterative refinement remains deferred. What was shipped is strictly narrower:
`DirectionRefinementMode::FULL_KKT_ITERATIVE_REFINEMENT` iterates the existing
dynamics-defect rollout up to `direction_refinement_max_passes` times or until
the rollout defect drops below `direction_refinement_tol`, and auto-degrades to
a single pass whenever active inequality duals are present so the OD-005
dual-consistency hazard is not re-amplified.

Evidence: `tests/test_direction_refinement.cpp` pins the contract:
unconstrained problems may consume up to `iterations * max_passes` primal
passes, constrained problems are pinned to one pass per iteration, and
`SolverInfo::direction_refinement_passes` /
`SolverInfo::direction_refinement_last_defect` expose the realized work.
True primal-dual KKT iterative refinement is still gated on a constrained
benchmark failure that the scoped variant cannot resolve.

### OD-007: Pareto Frontier Filter

Proposal: Replace the current bounded filter history with a full Pareto-frontier
filter implementation.

Category: algorithm-convention

Decision: modify (shipped as scoped Pareto-pruning policy)

Reason: A full IPOPT-style filter rewrite still requires a concrete failure
case, but the legacy circular-buffer policy was carrying redundant entries
that did not strengthen the filter and could only be evicted by FIFO once the
1024-slot history was exhausted. The shipped Pareto-pruning rule is strictly
contained: it never accepts a trial the IPOPT filter rule would reject
(because every dominated entry's forbidden region is a subset of the
dominating entry's), and it eliminates two failure modes — unbounded history
growth on strictly improving solves and silent FIFO eviction of entries that
may still matter.

The first revision used plain Pareto on (theta, phi) and claimed correctness
only for `gamma_phi = 0`. With non-zero `gamma_phi`, plain Pareto silently
relaxed the filter forbidden region: it both over-pruned existing entries
and dropped new entries as "redundant" when they were not. The corrected
implementation compares in (theta, psi) space where `psi(e) = phi_e -
gamma_phi * theta_e`. `Forbidden(e_a) ⊇ Forbidden(e_b)` iff
`theta_a ≤ theta_b AND psi(e_a) ≤ psi(e_b)`. For `gamma_phi = 0` this
collapses to plain Pareto, so the original collapse-on-improvement contract
is preserved.

Evidence: `tests/test_filter_pareto.cpp` (collapse contract,
`MeritLineSearch` zero-diagnostic, `SolverInfo::reset` clearing,
`NonZeroGammaPhiPreservesParetoIncomparableEntry` and
`NonZeroGammaPhiPrunesEntriesDominatedInPsiSpace` red tests for the
gamma_phi > 0 dominance bug, `ZeroGammaPhiBehavesAsPlainPareto` defense
test) plus
`test_line_search.FilterHistoryParetoCollapsesMonotonicallyImprovingSequence`.
`LineSearchResult::filter_entries_pruned` / `filter_redundant_inserts` /
`filter_size_after` and the cumulative `SolverInfo::filter_*` counters
expose the realized pruning work. The Pareto insertion is exposed via
`FilterLineSearch::try_insert_h_type_for_testing` so tests can drive the
dominance contract without simulating a full solver iteration.

A full Pareto-frontier filter (variable-margin tracking, dominance
arbitration on alternate metrics) is still gated on a concrete filter
failure case.

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

### OD-011: Adaptive Restoration Penalty Rho

Proposal: Replace the hardcoded `rho = 1000.0` quadratic-penalty rho in
`feasibility_restoration()` with a multi-scale adaptive policy that retunes
rho per restoration sub-iteration based on the live constraint violation.

Category: algorithm-convention

Decision: modify (shipped behind opt-in mode)

Reason: A single hardcoded rho silently couples the augmented Hessian
condition number to the magnitude of the user's constraint scaling. When
violation is large the augmented Hessian becomes badly conditioned; when
violation is small the penalty under-pulls. The fix is well-bounded:
`SolverConfig::RestorationPenaltyMode { FIXED, VIOLATION_ADAPTIVE }` plus
clamp parameters (`restoration_rho_init`, `restoration_rho_min`,
`restoration_rho_max`, `restoration_rho_violation_floor`).
`FIXED` (default) keeps the legacy 1000.0 behaviour bit-for-bit by reading
`restoration_rho_init` directly. `VIOLATION_ADAPTIVE` selects rho per
restoration sub-iteration as
`clamp(rho_init / max(theta_inf, floor), rho_min, rho_max)`, exposing the
realized work via `SolverInfo::restoration_rho_min_used`,
`restoration_rho_max_used`, and `restoration_rho_adaptive_steps`.

Evidence: `tests/test_restoration_penalty.cpp` pins the defaults, the
validation gates (rho_init / rho_min / rho_max / violation_floor positivity,
`rho_max >= rho_min`, enum membership), and the `SolverInfo::reset` contract.
`tests/test_bugfixes.cpp::AdaptiveRestorationRhoUsesGValOnly` pins the
metric/penalty consistency contract: the adaptive rho is sized from
`||g_val||_inf`, not `||g_val + s||_inf`, so it matches the residual the
linearised penalty actually penalises (`q += rho * C^T * g_val`,
`r += rho * D^T * g_val`). An earlier revision used `g_val + s`, which
silently undersized rho whenever s and g_val partially cancelled in
magnitude; the test contrives a state with `|g_val|=10`, `s=10` to make
the divergence detectable (rho 100 vs the buggy 50).

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
