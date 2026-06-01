# Globalization And Mehrotra Theory Plan

Date: 2026-05-03

Status: current-state ledger. N-THEORY-3, N-THEORY-6, and N-THEORY-1 are
implemented; N-THEORY-2 is deferred.

Related:

- [`../adr/0002-filter-line-search-switching.md`](../adr/0002-filter-line-search-switching.md)
- [`solver-refactor-plan.md`](solver-refactor-plan.md)
- [`../reviews/review-fix-plan-2026-05-02-deep.md`](../reviews/review-fix-plan-2026-05-02-deep.md)

## Purpose

The deep review originally grouped four theory items:

- N-THEORY-1: Filter line-search gaps against the Waechter-Biegler route
  (implemented except Pareto-frontier history).
- N-THEORY-2: Fixed ring-buffer filter history instead of a Pareto frontier
  (deferred).
- N-THEORY-3: Mehrotra affine step used one combined fraction-to-boundary
  instead of separate primal and dual affine step lengths (implemented).
- N-THEORY-6: Merit line search used finite-difference directional derivative
  instead of an analytic directional derivative (implemented for the default
  multiple-shooting merit path).

These were not one bug. They touched different solver phases and were handled
with separate tests and commits. This file now records the current state rather
than an open implementation queue.

## Source Survey

Local source survey:

- Ipopt's filter acceptor keeps `theta_max`, `theta_min`, `gamma_phi`,
  `gamma_theta`, and `alpha_min_frac`, rejects points whose violation exceeds
  `theta_max`, and augments the filter through a dedicated `AugmentFilter()`
  path rather than always adding every accepted trial.
- acados' merit backtracking computes an analytic merit directional derivative
  from already evaluated cost gradients, dynamics Jacobians, constraint
  Jacobians, and merit weights before applying Armijo sufficient descent.
- acados/HPIPM statistics expose `alpha_prim_aff`, `alpha_dual_aff`, `mu_aff`,
  `sigma`, `alpha_prim`, and `alpha_dual`, matching the standard primal-dual IPM
  distinction between primal and dual step lengths.

Historical baseline before the follow-up work:

- `FilterLineSearch::is_acceptable()` with sufficient decrease plus filter
  entries, but no `theta_max` sentinel, no f-type/h-type switch, and no f-type
  filter-augmentation skip.
- A fixed 1024-entry ring buffer for filter entries.
- `compute_fraction_to_boundary_()` returning one scalar and using that scalar
  for both `alpha_aff` and `mu_aff`.
- `MeritLineSearch` estimating `dphi` by a tiny trial-point finite difference.

## Design Rules

1. Do not add a public OOP globalization plugin framework.
2. Add config fields only for built-in behavior users actually need to select.
3. Resolve choices at the build/config boundary where possible.
4. Keep line-search hot paths fixed-size and zero-malloc.
5. Add one red test or benchmark per item before changing behavior.
6. Do not mix filter, merit, and Mehrotra changes in one commit.

## Item Plan

### N-THEORY-3: Split Mehrotra Affine Step Lengths

Status: implemented.

Current behavior:

- MiniSolver computes separate affine primal and dual step lengths.
- `compute_affine_barrier_mu_()` evaluates primal quantities with
  `alpha_primal_aff` and dual quantities with `alpha_dual_aff`.
- Accepted primal-dual damping may still use a combined alpha when a single
  interior step bound is required.

Implementation shape:

```cpp
struct FractionToBoundaryResult {
    double primal = 1.0;
    double dual = 1.0;
    double combined() const { return std::min(primal, dual); }
};
```

Current use:

- `primal` for `s` and L1 `soft_s`.
- `dual` for `lambda` and L1 `w-lambda`.
- `combined()` for accepted primal-dual step damping where one scalar is still
  required.
- `compute_affine_barrier_mu_()` evaluates `s` and `soft_s` with
  `alpha_primal`, and `lambda` / `w-lambda` with `alpha_dual`.

Evidence:

- `tests/test_bugfixes.cpp::BugfixTest.MehrotraUsesSeparateAffinePrimalAndDualStepLengths`.
- `tests/test_bugfixes.cpp::BugfixTest.MehrotraMuAffIncludesL1SoftPair`.

### N-THEORY-6: Analytic Merit Directional Derivative

Status: implemented for the default multiple-shooting merit path.

Current behavior:

- Merit line search computes the directional derivative analytically from
  already available cost, dynamics, and constraint quantities.
- The default hot path does not construct an extra finite-difference probe to
  estimate `dphi`.

Implementation shape:

- Keep the derivative helper internal, not a public API.
- Compute directional derivative of:
  - cost: `q^T dx + r^T du` using already evaluated gradients;
  - hard/L1/L2 residual penalties using signs of current true residuals and
    linearized residual directions;
  - dynamics defect penalties using signs of current defects and
    `dx_next - A dx - B du`;
  - barrier terms for `s`, `soft_s`, and `w-lambda`.

Evidence:

- `tests/test_line_search.cpp::LineSearchTest.MeritArmijoDoesNotBuildFiniteDifferenceProbe`.
- `tests/test_line_search.cpp::LineSearchTest.MeritNonFiniteDphiReturnsNumericalError`.

### N-THEORY-1: Filter Theory Completion

Status: implemented except Pareto-frontier history.

Sub-items:

1. Done: `theta_max` rejection.
2. Done: f-type/h-type classification and skip filter augmentation for f-type
   accepted steps.
3. Done: switching condition using the filter objective directional derivative.

Current implementation shape:

- `FilterAcceptanceKind { HType, FType }` is an internal result, not a user API.
- Keep `LineSearchType::FILTER` as the user-facing selection.
- `filter_theta_max_factor` is a config field for the implemented theta gate;
  `theta_min` remains an internal constant.

Evidence:

- `LineSearchTest.FilterRejectsTrialAboveThetaMax`.
- `LineSearchTest.FilterFTypeUsesArmijoAndDoesNotAugmentFilter`.
- `LineSearchTest.FilterHTypeAcceptanceStillAugmentsFilter`.

### N-THEORY-2: Pareto Frontier Filter

Classification: design-required, defer until N-THEORY-1 is stable.

Why not first:

- The fixed ring buffer is theoretically weaker, but typical NMPC horizons and
  iteration counts will not hit 1024 accepted filter entries in one solve.
- A correct Pareto frontier needs a clear filter-entry dominance convention and
  must coexist with f-type filter-skip. Implementing it before N-THEORY-1 risks
  churn.

Implementation shape:

- Replace append/overwrite with fixed-capacity Pareto pruning.
- Reject insertion if the new entry is dominated.
- Remove entries dominated by the new entry.
- If capacity is still exhausted, return a visible degraded status or fall back
  to restoration; do not silently overwrite certificate-relevant entries.

Red test:

- Construct an artificial sequence exceeding old capacity and verify acceptance
  is invariant to insertion order for non-dominated entries.

## Recommended Order

1. Done: N-THEORY-3 split affine primal/dual step lengths.
2. Done: N-THEORY-6 analytic merit directional derivative for merit line
   search.
3. Done: N-THEORY-1a `theta_max` gate.
4. Done: N-THEORY-1b f-type/h-type acceptance and filter augmentation rules.
5. Done: N-THEORY-1c switching condition using derivative machinery from
   N-THEORY-6.
6. Deferred: N-THEORY-2 Pareto-frontier filter, after more hard NMPC benchmark
   evidence.

## Deferred Decisions

- Whether the fixed-capacity ring buffer should become a Pareto-frontier filter
  history. Default: keep ring-buffer storage until a benchmark shows the
  capacity/certificate limitation matters in practice.
- Whether merit analytic derivative should support rollout mode in the first
  pass. Default: multiple-shooting only first, because default MiniSolver uses
  `enable_line_search_rollout = false`.
- Whether Pareto frontier capacity should be fixed by `MAX_N`, `max_iters`, or a
  separate constant. Default: keep fixed-size storage, no solve-time allocation.
