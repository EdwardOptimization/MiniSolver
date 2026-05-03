# Globalization And Mehrotra Theory Plan

Date: 2026-05-03

Status: design accepted, code changes must be split by item

Related:

- [`../adr/0002-filter-line-search-switching.md`](../adr/0002-filter-line-search-switching.md)
- [`solver-refactor-plan.md`](solver-refactor-plan.md)
- [`../reviews/review-fix-plan-2026-05-02-deep.md`](../reviews/review-fix-plan-2026-05-02-deep.md)

## Purpose

The deep review groups four remaining theory items:

- N-THEORY-1: Filter line-search gaps against the Waechter-Biegler route.
- N-THEORY-2: Fixed ring-buffer filter history instead of a Pareto frontier.
- N-THEORY-3: Mehrotra affine step uses one combined fraction-to-boundary
  instead of separate primal and dual affine step lengths.
- N-THEORY-6: Merit line search uses finite-difference directional derivative
  instead of an analytic directional derivative.

These are not one bug. They touch different solver phases and need separate red
tests and commits.

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

MiniSolver currently has:

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

Classification: theory-standard.

Why it is real:

- Standard Mehrotra computes separate affine primal and dual steps, then uses
  them to estimate the affine complementarity gap.
- MiniSolver currently uses one scalar fraction-to-boundary for `s`, `lambda`,
  `soft_s`, and `w-lambda`. This is safe but can be conservative because a
  primal-limited direction also shortens the dual affine step and vice versa.

Implementation shape:

```cpp
struct FractionToBoundaryResult {
    double primal = 1.0;
    double dual = 1.0;
    double combined() const { return std::min(primal, dual); }
};
```

Use:

- `primal` for `s` and L1 `soft_s`.
- `dual` for `lambda` and L1 `w-lambda`.
- `combined()` for accepted primal-dual step damping where one scalar is still
  required.
- `compute_affine_barrier_mu_()` should evaluate `s` and `soft_s` with
  `alpha_primal`, and `lambda` / `w-lambda` with `alpha_dual`.

Red test:

- Construct a small solver state where the affine primal step is strongly
  limited but the dual step is not.
- Verify `mu_aff` from split steps is lower than the current combined-step
  value and matches a direct recomputation using separate alphas.
- Verify L1 soft pair uses `soft_s + alpha_p * dsoft_s` and
  `w - (lambda + alpha_d * dlambda)`.

### N-THEORY-6: Analytic Merit Directional Derivative

Classification: theory-standard but implementation-sensitive.

Why it is real:

- Current finite-difference `dphi` costs an extra trial construction and model
  evaluation per search.
- It is also sensitive to `eps_alpha` and can misclassify nearly flat merit
  directions.
- acados computes the directional derivative analytically from cost,
  dynamics, and inequality contributions.

Implementation shape:

- Add an internal `MeritDirectionalDerivative` helper, not a public API.
- Compute directional derivative of:
  - cost: `q^T dx + r^T du` using already evaluated gradients;
  - hard/L1/L2 residual penalties using signs of current true residuals and
    linearized residual directions;
  - dynamics defect penalties using signs of current defects and
    `dx_next - A dx - B du`;
  - barrier terms for `s`, `soft_s`, and `w-lambda`.
- Keep finite-difference derivative behind a debug comparison flag only if a
  red test shows useful coverage; do not keep both paths in default hot logic.

Red tests:

- Compare analytic derivative against finite difference on a fixed small model
  for hard, L1, and L2 constraints.
- Cover rollout disabled first. Rollout-enabled merit derivative can be
  deferred unless the feature is used in benchmarks.
- Verify no extra model evaluation is needed to compute `dphi`.

### N-THEORY-1: Filter Theory Completion

Classification: design-required.

Sub-items:

1. Add `theta_max` rejection. This is a small, testable safety gate.
2. Add f-type/h-type classification and skip filter augmentation for f-type
   accepted steps.
3. Add the switching condition only after the f-type derivative or model
   decrease metric is available. This likely shares machinery with N-THEORY-6.

Implementation shape:

- Add `FilterAcceptanceKind { HType, FType }` as an internal result, not a user
  API.
- Keep `LineSearchType::FILTER` as the user-facing selection.
- Add `filter_theta_max_factor` / `filter_theta_min_factor` only if needed for
  behavior tuning; otherwise keep defaults internal first.

Red tests:

- `theta_max` rejects a trial that would otherwise satisfy a stale filter entry.
- f-type accepted step does not augment the filter.
- h-type accepted step still augments the filter.
- Switching condition is covered by a focused nonlinear constraint case or a
  benchmark evidence case before enabling it by default.

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

1. N-THEORY-3 split affine primal/dual step lengths. It is the smallest
   standard-route correctness/performance improvement.
2. N-THEORY-6 analytic merit directional derivative for merit line search.
3. N-THEORY-1a `theta_max` gate.
4. N-THEORY-1b f-type/h-type acceptance and filter augmentation rules.
5. N-THEORY-1c switching condition, sharing derivative/model-decrease machinery
   from N-THEORY-6.
6. N-THEORY-2 Pareto-frontier filter, after filter semantics are stable.

## Deferred Decisions

- Whether filter theory completion becomes a separate `GlobalizationMode` inside
  `SolverConfig` or remains the implementation of `LineSearchType::FILTER`.
  Default: keep it internal until a benchmark shows a need to select old vs new
  behavior.
- Whether merit analytic derivative should support rollout mode in the first
  pass. Default: multiple-shooting only first, because default MiniSolver uses
  `enable_line_search_rollout = false`.
- Whether Pareto frontier capacity should be fixed by `MAX_N`, `max_iters`, or a
  separate constant. Default: keep fixed-size storage, no solve-time allocation.
