# ADR 0002: Filter Line Search Switching Condition

Date: 2026-04-20

Status: **Implemented for f/h-type switching** — Pareto-frontier filter history
remains deferred

## Context

`FilterLineSearch::is_acceptable` (`include/minisolver/algorithms/line_search.h:442`)
uses the classical sufficient-decrease / filter acceptance rule from
Wächter-Biegler 2006 Eqn. (18):

```
θ(x_{k,α}) ≤ (1-γ_θ)·θ(x_k)   OR   φ(x_{k,α}) ≤ φ(x_k) - γ_φ·θ(x_k)
```

with `γ_θ = γ_φ = 1e-5`. When `θ(x_k)` is very small (iterate already
near-feasible), `γ_φ·θ_k` is microscopic, and the OR's second branch
degenerates: any barrier-objective improvement, however tiny, passes the
check. A candidate step that grows θ by 2× can therefore be accepted
silently.

### Observed symptom

On the chain_mass bench (multi-iter SQP), 4 of 30 closed-loop steps show a
deterministic 2.06× max-norm PrimInf jump at iter 9→10 — signature
identical across warm-started MPC steps 1–4:

```
Iter  PrimInf     Alpha     Action
 9    4.68e-5     0.262     accepted
10    9.63e-5     0.641     accepted  ← 2.06× max-norm jump
11    1.50e-5     1.000     accepted  (recovers)
```

Despite the overshoot, chain_mass converges: 25/25 bench success,
final max-viol ≈ 4e-11. The jump is not user-visible as a failure.

See `.claude/debug/chain-mass-priminf-overshoot-iter9-iter10/` (gitignored)
for the full Phase 1–3 case artifacts, including Phase 3 falsification
results that eliminated three naive fixes (f-type switch with wrong
threshold; γ tightening; absolute θ-growth cap) — each either failed to
address the OR degeneracy or broke unrelated tests by a scale-mismatch
between the acceptance code's SUM-norm θ and the oracle's MAX-norm
PrimInf.

## What IPOPT does (and we do not)

Per Wächter-Biegler 2006 §2.3, Eqn. (19)–(20), IPOPT does **not** fix
(18); it **switches away from (18) in the near-feasible regime** via a
two-part mechanism:

1. **Lazy-initialize thresholds on first iteration**:
   - `θ_min = 1e-4 · max(1, θ(x_0))`
   - `θ_max = 1e4  · max(1, θ(x_0))`

2. **Classify each candidate step as f-type or h-type**:
   - `f-type` ⟺ `θ(x_k) ≤ θ_min` AND the switching condition (Eqn. 19):

     ```
     ∇φᵀd < 0   AND   α·[-∇φᵀd]^{s_φ}  >  δ · θ_k^{s_θ}
     ```

     (defaults: `δ=1, s_θ=1.1, s_φ=2.3`)

   - `h-type` otherwise.

3. **Acceptance depends on type**:
   - f-type: **pure Armijo** on φ (Eqn. 20), filter ignored:

     ```
     φ(x_{k,α}) ≤ φ(x_k) + η_φ · α · ∇φᵀd     (η_φ = 1e-4)
     ```

   - h-type: filter + OR (Eqn. 18), unchanged from MiniSolver's current path.

4. **Filter seed with `θ_max` sentinel** (Eqn. 21): filter starts with
   `F₀ = {(θ, φ) : θ ≥ θ_max}`, enforcing a hard upper bound on θ
   regardless of φ.

5. **Filter augmented only on h-type acceptance** — f-type iterations do
   not add entries to the filter.

IPOPT source: `src/Algorithm/IpFilterLSAcceptor.cpp::IsFtype`,
`::ArmijoHolds`, `::IsAcceptableToCurrentIterate`.

### Norm choice

IPOPT uses `1-norm` (SUM) for θ by default, matching MiniSolver's
`compute_metrics`. Both projects default to 1-norm and expose the
norm as a user option. The MAX-norm `PrimInf` used in MiniSolver's iter
log is a display/diagnostic quantity, not the algorithm's acceptance
quantity — the two metrics can diverge and that is expected.

## Original Defer Analysis

This was the original reason for deferring the implementation. It is kept as
history because it documents why naive fixes were rejected.

1. **The observed chain_mass overshoot was not itself a failure.** An f-type step
   that temporarily grows the max-norm of one component is exactly what
   the switching mechanism allows. chain_mass reaches `Viol ≈ 4e-11`
   without intervention; the overshoot is an artifact of inspecting
   max-norm per-iter, not a convergence failure.

2. **MiniSolver's old formula handled h-type alone, not f-type.**
   That was a real architectural gap but had not yet demonstrated itself as a
   user-facing failure.

3. **Implementation cost was nontrivial** because it required a new `∇φᵀd`
   computation path:
   - Add `compute_grad_phi_times_d(active, N, mu, config)` that evaluates
     the directional derivative of the barrier objective along the search
     direction `(dx, du, ds, dsoft_s)` — needs contributions from:
       * cost gradient: `q·dx + r·du`
       * barrier gradient: `-μ · Σ(ds_i / s_i)` for hard constraints, plus
         L1-soft's `dsoft_s / soft_s` and `-dλ / (w - λ)` terms
       * L2-soft: `w · (g + s) · (C·dx + D·du + ds)`
   - Add private members `theta_min_`, `theta_max_` with lazy-init
   - Add `IsFtype(α)` helper
   - Branch `is_acceptable` on f-type vs h-type
   - Seed filter with `(θ_max, -∞)` sentinel on reset / on_barrier_update
   - Skip `filter.push_back(...)` on f-type acceptance
   - Unit tests covering both types

4. **Benefit/cost was not justified by data at the time.** No current bench
   case had been proven to fail because of the missing switching.

## Decision

Implement the switching condition inside `LineSearchType::FILTER`, not as a new
public strategy. The user-facing config remains `SolverConfig`; the filter
acceptor internally classifies accepted trial points as:

- **f-type**: near-feasible and satisfying the switching condition. The trial
  must pass Armijo sufficient decrease on the filter objective and does not
  augment the filter history.
- **h-type**: all other accepted filter steps. The existing sufficient-decrease
  and filter-entry checks apply, and accepted steps augment the filter.

The implementation keeps fixed-size storage and uses the already available
cost/barrier directional derivative. No solve-time allocation or public plugin
surface is added.

`N-THEORY-2` remains deferred: the filter history is still a fixed-capacity
ring buffer rather than a Pareto frontier. That should be revisited only after
the current f/h-type behavior has benchmark coverage on harder NMPC cases.

## Consequences

### Positive

- The near-feasible objective-progress path now follows the standard f-type
  Armijo route instead of the weaker h-type OR condition.
- f-type accepted steps no longer pollute the filter history.
- The existing h-type route remains unchanged for feasibility-driven steps.
- No new user-facing strategy object or plugin layer was added.

### Negative

- The ring-buffer filter is still a bounded engineering approximation, not a
  full Pareto-frontier certificate.
- Harder benchmarks may still need additional globalization tuning before this
  path should be claimed as production-grade.

## References

- Wächter, A., Biegler, L. T. (2006). "On the implementation of an
  interior-point filter line-search algorithm for large-scale nonlinear
  programming." *Math. Prog.* 106, 25-57.
  DOI: <https://doi.org/10.1007/s10107-004-0559-y>
  PDF: <https://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf>
  - Eqn. (18) sufficient decrease — current MiniSolver formula
  - Eqn. (19) switching condition
  - Eqn. (20) Armijo for f-type
  - Eqn. (21) θ_max filter sentinel
  - §2.3 algorithm statement; algorithm step A-5.4 branch

- IPOPT source (coin-or/Ipopt):
  <https://github.com/coin-or/Ipopt/blob/master/src/Algorithm/IpFilterLSAcceptor.cpp>
  - `IsFtype(...)` — the (19) implementation
  - `ArmijoHolds(...)` — the (20) implementation
  - `IsAcceptableToCurrentIterate(...)` — the (18) implementation
  - Lazy init: `theta_min_ = theta_min_fact_ * max(1, reference_theta_)`

- IPOPT options: <https://coin-or.github.io/Ipopt/OPTIONS.html>
  - `theta_min_fact = 1e-4`, `theta_max_fact = 1e4`, `eta_phi = 1e-8`,
    `s_phi = 2.3`, `s_theta = 1.1`, `delta = 1`, `gamma_theta = 1e-5`,
    `gamma_phi = 1e-8`, `alpha_min_frac = 0.05`,
    `constraint_violation_norm_type = "1-norm"` (default)
