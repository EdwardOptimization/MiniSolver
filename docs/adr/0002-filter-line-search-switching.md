# ADR 0002: Filter Line Search Switching Condition (Deferred)

Date: 2026-04-20

Status: **Deferred** — known gap, not implementing now

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

## Why defer

1. **Current behavior is IPOPT-algorithmically correct.** An f-type step
   that temporarily grows the max-norm of one component is exactly what
   the switching mechanism allows. chain_mass reaches `Viol ≈ 4e-11`
   without intervention; the overshoot is an artifact of inspecting
   max-norm per-iter, not a convergence failure.

2. **MiniSolver's current formula handles h-type alone, not f-type.**
   That is a real architectural gap but has not demonstrated itself as a
   user-facing failure. It may be the mechanism behind race_cars'
   9.4% INFEASIBLE rate (known via bench) or quadrotor_nav's precision
   gap vs acados — but this is unconfirmed.

3. **Implementation cost is substantial** (~30–50 LoC spread across
   `line_search.h`, `solver_options.h`, and requires a new `∇φᵀd`
   computation path that MiniSolver currently does not have):
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

4. **Benefit/cost not yet justified by data.** No current bench case has
   been proven to fail because of the missing switching — chain_mass
   succeeds, and the overshoot is IPOPT-legal.

## Decision

Do not implement the switching condition in the current iteration cycle.
Document the gap here. Revisit when one of the following triggers fires:

### Reopen triggers

- **race_cars 9.4% INFEASIBLE rate** is investigated and the failure
  cascade is traced to an acceptance decision that IPOPT's switching
  would have classified as f-type (i.e., we accepted a step in
  near-feasible regime that over-shot and triggered restoration / failure
  on the next iter).

- **quadrotor_nav precision gap vs acados** (`avg_abs_n = 4e-2` vs
  acados's `5e-5`) is investigated and traced to systematic iterate
  bias from OR-degenerate sufficient decrease in the near-feasible
  regime.

- **A new NMPC model** is added to the bench suite that exhibits a user-
  visible failure (not just a max-norm overshoot) attributable to
  this gap.

- **Future MiniSolver maintainers encounter a convergence bug** where
  the root-cause analysis matches the OR-degeneracy pattern.

## Consequences of keeping the status quo

### Positive

- No code to write, review, test.
- No new config options to explain to users.
- No risk of breaking the many currently-passing ctest/bench cases with
  a fix whose tradeoffs are not yet understood (Phase 3 showed all three
  naive fixes broke something).

### Negative

- Race_cars / quadrotor_nav gaps vs acados remain unexplained (may or may
  not be this gap; unknown until reopened).
- Future users who trace a convergence oscillation to this mechanism will
  need to rediscover the analysis (mitigation: this ADR).
- MiniSolver is not strictly "Wächter-Biegler-correct"; theoretical
  convergence guarantees for near-feasible regimes may not hold.

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
