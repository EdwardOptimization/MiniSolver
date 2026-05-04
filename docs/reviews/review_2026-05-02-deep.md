# MiniSolver Deep Review (14-Dimension Coverage)

**Date**: 2026-05-02
**Reviewer**: Single-pass review (Opus 4.7), no subagents
**Scope**: All 14 NMPC-solver-quality dimensions, follow-up to the 2026-05-01/02 static reviews
**Baseline**: `ctest --test-dir .build` shows 25/25 pass (Eigen, Release)
**Code base**: `include/` 8196 lines, `tests/` 9010 lines (158 tests across 22 files), `tools/` 12 benchmarks

> Companion documents:
> - [review_2026-05-01.md](review_2026-05-01.md) — module-level static review (4 deep-dive agents)
> - [review_2026-05-02.md](review_2026-05-02.md) — solver-refactor static review (6 deep-dive + 4 verification agents)
> - [review-fix-plan-2026-05-02.md](review-fix-plan-2026-05-02.md) — static-review fix ledger
> - [review-fix-plan-2026-05-02-deep.md](review-fix-plan-2026-05-02-deep.md) — this review's fix ledger
> - [solver-capability-adoption-plan.md](../architecture/solver-capability-adoption-plan.md) — capability roadmap
> - [ADR 0002](../adr/0002-filter-line-search-switching.md) — filter switching condition (deferred)

---

## Methodology

This review intentionally avoids re-doing the static code walk-through that the 2026-05-01/02 reviews already covered. Instead it focuses on the 6-8 dimensions those reviews did not deeply cover (AD correctness depth, embedded-deployment reality, API error quality, WCET evidence, scaling absence, golden-reference comparison, property-based testing, modeling unit checks) while doing follow-up verification on the 6 dimensions they did cover.

Each finding gives three things: evidence (source citation, grep result, prior-review cross-reference), judgment, and a priority. The companion fix ledger ([review-fix-plan-2026-05-02-deep.md](review-fix-plan-2026-05-02-deep.md)) restates findings in the project's standard ledger format with status / evidence path / next action columns.

## Coverage Matrix

| Dimension | Prior coverage | This review status | New findings |
| --- | --- | --- | --- |
| 1. Numerical stability | Partial (5/2 L1, L2, L9, L11) | Follow-up + new | 4 |
| 2. Convergence / Status layering | None (status semantics not audited) | New | 5 |
| 3. Modeling soundness (units, scaling, DoF) | None (ADR'd as deferred) | New | 3 |
| 4. Real-time / RT safety | Strong (5/2 P0/P1 zero-malloc) | Follow-up | 2 |
| 5. Test coverage (numerical-software view) | Partial (test list only) | New | 6 |
| 6. Observability / Diagnostics | None (ADR'd as deferred) | New | 3 |
| 7. API ergonomics / error quality | None | New | 3 |
| 8. Degradation modes | Partial (5/2 GPU, Riccati fallback) | Follow-up | 1 |
| 9. Embedded portability reality | Partial (5/2 -ffast-math opt-in) | New (deeper) | 2 |
| 10. AD correctness | Partial (FD verification existence noted) | Follow-up + new | 2 |
| 11. Sparse-structure exploitation | Partial (5/1 implicit-midpoint pattern) | Follow-up | 0 (existing finding still valid) |
| 12. Numerical precision strategy | None | New | 2 |
| 13. Dependencies & license | None | New | 2 |
| 14. Theory ↔ implementation alignment | Partial (ADR 0002 only) | New (deeper) | 6 |

Total: 41 new findings, 1 corrected interpretation of prior coverage.

---

## Overall Assessment

### Strengths

- The 5/1 and 5/2 reviews delivered. The 22+ confirmed findings from those passes are largely fixed (`review-fix-plan-2026-05-02.md` shows all P0/P1 closed, most P2 either fixed or deferred-with-design). This review confirms those fixes hold.
- The IPM core math is well-implemented: Mehrotra predictor-corrector, filter line search, Riccati backward/forward sweep, dual recovery for hard / L1-soft / L2-soft constraints, fast-inverse SPD path with Sylvester checks for 1×1/2×2/3×3, defect correction for multiple shooting. The reviewed code matches standard formulations from Wächter-Biegler 2006 and Nocedal-Wright Ch. 16 wherever it implements them.
- The zero-malloc enforcement is genuinely strong. [`tests/test_memory.cpp`](../../tests/test_memory.cpp) overrides global `operator new` and verifies allocation count = 0 across 6 test cases including (FILTER × MERIT × NONE) × rollout × SOC × max_iters scenarios, set_config-then-solve, isolated SOC path, and 100-iteration implicit integrator loops.
- Code generation (`python/minisolver/MiniModel.py`, 1484 lines) is hardened: identifier validation rejects C++ keywords, codegen-reserved names, duplicates; terminal-stage projection separates x-only and x+u terms; fused Riccati kernels with CSE.
- ADR/review/roadmap discipline is unusual for an open-source numerical project: each major design choice has an ADR (4 files), each review pass produces a dated artifact (`review_2026-05-01.md`, `review_2026-05-02.md`), and gap backlogs (`gap_backlog.md`, `test-coverage-gaps-2026-05-02.md`) cross-link to red tests.
- CI matrix covers Eigen × CustomMatrix × Release × Debug = 4 jobs, with Debug enabling ASan + UBSan (`-fsanitize=address,undefined -fno-omit-frame-pointer`). Style is gated by clang-format-16 pinned version.
- Existing FD verification of 1st-order generated derivatives is comprehensive: [`tests/test_solver_quality.cpp`](../../tests/test_solver_quality.cpp) lines 309-431 cross-check generated A, B, C, D, q, r against centered finite differences with tolerance 1e-4 on CarModel.
- Implicit integrator (Backward Euler, Implicit Midpoint, Gauss-Legendre 2-stage) discrete Jacobian derivation via implicit function theorem is correct; A/B FD-verified by `ImplicitIntegratorTest::JacobiansMatchFiniteDifferenceForAllImplicitSchemes`.

### Risks (in order of severity)

- **Embedded-deployment claim is significantly overstated relative to current code**. README markets "Compile with `-O3`. No external libraries required" and "Embedded Safety", but 11 headers `#include <iostream>`, 4 sites `throw std::invalid_argument`, `solver.h` uses `std::unordered_map<std::string, int>` for name lookup, and there is no ARM cross-compile job, no `-fno-rtti -fno-exceptions` build option, no binary-size measurement target. (Dim 9, P0)
- **`SolverStatus` has only 5 values, missing standard NLP solver status layering** (`MAX_ITER`, `UNBOUNDED`, `LINEAR_SOLVE_FAILED`, `RESTORATION_FAILED`, `INVALID_INPUT`). When max_iters elapses with primal_inf still above tolerance, the user receives `INFEASIBLE` even though the problem may be feasible — they cannot distinguish "increase max_iters" from "problem unsolvable". `tests/test_status.cpp:96-98` explicitly comments expecting MAX_ITER but the enum doesn't have it. (Dim 2, P0)
- **Riccati / KKT solve has no inertia detection**. Standard primal-dual IPM (IPOPT, HPIPM) verifies the inertia of the augmented system equals (n_positive=variables, n_negative=equality_constraints+slacks, n_zero=0); failure means the iterate is a saddle, not a minimizer. MiniSolver only retries with increased regularization on SPD failure, never checks inertia. The code may converge to non-minimizers without detection. (Dim 14, P0)
- **`Restoration` silently skips quadratic feasibility penalty under `BarrierStrategy::MEHROTRA`** ([`solver.h`](../../include/minisolver/solver/solver.h) line 1317 `if (config.barrier_strategy != BarrierStrategy::MEHROTRA)`). For Mehrotra users, restoration becomes a cost-only solve with identity Q and zero q — it does **not** push toward feasibility. Case-level impact belongs to MiniSolver-Bench; this repository should track only the generic solver bug. (Dim 14, P0)
- **Problem-level scaling is completely absent** — `grep scaling` in `solver.h` returns 0 hits. This is the headline P0 of [`solver-capability-adoption-plan.md`](../architecture/solver-capability-adoption-plan.md) (Ruiz equilibration / Ipopt NLP scaling). Without scaling, NMPC problems mixing meters / radians / normalized forces have constraint rows differing by 10^4 in natural scale all contributing equally to filter / merit / KKT residual checks. (Dim 3, P0)
- **Diagnostics surface is far below capability-adoption-plan baseline**. `SolverContext` exposes only `{mu, reg, current_iter, slack_reset_consecutive_count, last_prim_inf, last_dual_inf, last_alpha, last_mu_aff, last_alpha_aff}`. Missing: requested vs actual backend, degraded-fallback flag (Riccati small-NU freeze), linearization_age, SOC counts, restoration counts, warm-start mode/repaired flag, scaling state, linear solver factorization status, regularization escalation count, scaled vs unscaled residuals. (Dim 6, P1; ADR'd as deferred)
- **0 property-based tests, 0 fuzzing harnesses across 158 tests**. `grep -i "rapidcheck|libfuzzer|hypothesis|fuzz"` returns 0. Numerical software typically benefits from property tests like "any feasible QP solved by solver must satisfy KKT", which catches sign / dual-recovery / barrier-update bugs that fixed inputs miss. (Dim 5, P1)
- **Filter line search has not just the ADR'd missing switching condition**, but also two other Wächter-Biegler 2006 §2.3 elements: (a) no `θ_max` sentinel filter seed (Eqn 21), (b) f-type acceptance still augments filter. ADR 0002 documents only the switching condition gap. (Dim 14, P1)
- **Filter ring buffer overwrites oldest entries after 1024 accepted steps** ([`line_search.h`](../../include/minisolver/algorithms/line_search.h) lines 681-687). Standard filter implementations use a Pareto-frontier list, not a fixed-cap ring; overwriting breaks the formal "monotone over all history" property of the acceptance certificate. (Dim 14, P1)
- **Generated 2nd-order Hessians (Q, R, H) are not FD-verified** even though 1st-order are. Sign / factor-of-two bugs in symbolic Hessian generation are a classic codegen failure mode that the current test suite cannot catch. (Dim 5/10, P1)

---

## Verified Findings

### Dimension 1: Numerical Stability

#### N-NUM-1 (P1) — Mehrotra `update_mu` divides by `current_mu` without zero guard

[`barrier_update.h`](../../include/minisolver/algorithms/barrier_update.h) line 26:

```20:38:include/minisolver/algorithms/barrier_update.h
        case BarrierStrategy::MEHROTRA: {
            double ratio = avg_complementarity_gap / current_mu;
            if (ratio > 1.0) {
                ratio = 1.0;
            }
            double sigma = std::pow(ratio, 3);
            if (sigma < 0.05) {
                sigma = 0.05;
            }
            if (sigma > 0.8) {
                sigma = 0.8;
            }
            return std::max(config.mu_final, current_mu * sigma);
        }
```

When `current_mu == 0`, `ratio = avg / 0 = +Inf` (or NaN if `avg == 0` too). `Inf > 1.0` is true so `ratio = 1.0` clamps and recovers — but `0 / 0 = NaN`, and `NaN > 1.0` is false. `std::pow(NaN, 3) = NaN`, sigma comparison cascade yields NaN, `current_mu * NaN = NaN`, `std::max(mu_final, NaN)` is implementation-defined.

The companion `mehrotra_target_mu` (line 47) has the same `mu_aff / mu_curr` division and was patched in 5/2 via `if (sigma > 1.0) sigma = 1.0; if (sigma < 1e-4) sigma = 1e-4;` clamping. The 5/2 fix covers the `mu_aff > 0, mu_curr == 0` branch (Inf → 1.0) but not `mu_aff == 0, mu_curr == 0` (NaN). `update_mu` was not given the same treatment.

Test `BarrierResidualContractTest.MehrotraTargetMuHandlesZeroCurrentMu` (added in 5/2) covers `mehrotra_target_mu` only. Add equivalent regression for `update_mu`.

#### N-NUM-2 (P1) — `recover_dual_search_directions` does not floor `lam_i` before division

[`riccati.h`](../../include/minisolver/solver/riccati.h) line 270:

```222:284:include/minisolver/solver/riccati.h
    for (int i = 0; i < Knot::NC; ++i) {
        double s_i = kp.s(i);
        if (s_i < config.min_barrier_slack) {
            s_i = config.min_barrier_slack;
        }
        ...
        double lam_i = kp.lam(i);
        ...
        if (type == 1 && w > 1e-6) { // L1 Soft
            ...
            kp.dlam(i) = dlam;
            kp.ds(i) = (-r_y - s_i * dlam) / lam_i;          // <-- lam_i divisor, no floor
            kp.dsoft_s(i) = -(r_z - soft_s_i * dlam) / soft_dual_i;
        } else if (type == 2 && w > 1e-6) { // L2 Soft
            ...
        } else { // Hard
            double r_prim = g_val_i + s_i;
            double term_rhs = -r_y + lam_i * (r_prim + constraint_step(i));
            kp.dlam(i) = (1.0 / s_i) * term_rhs;             // <-- s_i is floored, OK
            kp.ds(i) = -r_prim - constraint_step(i);
        }
    }
```

`s_i` is floored to `min_barrier_slack` (line 224-226), but `lam_i` is read raw on line 237. In the L1 Soft branch, `lam_i` becomes a divisor on line 270 (`kp.ds(i) = (-r_y - s_i * dlam) / lam_i`). The `fraction_to_boundary` upstream guarantees `lam_i > 0` after a step, but during dual recovery in the *first* iteration with a degenerate initial guess, or after Mehrotra affine step when `lam` collapses near zero, there is no defensive floor. `min_barrier_slack = 1e-12` is the documented invariant for slacks but the code does not extend it to duals.

Companion contract: `has_valid_primal_dual_guess` ([`solver.h`](../../include/minisolver/solver/solver.h) lines 1240, 1245) does check `lam(i) > 0.0`, but only at warm-start entry, not during the iteration loop.

#### N-NUM-3 (P2) — Magic-number cluster in `InitializationKernel`

[`initialization.h`](../../include/minisolver/algorithms/initialization.h):

```33:75:include/minisolver/algorithms/initialization.h
        if (type == 1 && w > 1e-6) { // L1 Soft Constraint
            ...
            if (std::abs(a) < 1e-9) {           // L1 quadratic degeneracy
                lam_val = w / 2.0;
            } else {
                ...
            }
            lam_val = std::max(1e-8, std::min(w - 1e-8, lam_val));   // L1 clamp
            ...
        } else if (type == 2 && w > 1e-6) { // L2 Soft Constraint
            ...
            kp.lam(i) = std::max(1e-8, lam_val);                      // L2 clamp
            kp.s(i) = mu / kp.lam(i);
        } else { // Hard Constraint
            double s_val = std::max(1e-6, -g);                         // hard slack floor
            kp.s(i) = s_val;
            kp.lam(i) = mu / s_val;
        }
```

Five distinct magic numbers — `1e-6` (active-L1/L2 weight threshold), `1e-9` (quadratic degeneracy), `1e-8` (L1/L2 lambda clamp), `1e-6` (hard slack floor), `1e-8` (L1 box clamp). None reference `min_barrier_slack` (which is `1e-12`) or other config field. This implicitly assumes "all dimensionless / O(1) scaled" inputs; a constraint with `g_true ~ 1e-3` would have `-g ~ 1e-3 < 1e-6`, so hard slack initializes to `1e-6` regardless of true magnitude.

Recommendation: parameterize via a single `init_*_floor` config or document as "dimensionless O(1) assumption". Pairs naturally with the missing scaling pass (Dim 3, N-MOD-2).

#### N-NUM-4 (P2) — Restoration `improvement_tol = 1e-12 * max(1, before)` may be inconsistent for large-scale problems

[`solver.h`](../../include/minisolver/solver/solver.h) line 1443:

```1441:1447:include/minisolver/solver/solver.h
        const double violation_after = compute_max_violation(traj);
        const double feasible_bound = config.tol_con * config.feasible_tol_scale;
        const double improvement_tol = 1e-12 * std::max(1.0, violation_before);

        return success && MatOps::is_finite_scalar(violation_after)
            && (violation_after <= feasible_bound
                || violation_after < violation_before - improvement_tol);
```

Hardcoded `1e-12` factor. For `violation_before = 100`, `improvement_tol = 1e-10` — restoration "succeeds" if violation drops by more than 1 part in 1e12 of the original. For `violation_before = 1e-3`, `improvement_tol = 1e-12` — far smaller than `tol_con = 1e-4`. The asymmetry means restoration acceptance is much stricter for already-feasible problems than for badly infeasible ones, opposite of intuition.

### Dimension 2: Convergence / Status Layering

#### N-CONV-1 (P0) — `SolverStatus` lacks standard NLP solver status layering

[`types.h`](../../include/minisolver/core/types.h) lines 7-18:

```7:18:include/minisolver/core/types.h
enum class SolverStatus {
    UNSOLVED,
    OPTIMAL,
    FEASIBLE,
    INFEASIBLE,
    NUMERICAL_ERROR
};
```

Missing standard NLP-solver statuses:
- `MAX_ITER` — iteration budget exhausted but iterate is finite and improving
- `UNBOUNDED` — primal/dual unboundedness detected
- `LINEAR_SOLVE_FAILED` — Riccati factorization failed despite full reg escalation
- `RESTORATION_FAILED` — restoration phase exhausted without progress
- `INVALID_INPUT` — initial state has NaN/Inf, model returned invalid Jacobian on first eval, etc.

Direct evidence the gap is user-visible: `tests/test_status.cpp:96-98` comment explicitly mentions the expectation:

```96:98:tests/test_status.cpp
    // It might return PRIMAL_INFEASIBLE (mapped to INFEASIBLE now)
    // Or MAX_ITER if it cycles (but we want it to detect infeasibility if restoration fails)
```

Today the loop hits `max_iters`, `loop_status = UNSOLVED`, `postsolve` re-checks residuals, and if `primal_inf > tol_con * feasible_tol_scale`, returns `INFEASIBLE`. The user receives `INFEASIBLE` whether the problem is unsolvable or whether they simply needed `max_iters = 200`.

#### N-CONV-2 (P1) — `tol_grad` is a dead config field

`tol_grad = 1e-4` is declared in [`solver_options.h`](../../include/minisolver/core/solver_options.h) line 151 and set by 6 test/tool files. But [`termination.h`](../../include/minisolver/algorithms/termination.h) does not reference it — convergence uses only `tol_con` (primal feasibility), `tol_dual` (dual feasibility = stationarity), `tol_mu`, and `mu_final`. The user can call `config.tol_grad = 1e-12` and it has no effect on solver behavior.

Resolution options: (a) remove the field and update snapshot; (b) connect to a separate `max_grad_norm` check in `TerminationKernel` if there is a stationarity-only check distinct from `tol_dual`.

#### N-CONV-3 (P1) — `OPTIMAL` requires `mu <= mu_final`, breaking standard IPM semantics

[`termination.h`](../../include/minisolver/algorithms/termination.h) lines 14-21:

```14:21:include/minisolver/algorithms/termination.h
        const bool mu_converged = (mu <= config.mu_final);
        const bool primal_ok = (max_primal_inf <= config.tol_con);
        const bool dual_ok = (max_dual <= config.tol_dual);
        const bool kkt_ok
            = (max_barrier_complementarity_residual <= std::max(config.tol_mu, 10.0 * mu));

        return mu_converged && primal_ok && dual_ok && kkt_ok;
```

Standard IPM convergence is "KKT residual ≤ tol AND complementarity ≤ μ_target", and μ is then driven to zero. Here, the "outer" KKT check is gated on `mu <= mu_final`. A user who wants "low-precision real-time" with `mu_final = 1e-4` and `tol_con = 1e-3` will get `OPTIMAL` only if mu has reached 1e-4, even when primal/dual/complementarity already meet their tolerances at mu = 1e-3.

This effectively forces users to set `mu_final` extremely low (default 1e-6) to get OPTIMAL even on easy problems. The `FEASIBLE` fallback (postsolve level 2) absorbs these cases but the user cannot tell from the status whether the result is "stopped early because tolerances met but mu still high" vs "stopped at feasible-but-suboptimal".

#### N-CONV-4 (P2) — `status_to_string(OPTIMAL)` returns `"SOLVED"`, not `"OPTIMAL"`

[`types.h`](../../include/minisolver/core/types.h) line 26:

```20:36:include/minisolver/core/types.h
inline const char* status_to_string(SolverStatus status)
{
    switch (status) {
    case SolverStatus::UNSOLVED:
        return "UNSOLVED";
    case SolverStatus::OPTIMAL:
        return "SOLVED";
    ...
```

Code paths with `status_to_string` in logs or save files report `"SOLVED"` while user-side `if (status == SolverStatus::OPTIMAL)` matches the enum. Inconsistent semantics; snapshot round-trip fine (uses enum) but log and saved text don't match.

#### N-CONV-5 (note) — External benchmark root-cause tracking is out of scope here

ADR 0002 documents deferred globalization work and points to external benchmark gaps. Without running nmpc-bench (out of scope for this review per plan), this repository should not record case-specific root-cause hypotheses. However, **N-THEORY-5 (Mehrotra restoration skip)** below is a generic solver bug that remains valid independently of any benchmark case.

### Dimension 3: Modeling Soundness

#### N-MOD-1 (P1) — MiniModel DSL has no unit/dimension checking

`grep -i "unit|dimension|m\b|rad\b|kg\b|second|meter"` against `python/minisolver/MiniModel.py` (1484 lines) yields no real hits. The DSL accepts any `add_residual(theta - 1.0, weight=...)` regardless of whether `theta` is in radians or degrees. NMPC formulations frequently mix `m`, `rad`, `m/s`, `N`, normalized actuator commands; there is no DSL-level guard.

This is partly a design philosophy ("solver core stays modeling-agnostic"), but at minimum the user-facing documentation should state: "the solver assumes all variables and constraints are O(1) scaled in the user's chosen units; mixing scales by 10^3 or more without explicit scaling will degrade convergence". Currently `README.md` says nothing about this.

This pairs naturally with N-MOD-2 (no scaling) — together they explain why the solver behavior is non-uniform across problem formulations.

#### N-MOD-2 (P0) — Problem-level scaling is completely absent

Confirms `solver-capability-adoption-plan.md` P0 #1 ("Scaling And Normalization"). `grep` on [`solver.h`](../../include/minisolver/solver/solver.h) for "scaling | scale_constraint | scale_state | scale_control | equilibrat | ruiz" returns 0 matches. The strings `scale` that appear in the codebase are local matrix kernel scales (norm scaling), `feasible_tol_scale` (a tolerance multiplier), `reg_scale_up/down` (regularization step sizes), `restoration_alpha` (step size). None of these address per-state, per-control, per-constraint-row natural-scale normalization.

Operational consequence: a constraint row `1e-3 * (g_1 + s_1) ≤ 0` and a constraint row `10 * (g_2 + s_2) ≤ 0` contribute to filter `θ` with the same weight (sum of absolute values). KKT residual checks `|max_dual_inf|` are not scaled. The solver cannot tell "all rows nearly satisfied" from "one row 4 orders of magnitude off and the rest at zero".

Already prioritized as P0 in the capability adoption plan; this review confirms zero progress in code.

#### N-MOD-3 (P2) — Compile-time DoF is enforced, runtime N is silently clamped

`Model::NX/NU/NC/NP` are `static const int` so the templated `MiniSolver<Model, MAX_N>` enforces dimensions at compile time. However:

```118:120:include/minisolver/solver/solver.h
        if (initial_N < 0 || initial_N > MAX_N) {
            std::cerr << "Error: N (" << initial_N << ") outside [0, " << MAX_N << "]. Clamping.\n";
        }
```

is followed by `trajectory(std::max(0, std::min(initial_N, MAX_N)))` — silent clamp. A user who passes `initial_N = -1` (bug) gets `N = 0` (degenerate solve) instead of an exception.

### Dimension 4: Real-Time / RT Safety

#### N-RT-1 (P2) — `set_initial_state(string&)` and `set_parameter(int, string&)` use `std::cerr`, violating logger consistency

`solver.h` lines 257-265, 279-287, 299-307, 549-554, 198-204, 1903-1913 have `std::cerr <<` for warnings instead of `MLOG_WARN(...)`. On embedded toolchains without stderr the program may crash or fail to link. Direct stderr also bypasses any future user-redirectable log sink.

#### N-RT-2 (P2) — `print_iteration_log` uses `std::stringstream` + `std::endl`, breaking zero-malloc when enabled

[`solver.h`](../../include/minisolver/solver/solver.h) lines 1110-1173 use `std::stringstream ss;` (heap-allocates internal buffer for non-trivial output) and `<< std::endl` (forces flush). Default `print_level = NONE` short-circuits at line 1118 so the default solve is unaffected, but if the user enables `PrintLevel::ITER` for debugging, the solve becomes non-zero-malloc. This is consistent with the "profiling/logging is not part of the zero-malloc guarantee" note in `testing-matrix.md`, but the boundary should be explicit in user-facing documentation (today the README says zero-malloc without qualifying which print levels invalidate it).

### Dimension 5: Test Coverage (Numerical-Software View)

#### N-TEST-1 (P1) — Generated 2nd-order Hessians (Q, R, H) are not FD-verified

[`tests/test_solver_quality.cpp`](../../tests/test_solver_quality.cpp) lines 309-431 verify generated `A, B, C, D, q, r` against centered FD with tolerance 1e-4, but stop short of `Q, R, H`. These second-order objects are the most error-prone in symbolic codegen — sign flips on the cross-term `H`, factor-of-two errors in the symmetric Hessian, missing diagonal contributions from squared-residual terms. A simple extension covering Q via `kp_p.cost - 2*kp.cost + kp_m.cost / eps^2` would materially reduce risk.

The existing `tests/test_implicit_sparse_riccati.cpp` (3 tests, fused vs generic match to 1e-7) implicitly catches many Hessian errors via solve outcomes, but cannot diagnose where the error is.

#### N-TEST-2 (P1) — `test_autodiff.cpp` filename misleads

Historical note: this file was later renamed to
[`tests/test_car_model_basic.cpp`](../../tests/test_car_model_basic.cpp).

The old `tests/test_autodiff.cpp` was 88 lines, 3 tests:
- `CarModelDynamics`: asserts continuous dynamics returns `v*cos(0) = v` etc.
- `CarModelIntegratorEuler`: asserts one Euler step value
- `CostDerivatives`: asserts a single cost gradient component

None of these test "auto-differentiation". The actual FD-vs-analytical cross-checks live in `test_solver_quality.cpp` and `test_integrator.cpp`. Either rename to `test_car_model_basic.cpp` or expand to be the real autodiff regression file (covering Q/R/H, soft constraints, multiple models).

#### N-TEST-3 (P1) — Zero property-based testing, zero fuzzing

`grep -i "rapidcheck|RAPID_CHECK|libfuzzer|AFL|hypothesis|fuzz|property[\s_]*based"` returns 0 across the entire repo. All 158 tests use fixed `EXPECT_*` / `ASSERT_*` patterns.

NMPC-relevant property tests that would have high yield:
- "Any feasible convex QP solved by MiniSolver must satisfy KKT residual ≤ 10×tol_kkt"
- "After accepted line search step, true cost(after) + theta(after) ≤ true cost(before) + theta(before) + ε" (filter monotonicity)
- "Mehrotra `mu_aff` is always in [0, current_mu]"
- "After Riccati, `du[N].norm() == 0`" (terminal control was set to zero in [`riccati.h`](../../include/minisolver/solver/riccati.h):704)
- "Integrator step ‖x_next‖ stays bounded for any input ‖x‖ ≤ B, ‖u‖ ≤ U on a contractive linear system" (stability property)

Fuzzing on `set_initial_state` / `set_parameter` could surface the silent-failure API issues (Dim 7).

#### N-TEST-4 (P1) — No fixed golden-reference comparison in this repo

Resolution note: this is now covered in-tree by
[`tests/reference/asset_regression_reference_data.h`](../../tests/reference/asset_regression_reference_data.h)
and [`tests/test_asset_regressions.cpp`](../../tests/test_asset_regressions.cpp).
The header stores fixed golden values generated offline, and the C++ test
compares MiniSolver on kinematic bicycle and 3D double-integrator cases. Broader
solver-to-solver comparison remains in MiniSolver-Bench.

#### N-TEST-5 (P2) — ASan/UBSan in CI Debug builds but not in `build.sh`

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) line 83 enables `-fsanitize=address,undefined -fno-omit-frame-pointer` for Debug builds. [`build.sh`](../../build.sh) does plain `cmake ..` then `ctest` — local developers never see ASan unless they manually configure `.build_asan`. Add an `ASAN=1 ./build.sh` mode or document the manual command.

#### N-TEST-6 (P2) — `test_memory.cpp` does not exercise generated bicycle (NX=6, NU=2, NC=10) under zero-malloc

[`tests/test_memory.cpp`](../../tests/test_memory.cpp) tests use `CarModel` and inline toy models (`SocTriggerModel`, `MemTestImplicitModel`). The NMPC-realistic `BicycleExtModel` (with hard + soft + obstacle constraints) is not in the zero-malloc matrix. A constraint-heavy real model is the most likely place to hit a forgotten allocation.

### Dimension 6: Observability / Diagnostics

#### N-OBS-1 (P1) — `SolverContext` exposes a small fraction of capability-adoption-plan diagnostics

Current queryable fields ([`solver_context.h`](../../include/minisolver/core/solver_context.h)):
- `solve.{mu, reg, current_iter, slack_reset_consecutive_count}`
- `metrics.{last_prim_inf, last_dual_inf, last_alpha, last_mu_aff, last_alpha_aff}`

Capability adoption plan P0 list (recommended diagnostic fields):
- requested vs actual backend (after fallback)
- degraded fallback flag (Riccati small-NU freeze, line-search merit fallback)
- linear solver factorization status + retry count
- regularization escalation count
- barrier history summary
- line-search type, accepted alpha, backtracking count
- SOC attempted / accepted / rejected counts
- restoration attempted / accepted / rejected counts
- warm-start mode + repaired flag
- scaled vs unscaled residual summaries

10 of 10 fields missing. ADR'd as deferred-design (`solver-capability-adoption-plan.md` P0 #3). This review confirms zero progress.

#### N-OBS-2 (P1) — `logger.h` is not embedded-safe and not redirectable

[`logger.h`](../../include/minisolver/core/logger.h) (61 lines) writes directly to `std::cout` / `std::cerr` with hardcoded ANSI escape codes (`\033[31m` etc.):

```19:33:include/minisolver/core/logger.h
#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_ERROR
#define MLOG_ERROR(x)                                                                              \
    do {                                                                                           \
        std::cerr << "\033[31m[ERROR]\033[0m " << x << std::endl;                                  \
    } while (0)
...
#define MLOG_WARN(x)                                                                               \
    do {                                                                                           \
        std::cout << "\033[33m[WARN] \033[0m " << x << std::endl;                                  \
    } while (0)
```

Issues for embedded:
- `std::cout` / `std::cerr` may not exist (size-optimized STM32CubeIDE / IAR profiles)
- `std::endl` forces flush, RT-unsafe
- ANSI escapes pollute non-terminal logs
- No way to redirect to user-provided `log_callback(level, msg)` sink

For a production `embedded NMPC library`, this is the single most blocking embedded-portability issue together with N-EMBED-1. The simple fix (provide a weak symbol `void minisolver_log(int level, const char* msg)` that defaults to the current cout/cerr behavior, allowing override) is small but high-value.

#### N-OBS-3 (P2) — Iteration log does not include Mehrotra `α_aff / μ_aff`

[`solver.h`](../../include/minisolver/solver/solver.h) `print_iteration_log` outputs 7 columns (Iter, Cost, Log(Mu), Log(Reg), PrimInf, DualInf, Alpha) plus optional MinSlack at DEBUG. Mehrotra-mode users debugging convergence have to re-derive `α_aff / μ_aff` from `metrics.last_*` fields outside the log. These are already computed in `compute_mehrotra_direction_` and stored. Add 2 columns when `barrier_strategy == MEHROTRA`.

### Dimension 7: API Ergonomics / Error Quality

#### N-API-1 (P1) — Seven user-facing setter APIs are silent-failure or cerr-then-return

Inventory from `solver.h`:

| API | Failure mode | Behavior |
| --- | --- | --- |
| `set_initial_state(vector<double>& x0)` line 246 | `x0.size() != NX` | silent return, state not set |
| `set_initial_state(string& name, double)` line 257 | unknown name | `std::cerr` + return |
| `set_parameter(int stage, int idx, double)` line 268 | stage or idx out of range | silent return |
| `set_parameter(int stage, string& name, double)` line 279 | unknown name | `std::cerr` + return |
| `set_global_parameter(int idx, double)` line 289 | idx out of range | silent return |
| `set_global_parameter(string& name, double)` line 299 | unknown name | `std::cerr` + return |
| `set_dt(vector<double>& dts)` line 549 | `dts.size() > MAX_N` | `std::cerr` + truncate |
| `resize_horizon(int new_n)` line 199 | new_n out of range | `std::cerr` + return |
| `MiniSolver(int initial_N, ...)` line 112 | initial_N out of range | `std::cerr` + clamp |

When a user typos a state name, the constraint they think they're setting is just not set and the solve runs on uninitialized data. The only observable signal is a `std::cerr` line that may not appear if the application redirects/disables stderr.

For a library targeted at embedded robotics where errors must be detected programmatically, returning `bool` or `std::optional<int>` would let the caller assert success.

#### N-API-2 (P2) — `SolverConfig` has 50+ fields without preset profiles

[`solver_options.h`](../../include/minisolver/core/solver_options.h) lines 113-228. The capability-adoption-plan P1 "Solver Profiles" item proposes `Reference / Default / Speed / Robust` presets. Today users must manually configure `barrier_strategy`, `inertia_strategy`, `line_search_type`, `termination_profile`, `enable_soc`, `enable_corrector`, `enable_aggressive_barrier`, `enable_slack_reset`, `enable_feasibility_restoration`, `enable_defect_correction`, `enable_line_search_rollout`, `direction_refinement`, `hessian_approximation`, etc. plus tolerances. The example in [`examples/01_car_tutorial/main.cpp`](../../examples/01_car_tutorial/main.cpp) sets only 5 fields, which is the typical usage but does not cover important configurations.

#### N-API-3 (P2) — No explicit `set_warm_start_*` API; users mutate `trajectory[k].*` directly

The warm-start workflow uses `InitializationMode` + `WarmStartBarrierMode` + `WarmStartRegularizationMode` enums, but to actually populate `s/lam/soft_s` from a previous solve the user must directly access `solver.trajectory[k].s(i) = ...`. There is no `set_warm_start_dual(stage, vector)` style abstraction. This couples users to the internal `KnotPoint` layout.

### Dimension 8: Degradation Modes

#### N-DEG-1 (P1, follows up on 5/2 P2 confirmed/intentional) — Riccati small-NU freeze produces silent zero-control direction

[`riccati.h`](../../include/minisolver/solver/riccati.h) lines 471-577: when `Knot::NU <= 3` and `fast_inverse(R_bar, Quu_inv)` fails, the code falls back to "freeze a minimal set of control dimensions and solve only in a SPD principal subspace". For NU=2, this means `du(frozen_dim) = 0`. For NU=3 it freezes 1 or 2 dims.

5/2 review classified this as `intentional` (the alternative is failing the entire solve), but the user has no way to know it happened. From outside, the solver returns `OPTIMAL` with `du(frozen_dim) = 0` as if that were the optimal control — but it is not, it's a forced zero. This compounds with N-CONV-1 (no `LINEAR_SOLVE_FAILED` status) and N-OBS-1 (no degraded-fallback diagnostics).

Minimum fix: add an integer counter `degraded_riccati_freeze_count` to `SolverContext`. Users can then check it post-solve and treat the result as suspect.

### Dimension 9: Embedded Portability Reality

#### N-EMBED-1 (P0) — README markets "STM32 / no external libraries / Embedded Safety", but multiple build-time blockers exist

Concrete blockers:

| Issue | Location | Impact |
| --- | --- | --- |
| `#include <iostream>` in 11 headers | `solver.h, trajectory.h, solver_snapshot.h, line_search.h, riccati.h, matrix_defs.h, implicit_integrator.h, line_search_utils.h, mini_matrix.h, logger.h, backend_interface.h` | Many embedded toolchains (STM32CubeIDE size-optimized, IAR EWARM nano profile) do not provide iostream |
| `throw std::invalid_argument` (4 sites) | [`implicit_integrator.h`](../../include/minisolver/integrator/implicit_integrator.h) lines 76, 150, 418, 434 | `-fno-exceptions` build fails to compile |
| `std::unordered_map<std::string, int>` for name maps | [`solver.h`](../../include/minisolver/solver/solver.h) lines 1954-1956 | Heap allocation in constructor; `<unordered_map>` not always available |
| No ARM cross-compile job | `.github/workflows/ci.yml` matrix is `ubuntu-latest` only | Embedded claim never CI-validated |
| No `-fno-rtti -fno-exceptions` build option | [`CMakeLists.txt`](../../CMakeLists.txt) | Embedded ABI invariants not enforced |
| No binary size measurement | [`CMakeLists.txt`](../../CMakeLists.txt) | "Embedded-friendly" claim has no size budget |

Concrete fix path (in order):
1. Add an opt-in CMake option `MINISOLVER_EMBEDDED_PROFILE` that defines `MINISOLVER_NO_IOSTREAM` and `MINISOLVER_NO_EXCEPTIONS`, switches the logger to a callback hook (resolves N-OBS-2), replaces `throw` with `assert` + status return in `dispatch_compute_dynamics` / `dispatch_integrate`, switches name maps to `static constexpr` linear search.
2. Add an ARM Cortex-M GCC cross-compile job that builds at least `examples/01_car_tutorial` with the embedded profile.
3. Document binary size for that example.

This is a substantial effort but the marketing claim is currently unsupported.

#### N-EMBED-2 (P1) — `std::cerr` direct calls in 7 sites bypass logger

Already documented in N-RT-1 / N-API-1 from a different angle. From the embedded portability angle: even if `MLOG_*` is muted, the direct `std::cerr` calls remain compiled in.

### Dimension 10: AD Correctness

#### N-AD-1 (corrected) — Generated 1st-order derivatives ARE FD-verified

Initially I suspected the old `test_autodiff.cpp` (later renamed to
[`tests/test_car_model_basic.cpp`](../../tests/test_car_model_basic.cpp)) was the only AD test. On follow-up, [`tests/test_solver_quality.cpp`](../../tests/test_solver_quality.cpp):309-431 has comprehensive 1st-order coverage on CarModel; [`tests/test_integrator.cpp`](../../tests/test_integrator.cpp):767 (`JacobiansMatchFiniteDifferenceForAllImplicitSchemes`) covers implicit integrator A/B. **This is not a finding — correction to plan-stage assumption.**

#### N-AD-2 (P2) — Generated `jacobian_continuous` writes all zero entries explicitly

[`examples/02_advanced_bicycle/generated/bicycleextmodel.h`](../../examples/02_advanced_bicycle/generated/bicycleextmodel.h) lines 96-120: every `jac.Jx(i, j)` is explicitly assigned, including the 24 zeros of a 6×6 sparse matrix. Optimizers may or may not eliminate the zero stores depending on flags (`-O3` typically does for stack-local matrices, but for `kp.A`-style member-access it depends on aliasing analysis). This is a minor performance issue if `compute_dynamics` becomes a hot path; the dominant solution (used elsewhere) is fused Riccati that bypasses the dense Jacobian materialization entirely. Worth noting but not blocking.

### Dimension 11: Sparse-Structure Exploitation

No new findings. The 5/1 review's `Implicit Midpoint sparsity may be overly conservative` finding (matmul of `m_inv * m_pattern` produces dense pattern) remains valid — this review confirms by inspection of [`python/minisolver/MiniModel.py`](../../python/minisolver/MiniModel.py) lines 1047, 1065, 1225 that the codegen path is unchanged.

The fused vs generic Riccati equivalence test [`tests/test_implicit_sparse_riccati.cpp`](../../tests/test_implicit_sparse_riccati.cpp) provides integration-level guarantee that fused matches generic to 1e-7, but does not measure sparsity-pattern minimality vs the theoretical lower bound.

### Dimension 12: Numerical Precision

#### N-PREC-1 (P2) — Solver hard-codes `double`; no `float` or mixed-precision support

`MSVec<double, ...>` is fixed across the codebase (matrix_defs.h). Embedded targets like Cortex-M4F have a single-precision FPU only — running double on these requires software emulation, often 5-20× slower. A symbolic-codegen NMPC solver in the embedded space typically supports a templated `Scalar` to allow `float` builds. This would also enable mixed-precision (e.g. float forward, double KKT residual) approaches that have shown 2-3× wins in HPIPM.

Not a current bug, but a future-blocking design choice for the embedded marketing.

#### N-PREC-2 (P2) — Tolerance fields' physical meanings are not documented

`tol_con / tol_dual / tol_mu / tol_grad / tol_cost` defaults all 1e-4 to 1e-6 with no comment explaining what unit they're in. A user new to NMPC won't know that `tol_con = 1e-4` means "max constraint violation in the user's units ≤ 1e-4", and is therefore problem-scale-dependent. Tied to N-MOD-2 (no scaling).

### Dimension 13: Dependencies & License

#### N-DEP-1 (resolved) — CasADi is no longer a MiniSolver dependency

[`requirements.txt`](../../requirements.txt):

```text
sympy>=1.10
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
```

The earlier CasADi test/reference dependency was removed. Asset regression tests
now use checked-in fixed golden constants in
[`tests/reference/asset_regression_reference_data.h`](../../tests/reference/asset_regression_reference_data.h),
so configuring or running MiniSolver tests does not require CasADi/IPOPT.

#### N-DEP-2 (P2) — Python `requirements.txt` pins only minor floors, not exact versions

`sympy>=1.10` accepts any 1.x and 2.x. CI may break or change behavior on dependency updates. For deterministic reproducibility, either pin exact versions in CI (`pip install sympy==1.12 ...`) or generate `requirements-lock.txt`.

### Dimension 14: Theory ↔ Implementation Alignment

#### N-THEORY-1 (P1) — Filter line search is missing three Wächter-Biegler 2006 §2.3 mechanisms, ADR 0002 documents only one

Already-deferred per [ADR 0002](../adr/0002-filter-line-search-switching.md): switching condition Eqn 19-20.

Additionally missing:
- (a) **`θ_max` filter sentinel seed (Eqn 21)**: the filter should be initialized with `F_0 = {(θ, φ) : θ ≥ θ_max}` where `θ_max = 1e4 · max(1, θ(x_0))`. [`line_search.h`](../../include/minisolver/algorithms/line_search.h) `reset()` (line 577-581) only clears entries — no sentinel.
- (b) **f-type acceptance must NOT augment the filter** (Wächter-Biegler §2.3 algorithm step A-5.4). Today every accepted step pushes to filter (`line_search.h` line 681-687).

ADR 0002 should be updated to acknowledge all three gaps, or a sub-ADR added for (a)/(b). Cross-solver case-level impact should be measured and recorded in MiniSolver-Bench.

#### N-THEORY-2 (P1) — Filter ring buffer overwrites oldest entries after 1024 steps

[`line_search.h`](../../include/minisolver/algorithms/line_search.h) lines 681-687:

```679:687:include/minisolver/algorithms/line_search.h
        if (accepted) {
            trajectory.swap();
            if (filter_size_ < FILTER_CAPACITY) {
                filter[filter_size_] = { theta_0, phi_0 };
                ++filter_size_;
            } else {
                filter[filter_next_] = { theta_0, phi_0 };
                filter_next_ = (filter_next_ + 1) % FILTER_CAPACITY;
            }
        }
```

`FILTER_CAPACITY = 1024` (line 377). After 1024 accepted steps the oldest entry is overwritten. This breaks the formal "monotone over all history" property of Wächter-Biegler's filter — a candidate is checked against a recent window, not the full history. For NMPC with `max_iters = 100` this is moot, but for difficult problems requiring many iterations or for benchmarks with thousands of repeated solves under a single `MiniSolver` instance, this could mask non-monotone behavior.

Standard implementations use a Pareto-frontier list (entries dominated by another are pruned), bounding size by problem geometry rather than raw count.

Test coverage: `LineSearchTest.FilterHistoryWrapsAtFixedCapacity` documents the wraparound behavior but does not assert it preserves the certificate.

#### N-THEORY-3 (P1) — Mehrotra `α_aff` is a single primal+dual fraction, not split

Standard Mehrotra:
- α_aff_p = max α s.t. s + α·ds ≥ τ·s    (primal)
- α_aff_d = max α s.t. λ + α·dλ ≥ τ·λ    (dual)
- μ_aff = (s + α_aff_p·ds)·(λ + α_aff_d·dλ) / m

MiniSolver ([`solver.h`](../../include/minisolver/solver/solver.h) line 1005, `compute_fraction_to_boundary_(affine_traj)`) computes a single α covering all primal-dual variables. This produces a more conservative `μ_aff` (it takes `min(α_p, α_d)` implicitly), which makes `σ = (μ_aff/μ)^3` larger and therefore `μ_target` larger, i.e. less aggressive barrier reduction. This is a safe but suboptimal Mehrotra implementation.

Not a correctness bug; a performance-on-easy-problems issue. Case-level relevance should be measured in MiniSolver-Bench.

#### N-THEORY-4 (P0) — No inertia detection in Riccati / SPD recovery

[`riccati.h`](../../include/minisolver/solver/riccati.h) lines 461-665 (general SPD path): on factorization failure, the strategies are:
- `REGULARIZATION`: add `regularization_step` (1e-6) and retry once
- `IGNORE_SINGULAR`: scan diagonal, add `huge_penalty` (1e9) to entries below `singular_threshold`
- `SATURATION`: clamp diagonal entries to `max(reg, reg_min)`, retry

None of these check the **inertia** of the resulting (regularized) KKT system. Standard primal-dual IPM (Wächter-Biegler §3.1, IPOPT, HPIPM, `MA57`/`Pardiso` indefinite solvers) demand:
- `n_positive` = number of state+control unknowns
- `n_negative` = number of equality constraints + slack-on-bound terms
- `n_zero` = 0

If actual inertia differs (`n_negative` too small ⇒ saddle), the regularization should be **increased** (typically by powers of 10) until inertia matches. Today the `RiccatiSolver` regularization is purely SPD-failure driven — it can succeed in factorizing a regularized matrix that converges to a saddle, not a minimizer.

Implementation cost: requires LDLT (with diagonal pivoting) to read off positive/negative eigenvalue counts, replacing or augmenting the current LLT-based path. Significant work. Any case-level explanation belongs to MiniSolver-Bench evidence, not this repository.

#### N-THEORY-5 (P0) — Restoration silently skips quadratic feasibility penalty under Mehrotra

[`solver.h`](../../include/minisolver/solver/solver.h) lines 1317-1339:

```1311:1339:include/minisolver/solver/solver.h
            // Restoration linear solve.
            // Quadratic-penalty feasibility restoration:
            // Minimizing 0.5*||dx||^2 + 0.5*rho*||C*dx + D*du + g + s||^2
            ...
            if (config.barrier_strategy != BarrierStrategy::MEHROTRA) {
                double rho = 1000.0; // Fixed quadratic restoration penalty.
                for (int k = 0; k <= N; ++k) {
                    auto& kp = traj[k];
                    kp.Q.noalias() += rho * kp.C.transpose() * kp.C;
                    kp.R.noalias() += rho * kp.D.transpose() * kp.D;
                    kp.H.noalias() += rho * kp.D.transpose() * kp.C;
                    kp.q.noalias() += rho * kp.C.transpose() * kp.g_val;
                    kp.r.noalias() += rho * kp.D.transpose() * kp.g_val;
                }
            }
```

The `if (...!= MEHROTRA)` guard is unexplained — there is no comment justifying why Mehrotra mode skips the restoration penalty. The code immediately above (lines 1294-1309) zeros out cost (`Q.setIdentity(); q.setZero();` etc.), so under Mehrotra the restoration linear solve receives an all-zero / identity-cost system with no feasibility pressure. The result is `dx = 0, du = 0`, restoration "applies a step" but does not move toward feasibility.

This is a generic MiniSolver restoration bug. Restoration is supposed to be the safety net when filter line search collapses; under Mehrotra (the recommended strategy for nonlinear NMPC per `DEBUG_README.md` "Feasible Stagnation" section), the safety net is silently disabled.

The fix is one of:
- Remove the `if (... != MEHROTRA)` guard (apply quadratic penalty regardless).
- Add an explicit ADR explaining why Mehrotra restoration must use a different algorithm, then implement that algorithm.

Either way, the silent skip without comment is a confirmed bug-class finding.

#### N-THEORY-6 (P2) — Merit `dphi_` uses finite-difference instead of analytic directional derivative

[`line_search.h`](../../include/minisolver/algorithms/line_search.h) lines 317-326:

```317:326:include/minisolver/algorithms/line_search.h
        dphi_ = 0.0;
        if (config.armijo_c1 > 0.0 && alpha > 0.0) {
            const double eps_alpha = std::min(1.0e-6, std::max(1.0e-10, alpha * 1.0e-6));
            build_trial(candidate, active, dt_traj, N, eps_alpha, config);
            const double phi_eps = compute_merit(candidate, N, mu, config);
            dphi_ = (phi_eps - phi_0) / eps_alpha;
            if (!std::isfinite(dphi_)) {
                dphi_ = 0.0;
            }
        }
```

Each merit search calls `build_trial` (full-trajectory forward construction + N model evaluations) just to estimate `dphi_` via forward-difference. The merit function is `Σ cost - μ Σ log(s) + soft constraint barriers + merit_nu Σ violations`, all of whose gradients with respect to (dx, du, ds, dλ, dsoft_s) are available in closed form from `q, r, q_bar, r_bar` (after Riccati) and the constraint Jacobians (C, D). An analytic `compute_grad_phi_times_d(...)` would (a) eliminate the per-search trial-point construction overhead, (b) be exact instead of O(eps_alpha) accurate, (c) detect non-descent directions definitively rather than via noisy comparison.

This is the same mechanism ADR 0002 lists as needing-implementation for the f-type switching condition. Bundling the two would amortize the design cost.

---

## Verified Non-Issues / Corrections

| Initial concern | Verification | Resolution |
| --- | --- | --- |
| Old `test_autodiff.cpp` is the only AD test, FD verification absent | Found `test_solver_quality.cpp:309-431` (1st-order FD), `test_integrator.cpp:767` (implicit A/B FD); the old file was later renamed to `test_car_model_basic.cpp` | 1st-order FD coverage exists; only 2nd-order Hessians remain uncovered (recorded as N-TEST-1) |
| ASan/UBSan completely missing from CI | Found `.github/workflows/ci.yml:83` enables `-fsanitize=address,undefined` for Debug build matrix | Sanitizers in CI; only `build.sh` lacks the option (recorded as N-TEST-5) |
| GPU backend silently lies about being implemented | Found `src/cuda/gpu_ops.cu:2` `#error "GPU Backend implementation is incomplete..."` plus 5/2 `FeaturesTest.GPUBackendUnsupportedFailsExplicitly` confirmation | Honest-incomplete; not a finding |
| No golden reference possible without external project | Fixed golden constants are checked in under `tests/reference/asset_regression_reference_data.h` | Cross-check infrastructure exists without making CasADi a MiniSolver dependency |
| Filter line search has only the ADR'd switching condition gap | Re-read Wächter-Biegler §2.3: also missing `θ_max` sentinel and f-type filter-augmentation skip | ADR 0002 should be expanded (recorded as N-THEORY-1) |

---

## Cross-Reference With Prior Reviews / Gap Backlog

### Confirmed prior findings (no new evidence, just verification)

| Prior finding | Current status | Reference |
| --- | --- | --- |
| `10.0 * mu` hardcoded in `termination.h` (5/2 L2) | Confirmed (`termination.h:18`) | unchanged |
| Restoration `rho = 1000.0` hardcoded (5/2 L3) | Confirmed (`solver.h:1318`) | unchanged |
| Iter 1 skipped convergence checks (5/2 L4) | Confirmed (`solver.h:1610` `current_iter > 1`) | unchanged |
| Merit/filter metric duplication (5/2 L7) | Confirmed (`line_search.h` build_trial / compute_metrics duplicated 3x) | unchanged |
| `dphi_` member-state but local-use only (5/2 L8) | Confirmed (`line_search.h:132, 322`) | unchanged |
| Newton reg `1e-12` hardcoded (5/2 L9) | Confirmed (`solver_options.h:20`) | unchanged |
| matrix singular `1e-30` hardcoded (5/2 L11) | Confirmed (`matrix_defs.h:86, 97, 358, 435`) | unchanged |
| `SolverContext` partial scaffolding (5/2 M4) | Confirmed (`DirectionState.affine_mu / affine_alpha` exist but solver writes `metrics.last_mu_aff` instead) | unchanged |
| Newton solver `warm_start` dead code (5/1) | Confirmed (`implicit_integrator.h` uses local `NewtonSolver<...> ns;` everywhere) | unchanged |
| Numerical Jacobian per-iter overhead for handwritten models (5/1) | Confirmed (`implicit_integrator.h` `get_continuous_jacobians` SFINAE-falls-back to FD) | unchanged |

### Gap backlog (`.claude/gaps/gap_backlog.md`) intersection

| Backlog item | Status per backlog | This review's view |
| --- | --- | --- |
| `#11 Fused Riccati dispatch — is_fused_riccati_integrator_compatible 未被 riccati.h 调用` | pending | Confirmed: `riccati.h:393` does call it, but this might be the recent fix — needs verify against backlog date |
| `#12 Mehrotra alpha_aff L1 soft fraction-to-boundary 覆盖检查` | pending | Related to N-THEORY-3 above — split α_aff would naturally cover this |
| External benchmark convergence root cause | out of scope here | Track concrete case outcomes and root-cause hypotheses in MiniSolver-Bench; this review only records generic MiniSolver gaps |
| External benchmark precision gap | out of scope here | Track concrete case outcomes and root-cause hypotheses in MiniSolver-Bench; this review only records generic MiniSolver gaps |
| chain_mass priminf overshoot | deferred (ADR 0002) | Not advanced |

### New findings unique to this review (not in any prior document)

`N-NUM-1, N-NUM-2, N-NUM-3, N-NUM-4, N-CONV-1, N-CONV-2, N-CONV-3, N-CONV-4, N-MOD-1, N-MOD-3, N-RT-1, N-RT-2, N-TEST-1, N-TEST-2, N-TEST-3, N-TEST-4, N-TEST-5, N-TEST-6, N-OBS-2, N-OBS-3, N-API-1, N-API-2, N-API-3, N-DEG-1, N-EMBED-1, N-EMBED-2, N-AD-2, N-PREC-1, N-PREC-2, N-DEP-1, N-DEP-2, N-THEORY-1, N-THEORY-2, N-THEORY-3, N-THEORY-4, N-THEORY-5, N-THEORY-6` (37 items).

`N-MOD-2, N-OBS-1` are restatements of capability-adoption-plan items at higher severity (this review confirms zero progress).

`N-CONV-5` is a repository-boundary correction: case-specific benchmark validation belongs to MiniSolver-Bench.

---

## Action Matrix

P0 (block release / correctness-affecting):

| ID | Area | Estimated cost | Estimated impact |
| --- | --- | --- | --- |
| N-CONV-1 | Status layering | Small (add 5 enum values, update postsolve switch) | High user-side debuggability |
| N-EMBED-1 | Embedded reality | Large (CMake profile + logger callback + name-map rewrite + ARM CI) | High; matches marketing |
| N-MOD-2 | Problem scaling | Large (already P0 in capability adoption plan) | High; explains many edge-case failures |
| N-THEORY-4 | Inertia detection | Large (LDLT-based KKT + retry policy) | High; correctness on degenerate problems |
| N-THEORY-5 | Mehrotra restoration | Small (remove `if (... != MEHROTRA)` or add ADR) | High; generic restoration correctness gap |

P1 (significant):

| ID | Area | Estimated cost | Estimated impact |
| --- | --- | --- | --- |
| N-NUM-1 | Mehrotra `update_mu` zero guard | Trivial | Correctness corner case |
| N-NUM-2 | `lam_i` floor in dual recovery | Trivial | Numerical safety |
| N-CONV-2 | `tol_grad` dead config | Trivial (delete or wire) | API integrity |
| N-CONV-3 | `OPTIMAL` mu requirement | Small | API semantics |
| N-OBS-1 | Diagnostics surface | Medium (already ADR'd) | Debuggability |
| N-OBS-2 | Embedded-safe logger | Medium (callback hook) | Embedded portability |
| N-API-1 | Silent setter API | Medium (return bool / optional) | User-facing safety |
| N-DEG-1 | Riccati freeze diagnostics | Trivial (counter) | Debuggability |
| N-EMBED-2 | `std::cerr` direct calls | Trivial (replace with MLOG) | Embedded portability |
| N-TEST-1 | Hessian FD verification | Small (extend existing FD test) | AD codegen safety |
| N-TEST-2 | `test_autodiff.cpp` rename | Trivial | Test discoverability |
| N-TEST-3 | Property-based tests | Medium (RapidCheck + 5-10 properties) | Catch regression bugs |
| N-TEST-4 | Fixed golden-reference cross-check | Fixed by asset regression suite | Algorithm validation |
| N-MOD-1 | DSL unit checks | Medium (sympy-based) | Modeling safety |
| N-THEORY-1 | Filter `θ_max` + f-type | Medium (paired with ADR 0002 work) | Convergence theory |
| N-THEORY-2 | Filter Pareto-frontier | Small | Acceptance certificate completeness |
| N-THEORY-3 | Mehrotra split α_aff_p / α_aff_d | Small | Mehrotra performance |

P2 (cleanup, hardening):

| ID | Area | Estimated cost |
| --- | --- | --- |
| N-NUM-3 | Initialization magic numbers | Small (consolidate to config or document) |
| N-NUM-4 | Restoration improvement_tol | Trivial |
| N-CONV-4 | `status_to_string(OPTIMAL) == "SOLVED"` | Trivial |
| N-RT-1 | Logger consistency | Trivial |
| N-RT-2 | Iteration log allocation boundary | Trivial (documentation) |
| N-OBS-3 | Mehrotra log columns | Trivial |
| N-API-2 | Solver profiles | Medium |
| N-API-3 | warm-start API | Small |
| N-MOD-3 | Constructor N validation | Trivial (throw vs clamp) |
| N-AD-2 | Generated jacobian zero stores | Codegen change, low priority |
| N-PREC-1 | Float / mixed precision | Large; future |
| N-PREC-2 | Tolerance documentation | Trivial |
| N-DEP-1 | License documentation | Trivial |
| N-DEP-2 | Pin Python deps | Trivial |
| N-TEST-5 | ASan in build.sh | Trivial |
| N-TEST-6 | bicycle in zero-malloc matrix | Small |
| N-THEORY-6 | Analytic merit dphi | Small (paired with ADR 0002) |

---

## Recommended Sequencing

If this fix ledger is worked, the suggested order (maximizing per-step risk reduction):

1. **N-THEORY-5** (trivial code change, removes silent algorithm gap). Validate with a focused local regression; cross-solver case impact belongs to MiniSolver-Bench.
2. **N-CONV-1** (status layering — unblocks distinguishing failure modes for all subsequent debugging).
3. **N-CONV-2 + N-NUM-1 + N-NUM-2 + N-EMBED-2 + N-RT-1 + N-CONV-4 + N-OBS-3** (small individual fixes; bundle into one "hardening" commit).
4. **N-TEST-4 (fixed golden-reference cross-check)** — resolved by checked-in asset regression reference data.
5. **N-TEST-1 (Hessian FD)** — directly extends existing FD test pattern.
6. **N-TEST-3 (property tests)** — set up RapidCheck once, add 5-10 properties incrementally.
7. **N-OBS-1 + N-OBS-2 (diagnostics + logger callback)** — paired since both touch the observability surface.
8. **N-API-1 (silent setter cleanup)** — paired with N-OBS-2 hooks (optional return values, structured errors).
9. **N-EMBED-1 (embedded profile + ARM CI)** — large but well-scoped; treats README as a contract.
10. **N-MOD-2 (scaling)** — already prioritized P0 in capability adoption plan; this review just confirms urgency.
11. **N-THEORY-1 + N-THEORY-2 + N-THEORY-3 + N-THEORY-6** (filter theory + Mehrotra split + analytic merit dphi) — bundle as one filter/merit theory pass.
12. **N-THEORY-4 (inertia detection)** — largest and most uncertain; defer until benchmark evidence demands it.

Items not on this path (N-PREC-1 float/mixed precision, N-API-2 solver profiles, N-MOD-1 unit checks) are deferred-design candidates.

---

## Review Process Notes

- This review was conducted in 4 sequential phases (Phase 1 algorithmic / mathematical, Phase 2 AD / sparsity, Phase 3 RT / embedded / engineering, Phase 4 testing / modeling / dependencies) plus a synthesis phase. No subagents were used.
- All evidence is in-tree at HEAD as of 2026-05-02. `ctest` baseline: 25/25 pass on `.build` (Eigen, Release).
- No code modifications were made; all findings are recommendations.
- External benchmarks (acados / CasADi / IPOPT runtime comparisons) are out of scope per the project plan to use a separate `nmpc-bench` repository for that purpose.
