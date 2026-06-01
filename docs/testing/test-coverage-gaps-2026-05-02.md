# MiniSolver Test Coverage Gap Snapshot

**Date:** 2026-05-02
**Baseline:** 25 CTest entries after the `18da9fc` test split, Eigen and Custom backends
**Source:** [review_2026-05-02.md](../reviews/review_2026-05-02.md)
**Status:** historical snapshot. For current contract coverage, use
[`contract-coverage-matrix.md`](contract-coverage-matrix.md) and
[`contract-rollout-completion-audit.md`](contract-rollout-completion-audit.md).

**2026-05-02 follow-up:** `GAP-1` through `GAP-9` and `GAP-12` now have direct
regression coverage, except `GAP-6`, which remains a benchmark or real-case
trigger candidate. Covered items were confirmed as coverage gaps rather than
current behavior failures.

This file preserves the May 2 review follow-up state. Do not treat the test
counts below as the current suite inventory.

---

## 2026-05-02 Coverage Snapshot

| Test File | Test Count | Coverage Area |
|-----------|------------|---------------|
| `test_bugfixes.cpp` | 31 | NaN detection, Mehrotra, SOC, barrier update, dual recovery |
| `test_config_regressions.cpp` | 4 | Config, backend, build-state, horizon, query guards |
| `test_barrier_residual_contract.cpp` | 6 | Barrier residual snapshots, postsolve stale-residual guards, Mehrotra target-mu edge cases |
| `test_integrator.cpp` | 18 | Dispatch rejection, Newton convergence, three implicit schemes, stiffness, A/B Jacobians, terminal implicit evaluation, warm start |
| `test_solver_quality.cpp` | 15 | Finite-difference Jacobians, KKT optimality, analytic solutions, MPC closed loop, Mehrotra, reference config |
| `test_line_search.cpp` | 14 | Filter and merit acceptance, SOC candidate semantics, damping, model hook, rollout, filter capacity |
| `test_mini_matrix.cpp` | 13 | Cholesky, LDLT, LU, block views, dot, symmetrize, finite checks |
| `tests/minimodel/*` | 5 CTest entries | Identifiers, implicit patterns, constraint packet and SOC, residual costs, terminal projection |
| `test_memory.cpp` | 6 | Zero-malloc coverage across 6 configuration combinations |
| `test_solver_snapshot.cpp` | 14 | Round trip, `soft_s`, config fields, backend policy, model fingerprint, invalid format rejection, atomicity, failure capture |
| `test_soft_constraints.cpp` | 6 | L1/L2 convergence, invalid dual warm start, tiny L1 weight initialization |
| `test_features.cpp` | 4 | Basic features |
| `test_advanced.cpp` | 4 | Advanced scenarios |
| `test_solver.cpp` | 3 | Full convergence, infeasible recovery, horizon resize |
| `test_implicit_sparse_riccati.cpp` | 3 | Fused vs generic Riccati for the three implicit integrators |

Well-covered areas:

- Implicit integrator pipeline: dispatch, Newton, Jacobian, and Riccati.
- SOC refactor: candidate semantics, damping, and model hook.
- NaN propagation: Jacobian, dynamics, and soft slack paths.
- Mehrotra logic: mu gating and L1 soft-pair `mu_aff`.
- Zero-malloc solve path across line-search and integrator combinations.
- Snapshot round trip and invalid-format rejection.

---

## Gap Ledger

### High Priority

#### GAP-1: Terminal Dynamics Cost In `evaluate_model_stage`

- **Related issue:** High-priority review finding that terminal knots still called dynamics.
- **Coverage:** `ImplicitIntegratorTest.TerminalImplicitEvaluationAtZeroDtIsFinite`.
- **Original missing test:** Call `evaluate_model_stage` on a terminal knot with an implicit-integrator model and verify finite output.
- **Current conclusion:** Terminal implicit evaluation at `dt=0` is finite and produces identity/zero discrete Jacobians. Remaining risk is performance, not correctness.

#### GAP-2: Stale Dual Infeasibility In Convergence Checks

- **Related issue:** Medium-priority review finding that in-loop convergence used pre-line-search `max_dual_inf`.
- **Coverage:** `BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`.
- **Original missing test:** Construct a case where loop-level `OPTIMAL` would be too optimistic and verify postsolve downgrades it.
- **Current conclusion:** Fresh postsolve dual residuals protect the final status.

#### GAP-3: `HasNanImpl<true>` Early Return

- **Related issue:** Medium-priority review finding that `HasNanImpl<true>` uses a plain loop instead of `StaticFor`.
- **Coverage:** `MiniMatrixTest.Kernel_HasNanAndAllFiniteBoundaryCases`.
- **Original missing test:** Verify NaN detection at the first element, last element, and no-NaN cases.
- **Current conclusion:** The current behavior is correct. The plain loop remains an intentional early-return trade-off.

---

### Medium Priority

#### GAP-4: Mehrotra `mu_curr == 0`

- **Related issue:** Low-priority review finding that `avg_complementarity_gap / current_mu` had no zero guard.
- **Historical coverage:** `BarrierResidualContractTest.MehrotraTargetMuHandlesZeroCurrentMu`.
- **Original missing test:** Directly call the target-mu helper with `mu_curr = 0.0` and verify finite output.
- **Current conclusion:** Later review triage moved this from a hot-path guard to
  an invariant-boundary policy: nonpositive barrier state is outside valid
  solver invariants and should be prevented at config/build/initialization
  boundaries. See
  [`../reviews/review-fix-plan-2026-05-02-deep.md`](../reviews/review-fix-plan-2026-05-02-deep.md).

#### GAP-5: Tiny L1 Weight Initialization

- **Related issue:** Low-priority review finding that tiny L1 weights could create an empty clamp range.
- **Coverage:** `SoftConstraintTest.L1TinyWeightInitializationStaysFinite`.
- **Original missing test:** Build an L1 soft constraint with a very small weight and verify finite positive slack and dual values.
- **Current conclusion:** Initialization remains finite. Weights below the current active L1 threshold are not treated as active L1 dual-box constraints.

#### GAP-6: Armijo Directional-Derivative Cancellation

- **Related issue:** Low-priority review finding that the `eps_alpha` floor of `1e-10` might be too small for finite-difference directional derivatives.
- **Coverage:** Not converted into a strict regression test.
- **Reason:** A synthetic test would likely be fragile and overfit to floating-point noise.
- **Next trigger:** Add a benchmark or reproducer if a real merit/Armijo anomaly appears.

#### GAP-7: Stale Primal Infeasibility After Line Search

- **Related issue:** Low-priority review finding that `should_stop_after_line_search_` used a pre-line-search primal infeasibility snapshot.
- **Coverage:** `BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal`.
- **Original missing test:** Construct a loop-level optimality mismatch and verify postsolve reclassifies the result.
- **Current conclusion:** Fresh postsolve primal residuals protect final status. The in-loop early-exit condition still uses a snapshot by design.

#### GAP-8: Filter Circular Buffer Capacity

- **Related issue:** Informational finding that filter history has a fixed capacity of 1024.
- **Coverage:** `LineSearchTest.FilterHistoryWrapsAtFixedCapacity`.
- **Original missing test:** Stress filter history beyond capacity and verify wraparound behavior.
- **Current conclusion:** Direct line-search stress covers 1100 accepted entries and verifies the fixed capacity remains 1024.

---

### Low Priority

#### GAP-9: `all_finite` And `has_nan` Consistency

- **Related issue:** Low-priority review finding that `all_finite` does not use the policy system.
- **Coverage:** `MiniMatrixTest.Kernel_HasNanAndAllFiniteBoundaryCases`.
- **Original missing test:** Verify consistency between NaN and finite checks.
- **Current conclusion:** NaN at first and last element is detected. Inf is non-finite but intentionally not reported by `has_nan`.

#### GAP-10: Restoration Penalty `rho` Scale

- **Related issue:** Low-priority review finding that restoration uses a hardcoded penalty scale.
- **Coverage:** No direct scale-sensitive coverage.
- **Original missing test:** Compare restoration behavior across differently scaled infeasible constraints.
- **Current conclusion:** Keep as an algorithm and strategy issue. It needs a multi-scale infeasible benchmark before changing the fixed penalty.

#### GAP-11: Snapshot Stats Field Consistency

- **Related issue:** Medium-priority review finding that snapshot stats used raw `out.write` while config used helper functions.
- **Coverage:** Existing round-trip tests cover the overall snapshot, but not individual stat fields.
- **Original missing test:** Assert exact `iterations`, `total_cost`, and `mu` values after capture, save, and load.
- **Current conclusion:** Deferred. Snapshot/replay is not solver logging and should keep a clear format policy.

#### GAP-12: Merit Backtracking Failure-Path Assertion

- **Related issue:** Low-priority review finding that the failure path used a vacuous assertion.
- **Coverage:** `LineSearchTest.MeritFunctionBacktracking`.
- **Fix:** Failure path now asserts the solver exits with an expected non-success status instead of accepting any result.
- **Current conclusion:** This was a test-quality gap, not a solver behavior change.

---

## Test Quality Notes

### Demo Tests Versus Regression Tests

`test_bugfixes.cpp` includes improvement demos such as:

- `MeritLS_ArmijoRejectsTinyImprovement`
- `MeritLS_ArmijoVsSimpleDecrease_Iterations`

These demonstrate behavior but are not strict regression tests. If either behavior becomes a contract, extract focused assertions into a formal regression test.

### Implicit Integrator Jacobian Coverage

`ImplicitIntegratorTest::JacobiansMatchFiniteDifferenceForAllImplicitSchemes`
now verifies A/B Jacobians against finite differences for Backward Euler,
Implicit Midpoint, and Gauss-Legendre.

### Reference Config Coverage

Reference/default agreement now covers:

- Simple unconstrained QP.
- L1 soft-constraint QP.
- Implicit-integrator QP.

Residual-cost codegen is covered separately by `tests/minimodel/test_residual_costs.py`.

---

## Remaining Recommendations

1. **GAP-6, Armijo cancellation:** Do not add a brittle synthetic test yet. Wait for a real merit/Armijo anomaly or benchmark reproducer.
2. **Snapshot stats:** Keep deferred until the snapshot format, semantics, and compatibility policy are redesigned together.
3. **Restoration rho scale:** Keep as a strategy problem until multi-scale infeasible benchmarks exist.
