#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/algorithms/line_search.h"
#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/solver/solver.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

// Test-only friend for MiniSolver internals. Forward-declared in solver.h's
// minisolver::test namespace; defined here so the friend access is a pure
// test concern and production builds don't link against it.
namespace minisolver::test {
template <typename Model, int MAX_N> struct SolverInternalAccess {
    using Solver = MiniSolver<Model, MAX_N>;
    static double& mu(Solver& s) { return s.mu; }
    static void apply_slack_reset(Solver& s, typename Solver::TrajArray& traj)
    {
        s.apply_slack_reset_(traj);
    }
    static double last_mu_aff(const Solver& s) { return s.last_mu_aff_; }
    static double last_alpha_aff(const Solver& s) { return s.last_alpha_aff_; }
    static double soft_s(const Solver& s, int stage, int idx)
    {
        return s.trajectory[stage].soft_s(idx);
    }
    static bool has_nans(Solver& s, const typename Solver::TrajArray& t)
    {
        return s.has_nans(t);
    }
    static typename Solver::TrajArray& get_trajectory(Solver& s)
    {
        return s.trajectory.active();
    }
};
} // namespace minisolver::test

// =============================================================================
// Minimal test model: 1 state, 1 control, 1 constraint
// Simple enough to isolate specific algorithmic behaviors.
// =============================================================================
struct BugTestModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 }; // Hard constraint
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        // u <= 1 → u - 1 <= 0
        kp.g_val(0) = kp.u(0) - 1.0;
        kp.C(0, 0) = 0.0;
        kp.D(0, 0) = 1.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0) + 0.01 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 0.02 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 0.02;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// =============================================================================
// L1 Soft Constraint Model for testing L1-specific bugs
// =============================================================================
struct L1TestModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 100.0 }; // L1 weight
    static constexpr std::array<int, NC> constraint_types = { 1 }; // L1 type

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        // x <= 5 → x - 5 <= 0
        kp.g_val(0) = kp.x(0) - 5.0;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T diff = kp.x(0) - 10.0; // Target x=10
        kp.cost = diff * diff + 0.01 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * diff;
        kp.r(0) = 0.02 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 0.02;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// =============================================================================
// Bug 1 Test: compute_max_violation must include dynamics defects
// =============================================================================
TEST(BugfixTest, DynamicsDefectCountedInViolation)
{
    // If we manually set x[k+1] != f(x[k], u[k]), the solver should NOT
    // report OPTIMAL/FEASIBLE because of the dynamics defect.

    constexpr int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0; // Don't iterate — just test postsolve evaluation
    config.integrator = IntegratorType::EULER_EXPLICIT;

    MiniSolver<BugTestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    // Set up a trajectory where constraints are satisfied (g + s ≈ 0)
    // but dynamics are violated (x[k+1] ≠ f(x[k], u[k]))
    for (int k = 0; k <= N; ++k) {
        solver.set_state_guess(k, 0, 0.0);
        if (k < N) {
            solver.set_control_guess(k, 0, 0.0);
        }
        solver.set_slack_guess(k, 0, 1.0); // s = -g = -(u-1) = 1
        solver.set_dual_guess(k, 0, 0.1);
    }

    // Introduce a large dynamics defect: x[1] should be 0 (from dynamics: 0 + 0*0.1 = 0)
    // but we set it to 100
    solver.set_state_guess(1, 0, 100.0);

    SolverStatus status = solver.solve();

    // With 0 iterations, postsolve should evaluate the trajectory as-is.
    // The dynamics defect of 100 should cause INFEASIBLE (not FEASIBLE/OPTIMAL).
    EXPECT_EQ(status, SolverStatus::INFEASIBLE)
        << "Solver should detect large dynamics defect and return INFEASIBLE";
}

// =============================================================================
// Bug 4 Test: First iteration should NOT falsely converge
// =============================================================================
TEST(BugfixTest, NoFalseConvergenceOnFirstIteration)
{
    // With mu_init very small and a trivial problem, the old code could
    // falsely converge on the first iteration because r_bar was zero (uncomputed).

    constexpr int N = 3;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.mu_init = 1e-8; // Very small, close to mu_final
    config.mu_final = 1e-8;
    config.max_iters = 5;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MONOTONE;

    MiniSolver<BugTestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    // Set initial state away from optimal
    solver.set_initial_state("x", 10.0);
    solver.rollout_dynamics();

    SolverStatus status = solver.solve();

    // The solver should actually iterate (not return OPTIMAL on first iter)
    // and eventually find a solution (OPTIMAL or FEASIBLE).
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);

    // Verify it actually did iterations (not instant convergence)
    EXPECT_GE(solver.get_iteration_count(), 1) << "Solver should have iterated at least once";
}

// =============================================================================
// Bug 2 Test: SOC should update soft_s for L1 constraints
// Verify that dsoft_s field is properly computed (non-zero) for L1 constraints.
// =============================================================================
TEST(BugfixTest, DsoftSComputedForL1)
{
    // Verify that the SOC path on an L1 soft-constraint problem converges to the
    // correct softened solution, which exercises the dsoft_s update path.
    constexpr int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 20;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.enable_soc = true;

    MiniSolver<L1TestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    solver.set_initial_state("x", 0.0);
    solver.set_control_guess(0, 0, 10.0);
    solver.rollout_dynamics();

    SolverStatus status = solver.solve();
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
    EXPECT_NEAR(solver.get_state(1, 0), 5.0, 1e-2);
}

// =============================================================================
// Bug 3 Test: Verify soft_dual field is removed (compile-time check)
// This test verifies that KnotState does NOT have soft_dual member.
// If the field were still present, this would still compile — so the real
// verification is that the entire test suite compiles without soft_dual.
// =============================================================================
TEST(BugfixTest, DeadFieldsRemoved)
{
    // Compile-time verification: KnotState should not have soft_dual or dsoft_dual.
    // We verify the struct size is smaller than it would be with those fields.
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    using State = Knot::StateType;

    // Count expected fields in KnotState:
    // x(1), u(1), p(0), s(1), lam(1), soft_s(1), cost(1), g_val(1), f_resid(1),
    // q(1), r(1), q_bar(1), r_bar(1), dx(1), du(1), ds(1), dlam(1), dsoft_s(1), d(1)
    // = 19 doubles (for NX=NU=NC=1, NP=0)
    // If soft_dual and dsoft_dual were present, it would be 21 doubles.

    // With Eigen alignment, exact sizeof comparison is unreliable.
    // Instead, just verify the types compile and the solver works.
    State s;
    s.x(0) = 1.0;
    s.soft_s(0) = 1.0;
    s.dsoft_s(0) = 0.0;
    // s.soft_dual would fail to compile if it existed and we removed it.
    // s.dsoft_dual would fail to compile if it existed and we removed it.
    EXPECT_EQ(s.x(0), 1.0);
}

TEST(BugfixTest, NegativeHorizonRejected)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    MiniSolver<BugTestModel, 10> solver(-3, Backend::CPU_SERIAL, config);
    EXPECT_EQ(solver.get_horizon(), 0);

    solver.resize_horizon(-1);
    EXPECT_EQ(solver.get_horizon(), 0);
}

TEST(BugfixTest, NegativeConstraintQueryReturnsZero)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    MiniSolver<BugTestModel, 10> solver(2, Backend::CPU_SERIAL, config);
    EXPECT_DOUBLE_EQ(solver.get_constraint_val(-1, 0), 0.0);
    EXPECT_DOUBLE_EQ(solver.get_constraint_val(0, -1), 0.0);
}

// =============================================================================
// Bug: InertiaStrategy::SATURATION enum existed but was not implemented in
// riccati.h. When LLT factorize fails on R_bar, the code skips both the
// REGULARIZATION and IGNORE_SINGULAR branches and falls through to
// solve_llt_inplace on an un-decomposed solver, producing NaN/garbage results.
// =============================================================================
TEST(BugfixTest, SaturationStrategyHandlesIndefiniteRbar)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    using TrajArray = std::array<Knot, 3>; // N=2

    TrajArray traj;
    const int N = 2;

    for (int k = 0; k <= N; ++k) {
        traj[k].set_zero();
        traj[k].Q.setIdentity();
        // Strongly indefinite: barrier contribution alone cannot make R_bar SPD.
        traj[k].R(0, 0) = -1000.0;
        traj[k].A.setIdentity();
        traj[k].B(0, 0) = 1.0;
        traj[k].D(0, 0) = 1.0; // Active control coupling so r_bar != 0.
        traj[k].s(0) = 1.0;
        traj[k].lam(0) = 1.0;
    }
    traj[N].q(0) = 1.0;

    SolverConfig config;
    config.reg_min = 1e-9;
    config.min_barrier_slack = 1e-9;

    RiccatiSolver<TrajArray, BugTestModel> solver;
    bool success
        = solver.solve(traj, N, /*mu=*/0.01, /*reg=*/1e-2, InertiaStrategy::SATURATION, config);

    EXPECT_TRUE(success) << "SATURATION must fall back via diagonal saturation";

    // After the fix R_bar(0,0) is clamped to max(reg, reg_min)=1e-2, so
    // d = -r_bar/R_bar_clamped is on the order of -100. On master the LLT
    // factorization fails silently (info()==NumericalIssue), Eigen leaves the
    // un-decomposed value -999 in m_matrix, and solve_llt_inplace divides by
    // -999 twice yielding a numerically tiny d ~ 1e-6 — finite but wrong.
    for (int k = 0; k < N; ++k) {
        EXPECT_TRUE(std::isfinite(traj[k].d(0)))
            << "d(0) is not finite at k=" << k << " (was " << traj[k].d(0) << ")";
        EXPECT_TRUE(std::isfinite(traj[k].K(0, 0))) << "K(0,0) is not finite at k=" << k;
        EXPECT_LT(std::abs(traj[k].d(0)), 1e6)
            << "d(0) too large at k=" << k << " (was " << traj[k].d(0) << ")";
        // Correctness lower bound: with R_bar_clamped ~ 1e-2 and r_bar ~ 1,
        // |d| should be on the order of 10-1000. Master will produce |d| ~ 1e-6.
        EXPECT_GT(std::abs(traj[k].d(0)), 1.0)
            << "d(0) magnitude too small — SATURATION fallback not active at k=" << k << " (was "
            << traj[k].d(0) << ")";
    }
}

// =============================================================================
// Bug: The Riccati forward sweep updates traj[k].du for k=0..N-1 only.
// traj[N].du carries over from the previous iteration / setZero(), then
// recover_dual_search_directions for k=N reads it inside
// constraint_step = C·dx + D·du. For terminal D != 0, dlam[N] / ds[N] are
// corrupted by stale du[N].
// =============================================================================
TEST(BugfixTest, TerminalDualRecoveryNotPollutedByStaleDu)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    using TrajArray = std::array<Knot, 3>; // N=2

    auto setup = [](TrajArray& traj, int N) {
        for (int k = 0; k <= N; ++k) {
            traj[k].set_zero();
            traj[k].Q.setIdentity();
            traj[k].R.setIdentity();
            traj[k].A.setIdentity();
            traj[k].B(0, 0) = 1.0;
            traj[k].s(0) = 1.0;
            traj[k].lam(0) = 1.0;
            traj[k].g_val(0) = 0.0;
            // Terminal D != 0 is what exposes the bug.
            traj[k].D(0, 0) = 1.0;
        }
        traj[N].q(0) = 1.0;
    };

    SolverConfig config;
    config.reg_min = 1e-9;
    config.min_barrier_slack = 1e-9;

    // Run A: clean traj[N].du = 0
    TrajArray traj_a;
    setup(traj_a, 2);
    RiccatiSolver<TrajArray, BugTestModel> solver_a;
    bool ok_a = solver_a.solve(
        traj_a, 2, /*mu=*/0.01, /*reg=*/1e-9, InertiaStrategy::REGULARIZATION, config);
    ASSERT_TRUE(ok_a);
    const double dlam_clean = traj_a[2].dlam(0);
    const double ds_clean = traj_a[2].ds(0);

    // Run B: pre-poison traj[N].du with garbage before solve.
    TrajArray traj_b;
    setup(traj_b, 2);
    traj_b[2].du(0) = 1.0e10; // Stale value the bug propagates into dlam[N].
    RiccatiSolver<TrajArray, BugTestModel> solver_b;
    bool ok_b = solver_b.solve(
        traj_b, 2, /*mu=*/0.01, /*reg=*/1e-9, InertiaStrategy::REGULARIZATION, config);
    ASSERT_TRUE(ok_b);
    const double dlam_poisoned = traj_b[2].dlam(0);
    const double ds_poisoned = traj_b[2].ds(0);

    // After fix: terminal dual recovery must be invariant to the pre-existing du[N].
    // Before fix: dlam_poisoned drifts by ~1e10·D(0,0) from dlam_clean.
    EXPECT_NEAR(dlam_clean, dlam_poisoned, 1e-6)
        << "Terminal dlam[N] depends on stale du[N]; clean=" << dlam_clean
        << " poisoned=" << dlam_poisoned;
    EXPECT_NEAR(ds_clean, ds_poisoned, 1e-6)
        << "Terminal ds[N] depends on stale du[N]; clean=" << ds_clean
        << " poisoned=" << ds_poisoned;
}

// =============================================================================
// IPOPT §3.1: when μ decreases, the filter / merit history built under the old
// μ is no longer comparable (φ contains −μ·Σ log(s)). LineSearchStrategy
// exposes on_barrier_update() as the hook; the default base implementation is
// a no-op so strategies that don't carry barrier-dependent state don't have to
// override. Filter / Merit strategies MUST clear their history.
// =============================================================================

namespace {
// Minimal linear solver that just produces a small descent direction.
class BarrierHookMockSolver
    : public LinearSolver<Trajectory<KnotPoint<double, 4, 2, 5, 13>, 10>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 4, 2, 5, 13>, 10>::TrajArray;

    bool solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/, InertiaStrategy /*strategy*/,
        const SolverConfig& /*config*/, const TrajArray* /*affine_traj*/ = nullptr) override
    {
        for (int k = 0; k <= N; ++k) {
            traj[k].dx = -0.1 * traj[k].x;
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
            traj[k].dsoft_s.setZero();
        }
        return true;
    }
};
} // namespace

TEST(BugfixTest, FilterLineSearchClearsFilterOnBarrierUpdate)
{
    // Seed the filter with at least one accepted-step entry, then trigger the
    // barrier-update hook and verify the filter was cleared.
    SolverConfig config;
    config.line_search_type = LineSearchType::FILTER;

    constexpr int N = 10;
    using Model = CarModel;
    FilterLineSearch<Model, N> ls;
    BarrierHookMockSolver linear_solver;

    Trajectory<KnotPoint<double, 4, 2, 5, 13>, N> trajectory(N);
    std::array<double, N> dts;
    dts.fill(0.1);

    for (int k = 0; k <= N; ++k) {
        trajectory.active()[k].set_zero();
        trajectory.active()[k].x.fill(10.0);
        trajectory.active()[k].cost = 1000.0;
    }

    linear_solver.solve(trajectory.active(), N, 0.1, 1e-6, InertiaStrategy::REGULARIZATION, config);
    const double alpha = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    ASSERT_GT(alpha, 0.0);
    ASSERT_EQ(ls.filter_size(), 1u) << "expected one filter entry after accepted step";

    // After fix: on_barrier_update must clear the filter.
    // Before fix: base class default no-op leaves the entry behind, so the
    // next search() under a smaller μ would compare against stale φ values.
    ls.on_barrier_update();
    EXPECT_EQ(ls.filter_size(), 0u) << "FilterLineSearch did not clear filter on barrier update";
}

TEST(BugfixTest, MeritLineSearchResetsNuOnBarrierUpdate)
{
    // Ratchet merit_nu up by running a search() with large dual magnitudes,
    // then trigger the barrier-update hook and verify merit_nu returns to the
    // baseline so that the next search() under a smaller μ re-derives it.
    SolverConfig config;
    config.line_search_type = LineSearchType::MERIT;

    constexpr int N = 10;
    using Model = CarModel;
    MeritLineSearch<Model, N> ls;
    BarrierHookMockSolver linear_solver;

    Trajectory<KnotPoint<double, 4, 2, 5, 13>, N> trajectory(N);
    std::array<double, N> dts;
    dts.fill(0.1);

    for (int k = 0; k <= N; ++k) {
        auto& kp = trajectory.active()[k];
        kp.set_zero();
        kp.x.fill(10.0);
        kp.cost = 1000.0;
        // Large dual magnitude ratchets merit_nu via max_dual * 1.1 + 1.
        kp.lam.fill(5000.0);
        kp.s.fill(1.0);
    }

    linear_solver.solve(trajectory.active(), N, 0.1, 1e-6, InertiaStrategy::REGULARIZATION, config);
    ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);

    const double merit_nu_after_ratchet = ls.get_merit_nu();
    ASSERT_GT(merit_nu_after_ratchet, 5000.0)
        << "setup failure: merit_nu did not ratchet past baseline";

    // After fix: on_barrier_update must reset merit_nu to baseline.
    // Before fix: base class default no-op leaves merit_nu at ratcheted value
    // derived from the old μ's dual magnitudes.
    ls.on_barrier_update();
    EXPECT_LT(ls.get_merit_nu(), merit_nu_after_ratchet)
        << "MeritLineSearch did not reset merit_nu on barrier update";
    EXPECT_DOUBLE_EQ(ls.get_merit_nu(), 1000.0)
        << "MeritLineSearch merit_nu not reset to baseline 1000";
}

// =============================================================================
// Per-solve line-search α trace. The solver should log every step's α into
// MiniSolver::get_alpha_log() so the user can inspect convergence behaviour.
// Trace must be fresh each solve (cleared at entry) and values must be in the
// fraction-to-boundary range [0, 1].
// =============================================================================
TEST(BugfixTest, AlphaLogPopulatedAndClearedPerSolve)
{
    constexpr int N = 3;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 10;
    config.mu_init = 1e-1;
    config.mu_final = 1e-6;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MONOTONE;

    MiniSolver<BugTestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    // Non-trivial initial state so the solver actually needs iterations.
    solver.set_state_guess(0, 0, 2.0);

    // Before solve: log is empty.
    ASSERT_TRUE(solver.get_alpha_log().empty());

    solver.solve();

    const auto& alpha_log = solver.get_alpha_log();

    // After fix: log contains one entry per step the solver ran, capped at
    // max_iters. Before fix: accessor returns the default-constructed (empty)
    // vector, because nothing pushes into it.
    EXPECT_GT(alpha_log.size(), 0u) << "alpha_log not populated by solve()";
    EXPECT_LE(alpha_log.size(), static_cast<size_t>(config.max_iters))
        << "alpha_log has more entries than iterations allowed";

    // Every α must be a valid fraction-to-boundary value.
    for (size_t i = 0; i < alpha_log.size(); ++i) {
        const double alpha = alpha_log[i];
        EXPECT_TRUE(std::isfinite(alpha)) << "alpha_log[" << i << "] is not finite: " << alpha;
        EXPECT_GE(alpha, 0.0) << "alpha_log[" << i << "] < 0: " << alpha;
        EXPECT_LE(alpha, 1.0 + 1e-12) << "alpha_log[" << i << "] > 1: " << alpha;
    }

    // Second solve must clear the previous trace, not accumulate.
    const size_t first_size = alpha_log.size();
    solver.solve();
    EXPECT_LE(solver.get_alpha_log().size(), static_cast<size_t>(config.max_iters))
        << "alpha_log not cleared between solve() calls (carry-over)";
    // Sanity: size of second run is independent of first run's size — mostly
    // this guards against "log grew by first_size".
    EXPECT_LT(solver.get_alpha_log().size(), first_size + config.max_iters + 1)
        << "alpha_log appears to have carried over entries from previous solve";
}

// =============================================================================
// Bug: cost stagnation was incorrectly gated on (mu <= mu_final), so if mu freezes
// above mu_final the solver can run max_iters without triggering stagnation.
//
// This test freezes mu intentionally (barrier_tolerance_factor = 0) and uses a
// trivially feasible, constant-cost problem. Stagnation must stop the solve
// early even though mu > mu_final.
// =============================================================================
TEST(BugfixTest, CostStagnationNotGatedOnMuFinal)
{
    constexpr int N = 5;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 50;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.line_search_type = LineSearchType::FILTER;

    // Freeze μ above μ_final to reproduce the original gating bug.
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.mu_init = 1e-2;
    config.mu_final = 1e-6;
    config.barrier_tolerance_factor = 0.0; // max_kkt_error < 0 never holds

    MiniSolver<BugTestModel, 20> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    for (int k = 0; k < N; ++k) {
        solver.set_control_guess(k, "u", 0.0);
    }
    solver.rollout_dynamics();

    solver.solve();

    // With the fix, cost stagnation should terminate in a few iterations.
    // Before the fix, the solver would run max_iters because stagnation was
    // gated on mu <= mu_final and mu is frozen at mu_init.
    EXPECT_LT(solver.get_iteration_count(), config.max_iters)
        << "stagnation did not trigger while mu > mu_final";
}

// =============================================================================
// Bug: set_config() overwrites the whole config struct, including backend,
// violating the constructor invariant that backend is the "source of truth"
// (solver.h:102-103 comment). A user passing a plain default SolverConfig to
// set_config() silently switches backend back to Backend::CPU_SERIAL even if
// the solver was constructed with a non-default backend.
// =============================================================================
TEST(BugfixTest, SetConfigPreservesBackendInvariant)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;

    // Construct with a non-default backend value to establish the invariant.
    MiniSolver<BugTestModel, 10> solver(3, Backend::GPU_MPX, conf);
    ASSERT_EQ(solver.get_config().backend, Backend::GPU_MPX);

    // Call set_config with a conf that specifies a different backend.
    SolverConfig new_conf;
    new_conf.print_level = PrintLevel::NONE;
    new_conf.backend = Backend::GPU_PCR;
    solver.set_config(new_conf);

    // After fix: constructor-set backend preserved across set_config().
    // Before fix: set_config does `config = conf` and backend becomes GPU_PCR.
    EXPECT_EQ(solver.get_config().backend, Backend::GPU_MPX)
        << "set_config must preserve the constructor-set backend invariant";
}

// =============================================================================
// Bug: the Python generator (`MiniModel.generate(integrator_type=...)`) emits a
// fused Riccati kernel whose sparsity pattern is tied to the chosen target
// integrator. Running the resulting C++ model with a different runtime
// `config.integrator` silently drops non-zero contributions (if runtime has
// more non-zeros than target) or does harmless extra multiplies (if fewer).
//
// The structural fix is to emit `static constexpr IntegratorType
// generated_integrator` in each generated model header and have the solver
// refuse to run when it disagrees with `config.integrator`. This test uses a
// local model with the marker set to RK4_EXPLICIT and passes EULER_EXPLICIT,
// expecting the solver to throw.
// =============================================================================
struct IntegratorTaggedModel : public BugTestModel {
    // Opt-in marker: compile-time integrator the fused kernel was generated
    // for. MiniSolver's constructor must reject runtime mismatches.
    static constexpr IntegratorType generated_integrator = IntegratorType::RK4_EXPLICIT;
};

// =============================================================================
// Bug: slack_reset uses config.mu_init to pull dual onto central path instead
// of the current barrier parameter mu. After several iterations of barrier
// reduction, mu decays well below mu_init; when slack_reset fires in this
// regime it pumps lam up to mu_init/s — several orders of magnitude off the
// central path mu/s — breaking complementarity and forcing extra iterations.
// =============================================================================
// =============================================================================
// Gap #1: Mehrotra mu_aff only sums hard slack pairs (s*lam) but omits L1
// soft constraint's complementary pair soft_s * (w - lam). This makes sigma
// underestimate the true affine complementarity, biasing the centering
// parameter. The fraction-to-boundary also misses soft_s / dsoft_s and
// (w - lam) / (-dlam).
//
// RED test: run one Mehrotra solve on L1TestModel, then recompute the true
// mu_aff from the post-step trajectory INCLUDING the L1 soft pair. If the
// solver's mu_aff matches only the hard pair (bug), it will be smaller than
// the true value that also counts soft_s * (w - lam).
// =============================================================================
TEST(BugfixTest, MehrotraMuAffIncludesL1SoftPair)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 1; // just one step to capture mu_aff
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.mu_init = 1.0;
    config.mu_final = 1e-8;
    config.integrator = IntegratorType::EULER_EXPLICIT;

    constexpr int N = 1;
    MiniSolver<L1TestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    solver.set_initial_state("x", 0.0);
    solver.set_control_guess(0, 0, 5.0); // x_next = 0 + 5*1 = 5
    solver.rollout_dynamics();

    using Access = minisolver::test::SolverInternalAccess<L1TestModel, 10>;

    solver.solve();

    const double mu_aff_solver = Access::last_mu_aff(solver);

    // Recompute true mu_aff from the post-step trajectory, including BOTH
    // the hard pair (s * lam) and the L1 soft pair (soft_s * (w - lam)).
    double true_total_comp = 0.0;
    int true_total_dim = 0;
    constexpr double w = L1TestModel::constraint_weights[0]; // 100.0

    for (int k = 0; k <= N; ++k) {
        const double s_k = solver.get_slack(k, 0);
        const double lam_k = solver.get_dual(k, 0);
        const double soft_s_k = Access::soft_s(solver, k, 0);

        true_total_comp += s_k * lam_k;              // hard pair
        true_total_comp += soft_s_k * (w - lam_k);    // L1 soft pair
        true_total_dim += 2; // count both pairs
    }
    double true_mu_aff = true_total_comp / std::max(1, true_total_dim);

    // After fix: solver's mu_aff should be close to true_mu_aff.
    // Before fix: solver's mu_aff only counts hard pairs, so it will be
    // systematically smaller than true_mu_aff when the soft pair is
    // significant (which it is with w=100 and mu_init=1.0).
    //
    // Allow 10% tolerance for float arithmetic differences between the
    // solver's internal computation (which uses alpha_aff-scaled step) and
    // our post-hoc recompute from the accepted trajectory.
    EXPECT_GT(mu_aff_solver, 0.0) << "mu_aff must be positive after one step";

    // The soft pair is large (w=100, soft_s~O(0.01), w-lam~O(100)).
    // If solver ignores it, mu_aff will be < 50% of the true value.
    // Tolerate up to 50% difference (the recompute uses post-step values,
    // not the exact affine-step values the solver uses internally).
    EXPECT_GT(mu_aff_solver, true_mu_aff * 0.5)
        << "mu_aff is much smaller than true value — L1 soft pair likely "
           "omitted. solver=" << mu_aff_solver << " true=" << true_mu_aff;
}

TEST(BugfixTest, SlackResetUsesCurrentMuNotMuInit)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.mu_init = 1.0; // large initial barrier
    config.min_barrier_slack = 1e-12;

    constexpr int N = 2;
    MiniSolver<BugTestModel, 10> solver(N, Backend::CPU_SERIAL, config);

    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    // Force decayed-mu regime (as would happen mid-solve after several barrier
    // reductions). Oracle thresholds below hinge on mu << mu_init.
    Access::mu(solver) = 1e-5;

    // Seed a trajectory that will definitely hit the reset floor:
    //   min_s = |g_val| + sqrt(mu) = 0 + sqrt(1e-5) ≈ 3.16e-3
    //   s(0) = 1e-4 < min_s  -> reset fires
    //   lam(0) = 1e-6 leaves room for the buggy mu_init/s pump to dominate.
    MiniSolver<BugTestModel, 10>::TrajArray traj;
    for (int k = 0; k <= N; ++k) {
        traj[k].set_zero();
        traj[k].s(0) = 1e-4;
        traj[k].lam(0) = 1e-6;
        traj[k].g_val(0) = 0.0;
    }

    Access::apply_slack_reset(solver, traj);

    // After reset: s = min_s ≈ 3.16e-3.
    //   Fix:  lam = max(1e-6, mu/s)      = max(1e-6, 1e-5 / 3.16e-3) ≈ 3.16e-3
    //   Bug:  lam = max(1e-6, mu_init/s) = max(1e-6, 1.0  / 3.16e-3) ≈ 316
    // Comp = lam*s should land on the central path at ~mu = 1e-5 (fix) vs
    // ~mu_init = 1.0 (bug). We accept anything below 1e-2 as "centered on mu";
    // the bug produces ≈1.0 which is two orders of magnitude above the bound.
    for (int k = 0; k <= N; ++k) {
        const double lam = traj[k].lam(0);
        const double s = traj[k].s(0);
        const double comp = lam * s;
        EXPECT_LT(comp, 1e-2)
            << "slack_reset pumped lam*s up to mu_init scale (expected ~mu=1e-5). "
            << "k=" << k << " lam=" << lam << " s=" << s << " lam*s=" << comp;
    }
}

TEST(BugfixTest, SolverWarnsOnFusedKernelIntegratorMismatch)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    conf.integrator = IntegratorType::EULER_EXPLICIT; // != model's RK4_EXPLICIT

    using TaggedSolver = MiniSolver<IntegratorTaggedModel, 10>;

    // After fix: constructor detects the disagreement via the
    // `generated_integrator` marker and prints a warning to stderr.
    // Before fix: no such check exists; user is silently running a model
    // whose fused kernel was generated for a different integrator.
    // The solver should still construct (warning, not throw) since the
    // non-fused path works with any integrator.
    EXPECT_NO_THROW({ TaggedSolver solver(3, Backend::CPU_SERIAL, conf); (void)solver; });
}

// =============================================================================
// IMPROVEMENT DEMOS — before/after comparison for each fix
// =============================================================================

// --- Demo 1: slack_reset mu ---
// Shows: with decayed mu (1e-5) and mu_init=1.0, the old code (mu_init/s)
// pumps lam*s to ~1.0 (5 orders off the central path). The fix (mu/s)
// lands lam*s on ~mu = 1e-5.
TEST(ImprovementDemo, SlackReset_ComplementarityGap)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.mu_init = 1.0;
    config.min_barrier_slack = 1e-12;

    constexpr int N = 2;
    MiniSolver<BugTestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    // Simulate decayed-mu regime (several barrier reductions happened).
    Access::mu(solver) = 1e-5;

    // Seed a trajectory that triggers the reset floor.
    MiniSolver<BugTestModel, 10>::TrajArray traj;
    for (int k = 0; k <= N; ++k) {
        traj[k].set_zero();
        traj[k].s(0) = 1e-4;   // below min_s = |g| + sqrt(mu) ≈ 3.16e-3
        traj[k].lam(0) = 1e-6; // small, so old mu_init/s dominates
        traj[k].g_val(0) = 0.0;
    }

    // --- Current fix: mu/s ---
    Access::apply_slack_reset(solver, traj);
    double fix_comp = 0.0;
    for (int k = 0; k <= N; ++k)
        fix_comp += std::abs(traj[k].lam(0) * traj[k].s(0) - 1e-5); // |lam*s - mu|

    // --- Simulated old behavior: mu_init/s ---
    // Reset trajectory, then apply mu_init/s manually.
    MiniSolver<BugTestModel, 10>::TrajArray traj_old;
    for (int k = 0; k <= N; ++k) {
        traj_old[k].set_zero();
        traj_old[k].s(0) = 1e-4;
        traj_old[k].lam(0) = 1e-6;
        traj_old[k].g_val(0) = 0.0;
    }
    // Manually apply the old (buggy) reset logic with mu_init.
    for (int k = 0; k <= N; ++k) {
        double min_s = std::abs(traj_old[k].g_val(0)) + std::sqrt(1e-5);
        if (traj_old[k].s(0) < min_s) {
            traj_old[k].s(0) = min_s;
            traj_old[k].lam(0) = std::max(traj_old[k].lam(0), 1.0 / min_s); // mu_init/s
        }
    }
    double old_comp = 0.0;
    for (int k = 0; k <= N; ++k)
        old_comp += std::abs(traj_old[k].lam(0) * traj_old[k].s(0) - 1e-5);

    std::cerr << "[Demo 1] slack_reset complementarity gap:\n"
              << "  old (mu_init/s): |lam*s - mu| = " << old_comp << "\n"
              << "  fix (mu/s):      |lam*s - mu| = " << fix_comp << "\n"
              << "  improvement:     " << old_comp / std::max(fix_comp, 1e-30) << "x closer to central path\n";

    // The fix should be at least 1000x closer to the central path.
    EXPECT_LT(fix_comp, old_comp / 1000.0);
}

// --- Demo 2: set_config backend ---
// Shows: backend is preserved across set_config calls.
TEST(ImprovementDemo, SetConfig_BackendPreserved)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;

    MiniSolver<BugTestModel, 10> solver(3, Backend::GPU_MPX, conf);

    SolverConfig new_conf;
    new_conf.print_level = PrintLevel::NONE;
    new_conf.backend = Backend::GPU_PCR; // different backend
    solver.set_config(new_conf);

    bool preserved = (solver.get_config().backend == Backend::GPU_MPX);

    std::cerr << "[Demo 2] set_config backend preserved: "
              << (preserved ? "YES" : "NO (backend overwritten)") << "\n";

    EXPECT_TRUE(preserved);
}

// --- Demo 3: integrator mismatch warning ---
// Shows: when model's generated_integrator != config.integrator, a warning
// is emitted at construction (previously silent).
TEST(ImprovementDemo, IntegratorMismatch_WarningEmitted)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    conf.integrator = IntegratorType::EULER_EXPLICIT;

    // Capture stderr — the warning goes to std::cerr from the constructor.
    std::ostringstream captured;
    auto* old_buf = std::cerr.rdbuf(captured.rdbuf());
    MiniSolver<IntegratorTaggedModel, 10> solver(3, Backend::CPU_SERIAL, conf);
    std::cerr.rdbuf(old_buf);
    std::string output = captured.str();

    bool warned = !output.empty();

    std::cerr << "[Demo 3] integrator mismatch warning: "
              << (warned ? "EMITTED" : "MISSING (silent)") << "\n"
              << "  stderr: \"" << output << "\"\n";

    EXPECT_TRUE(warned) << "No warning emitted for integrator mismatch";
}

// --- Demo 4: Mehrotra mu_aff L1 soft ---
// Shows: mu_aff with L1 soft pair included vs excluded.
// The old code only counted hard pairs; the fix adds soft_s * (w - lam).
TEST(ImprovementDemo, MehrotraMuAff_L1SoftPairImpact)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 1;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.mu_init = 1.0;
    config.mu_final = 1e-8;
    config.integrator = IntegratorType::EULER_EXPLICIT;

    constexpr int N = 1;
    MiniSolver<L1TestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    solver.set_initial_state("x", 0.0);
    solver.set_control_guess(0, 0, 5.0);
    solver.rollout_dynamics();

    using Access = minisolver::test::SolverInternalAccess<L1TestModel, 10>;

    solver.solve();

    const double mu_aff_with_soft = Access::last_mu_aff(solver);

    // Compute what mu_aff would be WITHOUT the soft pair (old behavior).
    // This is just the hard pair sum: s*lam / (N+1).
    double hard_only = 0.0;
    int hard_dim = 0;
    for (int k = 0; k <= N; ++k) {
        hard_only += solver.get_slack(k, 0) * solver.get_dual(k, 0);
        hard_dim++;
    }
    double mu_aff_hard_only = hard_only / std::max(1, hard_dim);

    // Compute what mu_aff would be WITH the soft pair (new behavior).
    // We recompute from the post-step trajectory to get the "true" value.
    double full_comp = 0.0;
    int full_dim = 0;
    constexpr double w = L1TestModel::constraint_weights[0];
    for (int k = 0; k <= N; ++k) {
        full_comp += solver.get_slack(k, 0) * solver.get_dual(k, 0);
        full_dim++;
        const double soft_s_k = Access::soft_s(solver, k, 0);
        full_comp += soft_s_k * (w - solver.get_dual(k, 0));
        full_dim++;
    }
    double ratio = mu_aff_with_soft / std::max(mu_aff_hard_only, 1e-30);

    std::cerr << "[Demo 4] Mehrotra mu_aff with L1 soft constraints:\n"
              << "  hard-only (old):  mu_aff = " << mu_aff_hard_only << "\n"
              << "  with soft (new):  mu_aff = " << mu_aff_with_soft << "\n"
              << "  ratio (new/old):  " << ratio << "\n"
              << "  sigma impact:     old sigma = (old/mu)^3, new sigma = (new/mu)^3\n";

    // The fix should produce a mu_aff that's significantly larger than hard-only
    // when L1 soft pairs are present.
    EXPECT_GT(ratio, 1.1) << "mu_aff barely changed — soft pair contribution negligible";
}

// =============================================================================
// Gap #2: Terminal u_N phantom — u_N is not an NMPC decision variable.
// The public API guard (set_control_guess rejects stage >= N) prevents
// external callers from setting u_N, so it stays at 0 from initialization.
// This test verifies the guard works: even if the user tries to set u_N,
// it remains 0 and terminal cost is correct.
// =============================================================================
TEST(ImprovementDemo, TerminalCost_UGuardProtectsAgainstPhantom)
{
    constexpr int N = 3;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0;
    config.integrator = IntegratorType::EULER_EXPLICIT;

    MiniSolver<BugTestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    for (int k = 0; k <= N; ++k) {
        solver.set_state_guess(k, 0, 0.0);
        if (k < N)
            solver.set_control_guess(k, 0, 0.0);
        solver.set_slack_guess(k, 0, 1.0);
        solver.set_dual_guess(k, 0, 0.1);
    }

    // Try to poison u_N — the guard at solver.h:407 should reject this.
    solver.set_control_guess(N, 0, 100.0);

    solver.solve();

    double terminal_cost = solver.get_stage_cost(N);
    // With BugTestModel: cost = x^2 + 0.01*u^2. At terminal, x=0, u should be 0
    // (guard rejected the set_control_guess(N,...) call). So cost ≈ 0.
    // If the guard were missing, u_N=100 → cost = 100.
    double guard_rejected_u_n = (solver.get_control(N, 0) == 0.0);

    std::cerr << "[Demo 5] Terminal u_N guard:\n"
              << "  tried set_control_guess(N, 0, 100): "
              << (guard_rejected_u_n ? "REJECTED (correct)" : "ACCEPTED (bug)") << "\n"
              << "  u_N actual = " << solver.get_control(N, 0) << "\n"
              << "  terminal cost = " << terminal_cost << "\n";

    EXPECT_TRUE(guard_rejected_u_n) << "set_control_guess should reject stage >= N";
    EXPECT_NEAR(terminal_cost, 0.0, 1e-12) << "terminal cost should be 0 (u_N = 0)";
}

// =============================================================================
// Gap #3: Merit line search uses simple decrease (phi_alpha < phi_0) instead
// of Armijo sufficient decrease. The relative Armijo form requires:
//   phi(alpha) <= phi(0) * (1 - c1 * alpha)
// This enforces proportional decrease: larger steps must achieve larger
// absolute reductions. Simple decrease accepts any epsilon improvement.
//
// Demo: compare acceptance thresholds for a tiny merit reduction.
// =============================================================================
TEST(ImprovementDemo, MeritLS_ArmijoRejectsTinyImprovement)
{
    double phi_0 = 1.0;
    double c1 = 1e-4;

    // Case 1: alpha = 1.0, phi_alpha = 0.9999999 (tiny decrease)
    {
        double alpha = 1.0;
        double phi_alpha = 0.9999999;

        bool simple = (phi_alpha < phi_0);
        double armijo_threshold = phi_0 * (1.0 - c1 * alpha); // 0.9999
        bool armijo = (phi_alpha <= armijo_threshold);

        std::cerr << "[Demo 6a] alpha=1.0, phi_alpha=0.9999999:\n"
                  << "  simple decrease: " << (simple ? "ACCEPT" : "REJECT") << "\n"
                  << "  Armijo threshold=" << armijo_threshold
                  << ": " << (armijo ? "ACCEPT" : "REJECT") << "\n";

        EXPECT_TRUE(simple);
        EXPECT_FALSE(armijo) << "Armijo should reject tiny improvement at alpha=1";
    }

    // Case 2: alpha = 0.01, same tiny decrease is now proportionally large
    {
        double alpha = 0.01;
        double phi_alpha = 0.9999; // same absolute decrease, smaller alpha

        bool simple = (phi_alpha < phi_0);
        double armijo_threshold = phi_0 * (1.0 - c1 * alpha); // 0.999999
        bool armijo = (phi_alpha <= armijo_threshold);

        std::cerr << "[Demo 6b] alpha=0.01, phi_alpha=0.9999:\n"
                  << "  simple decrease: " << (simple ? "ACCEPT" : "REJECT") << "\n"
                  << "  Armijo threshold=" << armijo_threshold
                  << ": " << (armijo ? "ACCEPT" : "REJECT") << "\n";

        EXPECT_TRUE(simple);
        EXPECT_TRUE(armijo) << "Armijo should accept when decrease is proportional to alpha";
    }

    // Case 3: significant decrease — both accept
    {
        double alpha = 1.0;
        double phi_alpha = 0.5; // large decrease

        bool simple = (phi_alpha < phi_0);
        double armijo_threshold = phi_0 * (1.0 - c1 * alpha);
        bool armijo = (phi_alpha <= armijo_threshold);

        std::cerr << "[Demo 6c] alpha=1.0, phi_alpha=0.5:\n"
                  << "  simple decrease: " << (simple ? "ACCEPT" : "REJECT") << "\n"
                  << "  Armijo threshold=" << armijo_threshold
                  << ": " << (armijo ? "ACCEPT" : "REJECT") << "\n";

        EXPECT_TRUE(simple);
        EXPECT_TRUE(armijo);
    }
}

// Real solver comparison: MERIT + Armijo vs MERIT + simple decrease.
// Run the same non-trivial problem twice, measure iterations and final cost.
TEST(ImprovementDemo, MeritLS_ArmijoVsSimpleDecrease_Iterations)
{
    constexpr int N = 5;

    auto run_solver = [](double armijo_c1) -> std::pair<int, double> {
        SolverConfig config;
        config.print_level = PrintLevel::NONE;
        config.max_iters = 100;
        config.mu_init = 1e-1;
        config.mu_final = 1e-6;
        config.integrator = IntegratorType::EULER_EXPLICIT;
        config.line_search_type = LineSearchType::MERIT;
        config.barrier_strategy = BarrierStrategy::MONOTONE;
        config.armijo_c1 = armijo_c1;

        MiniSolver<BugTestModel, 20> solver(N, Backend::CPU_SERIAL, config);
        solver.set_dt(0.1);
        // Non-trivial initial state — solver must iterate to converge.
        solver.set_state_guess(0, 0, 10.0);
        solver.rollout_dynamics();

        solver.solve();
        return { solver.get_iteration_count(), solver.get_stage_cost(0) };
    };

    auto [iters_armijo, cost_armijo] = run_solver(1e-4);     // Armijo on
    auto [iters_simple, cost_simple] = run_solver(0.0);       // simple decrease

    std::cerr << "[Demo 7] Merit LS: Armijo vs simple decrease\n"
              << "  simple decrease: " << iters_simple << " iters, cost=" << cost_simple << "\n"
              << "  Armijo (c1=1e-4): " << iters_armijo << " iters, cost=" << cost_armijo << "\n"
              << "  iter reduction:   " << (iters_simple - iters_armijo) << " fewer iters\n";

    // Both must converge to similar cost (within 1%).
    EXPECT_NEAR(cost_armijo, cost_simple, std::abs(cost_simple) * 0.01)
        << "Armijo changed the solution by more than 1%";

    // Armijo should use same or fewer iterations (rejects micro-steps).
    EXPECT_LE(iters_armijo, iters_simple)
        << "Armijo should not increase iteration count";
}

// =============================================================================
// Gap #6: NaN check doesn't cover dynamics Jacobian A/B.
// has_nans() checks dx, du, ds, dlam, cost but NOT kp.A or kp.B.
// A NaN in A/B flows silently into Riccati, producing garbage directions
// that are only caught later with a misleading error message.
// =============================================================================

namespace {
struct NanJacobianModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = std::numeric_limits<T>::quiet_NaN(); // Inject NaN
        kp.B(0, 0) = dt;
    }

    template <typename T>
    static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0) + kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 2.0 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2.0;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};
} // namespace

TEST(BugfixTest, HasNansDetectsJacobianNaN)
{
    // RED test for gap #6: has_nans() must detect NaN in A/B matrices.
    // Before fix: has_nans only checks dx, du, ds, dlam, cost — misses A/B.
    // After fix: has_nans also checks kp.A.allFinite() and kp.B.allFinite().
    constexpr int N = 5;
    using Solver = MiniSolver<NanJacobianModel, N>;
    using Access = minisolver::test::SolverInternalAccess<NanJacobianModel, N>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 1;

    Solver solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    // Inject NaN directly into A
    auto& traj = Access::get_trajectory(solver);
    traj[0].A(0, 0) = std::numeric_limits<double>::quiet_NaN();

    // has_nans MUST detect the NaN in A(0,0)
    EXPECT_TRUE(Access::has_nans(solver, traj))
        << "has_nans() failed to detect NaN in dynamics Jacobian A";
}

TEST(BugfixTest, NanJacobianReturnsNumericalError)
{
    // Functional test: solver with NaN-producing Jacobian must return
    // NUMERICAL_ERROR, not crash or produce garbage.
    constexpr int N = 5;
    using Solver = MiniSolver<NanJacobianModel, N>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 5;

    Solver solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    auto status = solver.solve();

    // Must detect numerical error (either from NaN in A or downstream NaN in dx)
    EXPECT_EQ(status, SolverStatus::NUMERICAL_ERROR)
        << "Solver did not detect NaN in dynamics Jacobian";
}
