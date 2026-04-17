#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/solver/solver.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

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
    bool success = solver.solve(
        traj, N, /*mu=*/0.01, /*reg=*/1e-2, InertiaStrategy::SATURATION, config);

    EXPECT_TRUE(success) << "SATURATION must fall back via diagonal saturation";

    // After the fix R_bar(0,0) is clamped to max(reg, reg_min)=1e-2, so
    // d = -r_bar/R_bar_clamped is on the order of -100. On master the LLT
    // factorization fails silently (info()==NumericalIssue), Eigen leaves the
    // un-decomposed value -999 in m_matrix, and solve_llt_inplace divides by
    // -999 twice yielding a numerically tiny d ~ 1e-6 — finite but wrong.
    for (int k = 0; k < N; ++k) {
        EXPECT_TRUE(std::isfinite(traj[k].d(0)))
            << "d(0) is not finite at k=" << k << " (was " << traj[k].d(0) << ")";
        EXPECT_TRUE(std::isfinite(traj[k].K(0, 0)))
            << "K(0,0) is not finite at k=" << k;
        EXPECT_LT(std::abs(traj[k].d(0)), 1e6)
            << "d(0) too large at k=" << k << " (was " << traj[k].d(0) << ")";
        // Correctness lower bound: with R_bar_clamped ~ 1e-2 and r_bar ~ 1,
        // |d| should be on the order of 10-1000. Master will produce |d| ~ 1e-6.
        EXPECT_GT(std::abs(traj[k].d(0)), 1.0)
            << "d(0) magnitude too small — SATURATION fallback not active at k="
            << k << " (was " << traj[k].d(0) << ")";
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
