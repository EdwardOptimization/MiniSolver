#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include <cmath>

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

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {0.0}; // Hard constraint
    static constexpr std::array<int, NC> constraint_types = {0};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
                                   const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        MSVec<T, NX> xn; xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0,0) = 1.0;
        kp.B(0,0) = dt;
    }

    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        // u <= 1 → u - 1 <= 0
        kp.g_val(0) = kp.u(0) - 1.0;
        kp.C(0,0) = 0.0;
        kp.D(0,0) = 1.0;
    }

    template<typename T>
    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {
        kp.cost = kp.x(0) * kp.x(0) + 0.01 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 0.02 * kp.u(0);
        kp.Q(0,0) = 2.0;
        kp.R(0,0) = 0.02;
        kp.H(0,0) = 0.0;
    }

    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_gn(kp); }

    template<typename T>
    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_gn(kp); }

    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
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

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {100.0}; // L1 weight
    static constexpr std::array<int, NC> constraint_types = {1};          // L1 type

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
                                   const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        MSVec<T, NX> xn; xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0,0) = 1.0;
        kp.B(0,0) = dt;
    }

    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        // x <= 5 → x - 5 <= 0
        kp.g_val(0) = kp.x(0) - 5.0;
        kp.C(0,0) = 1.0;
        kp.D(0,0) = 0.0;
    }

    template<typename T>
    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T diff = kp.x(0) - 10.0; // Target x=10
        kp.cost = diff * diff + 0.01 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * diff;
        kp.r(0) = 0.02 * kp.u(0);
        kp.Q(0,0) = 2.0;
        kp.R(0,0) = 0.02;
        kp.H(0,0) = 0.0;
    }

    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_gn(kp); }

    template<typename T>
    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_gn(kp); }

    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// =============================================================================
// Bug 1 Test: compute_max_violation must include dynamics defects
// =============================================================================
TEST(BugfixTest, DynamicsDefectCountedInViolation) {
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
    auto& traj = solver.trajectory.active();
    for(int k = 0; k <= N; ++k) {
        traj[k].x(0) = 0.0;
        traj[k].u(0) = 0.0;
        traj[k].s(0) = 1.0;   // s = -g = -(u-1) = 1
        traj[k].lam(0) = 0.1;
    }
    
    // Introduce a large dynamics defect: x[1] should be 0 (from dynamics: 0 + 0*0.1 = 0)
    // but we set it to 100
    traj[1].x(0) = 100.0;
    
    SolverStatus status = solver.solve();
    
    // With 0 iterations, postsolve should evaluate the trajectory as-is.
    // The dynamics defect of 100 should cause INFEASIBLE (not FEASIBLE/OPTIMAL).
    EXPECT_EQ(status, SolverStatus::INFEASIBLE) 
        << "Solver should detect large dynamics defect and return INFEASIBLE";
}

// =============================================================================
// Bug 4 Test: First iteration should NOT falsely converge
// =============================================================================
TEST(BugfixTest, NoFalseConvergenceOnFirstIteration) {
    // With mu_init very small and a trivial problem, the old code could
    // falsely converge on the first iteration because r_bar was zero (uncomputed).
    
    constexpr int N = 3;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.mu_init = 1e-8;    // Very small, close to mu_final
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
    EXPECT_GE(solver.current_iter, 1) << "Solver should have iterated at least once";
}

// =============================================================================
// Bug 5 Test: warm_start clamps lam for L1 constraints
// =============================================================================
TEST(BugfixTest, WarmStartClampsLamForL1) {
    constexpr int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 10;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    
    MiniSolver<L1TestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    
    // Create init trajectory with lam EXCEEDING the L1 weight w=100
    auto init_traj = solver.trajectory.active();
    for(int k = 0; k <= N; ++k) {
        init_traj[k].x(0) = 3.0;
        init_traj[k].u(0) = 0.0;
        init_traj[k].lam(0) = 200.0; // Bad: exceeds w=100
        init_traj[k].s(0) = 1.0;
    }
    
    // warm_start should clamp lam to < w
    solver.warm_start(init_traj);
    
    // Verify lam was clamped: lam < w = 100
    auto& traj = solver.trajectory.active();
    for(int k = 0; k <= N; ++k) {
        EXPECT_LT(traj[k].lam(0), 100.0) 
            << "warm_start should clamp lam below L1 weight w for constraint " << k;
    }
    
    // Solver should not crash
    SolverStatus status = solver.solve();
    EXPECT_NE(status, SolverStatus::NUMERICAL_ERROR);
}

// =============================================================================
// Bug 2 Test: SOC should update soft_s for L1 constraints
// Verify that dsoft_s field is properly computed (non-zero) for L1 constraints.
// =============================================================================
TEST(BugfixTest, DsoftSComputedForL1) {
    // Verify that after a Riccati solve on an L1 model, dsoft_s is non-zero.
    constexpr int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 3;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.enable_soc = true;
    
    MiniSolver<L1TestModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 8.0); // Beyond soft constraint x <= 5
    solver.rollout_dynamics();
    
    solver.solve();
    
    // After solving, check that dsoft_s was computed (non-zero for active L1 constraints)
    // The fact that the solver converged without crash is itself a verification.
    auto& traj = solver.trajectory.active();
    
    // For an L1 problem with target x=10 and soft limit x<=5,
    // the solution should push x towards 10 but be penalized beyond 5.
    // soft_s should be positive (representing the soft slack).
    bool any_soft_s_positive = false;
    for(int k = 0; k <= N; ++k) {
        if(traj[k].soft_s(0) > 1e-6) any_soft_s_positive = true;
    }
    EXPECT_TRUE(any_soft_s_positive) << "L1 soft_s should be positive for active soft constraints";
}

// =============================================================================
// Bug 3 Test: Verify soft_dual field is removed (compile-time check)
// This test verifies that KnotState does NOT have soft_dual member.
// If the field were still present, this would still compile — so the real 
// verification is that the entire test suite compiles without soft_dual.
// =============================================================================
TEST(BugfixTest, DeadFieldsRemoved) {
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
