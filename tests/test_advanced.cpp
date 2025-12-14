#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"

using namespace minisolver;

// Since CarModel constraints are fixed at compile time (Hard/Soft defined in gen script),
// we cannot easily switch between Hard and Soft in C++ tests without regenerating code.
// However, the current generated code has Hard constraints (default).
// To test Soft Constraints logic, we need a model with soft constraints.
// Since we don't want to regenerate code inside C++ test build, we can:
// 1. Manually create a MockModel struct that mimics generated code but with Soft Constraint flags.
// 2. Or just verify the logic path in RiccatiSolver using a MockModel.

// Let's define a simple Mock Model with 1 state, 1 control, 1 constraint (Soft L2)
struct SoftModel {
    static const int NX=1;
    static const int NU=1;
    static const int NC=1;
    static const int NP=0;
    
    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    
    // Soft Constraint Config: Type 2 (L2), Weight 10.0
    static constexpr std::array<double, NC> constraint_weights = {10.0};
    static constexpr std::array<int, NC> constraint_types = {2};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        return x + u * dt; // x_next = x + u*dt
    }

    template<typename T>
    static void compute_dynamics(KnotPointV2<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        kp.f_resid(0) = kp.x(0) + kp.u(0)*dt;
        kp.A(0,0) = 1.0;
        kp.B(0,0) = dt;
    }

    template<typename T>
    static void compute_constraints(KnotPointV2<T,NX,NU,NC,NP>& kp) {
        // g(x,u) = u - 1.0 <= 0
        kp.g_val(0) = kp.u(0) - 1.0;
        kp.C.setZero();
        kp.D(0,0) = 1.0;
    }
    
    template<typename T>
    static void compute_cost_exact(KnotPointV2<T,NX,NU,NC,NP>& kp) {
        // Min u^2
        kp.cost = kp.u(0)*kp.u(0);
        kp.r(0) = 2.0*kp.u(0);
        kp.R(0,0) = 2.0;
    }
    
    // GN placeholder
    template<typename T>
    static void compute_cost_gn(KnotPointV2<T,NX,NU,NC,NP>& kp) {
        compute_cost_exact(kp);
    }
    
    // Wrappers expected by Solver
    template<typename T>
    static void compute(KnotPointV2<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost_exact(kp); // Default to Exact
    }

    template<typename T>
    static void compute_exact(KnotPointV2<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost_exact(kp);
    }
};

TEST(AdvancedFeaturesTest, SoftConstraintL2) {
    // Problem: Min u^2 s.t. u <= 1 (Soft L2, w=10)
    // Objective: u^2 + 10 * max(0, u-1)^2
    // If we force u to violate, say target requires u=2.
    // Let's set x_target = 10, N=1, dt=1. x0=0.
    // x1 = u. Cost = (x1 - 10)^2.
    // This requires adding state cost.
    // Let's just rely on the solver to minimize the Soft Constraint penalty vs nothing?
    // Wait, if objective is u^2, optimal u is 0. Constraint u<=1 satisfied.
    // We need to force violation.
    // Let's modify cost manually in test? No, compute_cost overwrites.
    
    // Use the solver on SoftModel but with initial guess violating constraint?
    // And see if it reduces violation but allows some?
    
    int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG; // Enable debug logs
    config.max_iters = 500;
    
    MiniSolver<SoftModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    
    // To force violation, we need a cost that pulls u > 1.
    // But SoftModel cost is u^2 (pulls to 0).
    // Let's modify SoftModel's cost function for this test?
    // We can't easily.
    // But we can check if the Riccati solver computes the correct Dual Regularization term.
    // Sigma should be 1 / (s/lam + 1/w).
    // If w=10.
    
    // Initialize solver with u=2 (violated)
    for(int k=0; k<N; ++k) solver.set_control_guess(k, "u", 2.0);
    solver.rollout_dynamics();
    
    // To ensure convergence to 0, we need tighter tolerances or more iterations.
    // L2 Soft Constraint with w=10 and u_init=2 is a tough problem for cold start if mu isn't tuned.
    // But it should move towards 0.
    // Let's check that it reduces u significantly below 1.0 (boundary).
    // And returns SOLVED or FEASIBLE.
    
    SolverStatus status = solver.solve();
    
    // Accept SOLVED, FEASIBLE, or MAX_ITER (as long as it didn't crash)
    bool acceptable = (status == SolverStatus::OPTIMAL || 
                       status == SolverStatus::FEASIBLE);
    EXPECT_TRUE(acceptable);
    
    // Check if control is pulled towards 0 (Optimal for u^2 + penalty)
    // The previous run stuck at ~0.07 with MAX_ITER. This is very close to 0 given the difficulty.
    // 0.07 is well within feasible region (< 1).
    // Penalty is 0 there. Gradient is 2*u = 0.14. 
    // It should go to 0.
    // We relax the check to < 0.1 to account for barrier/numerical residue.
    EXPECT_LT(solver.get_control(0, 0), 1.0e-3);
}

// Test GN
TEST(AdvancedFeaturesTest, GaussNewtonOption) {
    SolverConfig config;
    config.hessian_approximation = HessianApproximation::GAUSS_NEWTON;
    config.print_level = PrintLevel::DEBUG; // Enable logging to debug failure
    
    // Robust Config
    config.barrier_strategy = BarrierStrategy::MONOTONE; 
    config.line_search_type = LineSearchType::FILTER;
    config.max_iters = 100;
    
    config.enable_slack_reset = false; 
    config.enable_feasibility_restoration = true;

    MiniSolver<CarModel, 50> solver(10, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    
    // Fix: Set parameters to avoid L=0 and ensure feasibility at start
    for(int k=0; k<=10; ++k) {
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        solver.set_parameter(k, "obs_x", 100.0); // Move obstacle far away
        solver.set_parameter(k, "obs_rad", 1.0);
    }
    
    // Set Target to make problem well-posed (non-zero gradient)
    for(int k=0; k<=10; ++k) {
        solver.set_parameter(k, "v_ref", 1.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_steer", 0.1); // Add steer weight to avoid singularity
        solver.set_parameter(k, "w_acc", 0.1);   // Add acc weight
    }
    
    // Just run it to ensure no crash and correct dispatch
    SolverStatus status = solver.solve();
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
}

// Test SQP-RTI
TEST(AdvancedFeaturesTest, SQP_RTI) {
    SolverConfig config;
    config.enable_rti = true;
    config.max_iters = 1; // RTI usually does 1 iter
    config.print_level = PrintLevel::NONE;
    
    MiniSolver<CarModel, 50> solver(10, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    
    // Fix: Set parameters to avoid L=0 and ensure feasibility at start
    for(int k=0; k<=10; ++k) {
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        solver.set_parameter(k, "obs_x", 100.0); // Move obstacle far away
        solver.set_parameter(k, "obs_rad", 1.0);
    }
    
    // Solve
    SolverStatus status = solver.solve();
    
    // Should return SOLVED immediately (RTI treats one step as done)
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
    EXPECT_EQ(solver.current_iter, 1);
}

