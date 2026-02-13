/**
 * @file test_features.cpp
 * @brief Tests for individual solver features: cost stagnation, parameter
 *        persistence, GPU backend fallback, and iterative refinement.
 */
#include <gtest/gtest.h>
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include <array>

using namespace minisolver;

// =============================================================================
// Model: Flat Cost for stagnation termination test
// =============================================================================
struct FlatCostModel {
    static const int NX=1;
    static const int NU=1;
    static const int NC=1;
    static const int NP=0;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {0.0};
    static constexpr std::array<int, NC> constraint_types = {0};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& /*u*/,
                                   const MSVec<T, NP>& /*p*/, double /*dt*/, IntegratorType /*type*/) {
        return x;
    }

    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double /*dt*/) {
        kp.f_resid(0) = kp.x(0);
        kp.A(0,0) = 1.0;
        kp.B(0,0) = 0.0;
        double w = 1e-7;
        kp.cost = w * kp.x(0) * kp.x(0);
        kp.Q(0,0) = 2*w;
        kp.q(0) = 2*w * kp.x(0);
        kp.R.setIdentity();
        kp.g_val(0) = -10.0 - kp.x(0);
        kp.C(0,0) = -1.0;
    }
    
    template<typename T> static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T> static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T> static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) { compute(kp, type, dt); }
    template<typename T> static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
};

// =============================================================================
// Feature: Cost Stagnation Termination
// =============================================================================
TEST(FeaturesTest, CostStagnationTermination) {
    int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.tol_grad = 1e-12;
    config.tol_con = 1e-12;
    config.tol_mu = 1e-12;
    config.mu_final = 1e-9;
    config.tol_cost = 1e-5; 
    config.max_iters = 50; 
    
    MiniSolver<FlatCostModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0);
    
    SolverStatus status = solver.solve();
    
    EXPECT_EQ(status, SolverStatus::OPTIMAL);
    EXPECT_LT(solver.current_iter, config.max_iters);
}

// =============================================================================
// Feature: Parameter Persistence Through Solve (Ghost Cost Bug prevention)
// =============================================================================
TEST(FeaturesTest, ParameterPersistenceCheck) {
    int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.max_iters = 1;
    
    MiniSolver<CarModel, 10> solver(N, Backend::CPU_SERIAL, config);
    
    double magic_val = 123.456;
    solver.set_parameter(2, "x_ref", magic_val);
    
    EXPECT_DOUBLE_EQ(solver.get_parameter(2, "x_ref"), magic_val);
    EXPECT_DOUBLE_EQ(solver.trajectory.active()[2].p(1), magic_val);
    
    solver.solve();
    
    double val_after = solver.get_parameter(2, "x_ref");
    EXPECT_DOUBLE_EQ(val_after, magic_val) << "Parameter lost after solve() iteration (Ghost Cost Bug)";
    EXPECT_DOUBLE_EQ(solver.trajectory.candidate()[2].p(1), magic_val) << "Candidate buffer parameter out of sync";
}

// =============================================================================
// Feature: GPU Backend Fallback (CPU-only build should not crash)
// =============================================================================
TEST(FeaturesTest, GPUBackendFallback) {
    int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.backend = Backend::GPU_MPX; 
    
    MiniSolver<CarModel, 10> solver(N, config.backend, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    
    SolverStatus status = solver.solve();
    
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
}

// =============================================================================
// Feature: Iterative Refinement (runs without crash, maintains convergence)
// =============================================================================
TEST(FeaturesTest, IterativeRefinement) {
    int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.integrator = IntegratorType::RK4_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.inertia_strategy = InertiaStrategy::REGULARIZATION;
    config.enable_iterative_refinement = true;
    
    MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("v", 0.0);
    
    for(int k=0; k<=N; ++k) {
        solver.set_parameter(k, "v_ref", 1.0);
        solver.set_parameter(k, "x_ref", k * 0.1); 
        solver.set_parameter(k, "y_ref", 0.0);
        solver.set_parameter(k, "obs_x", 100.0);
        solver.set_parameter(k, "obs_y", 100.0);
        solver.set_parameter(k, "obs_rad", 1.0);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        solver.set_parameter(k, "w_pos", 10.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_theta", 0.1);
        solver.set_parameter(k, "w_acc", 0.1);
        solver.set_parameter(k, "w_steer", 1.0);
    }
    
    solver.rollout_dynamics();
    SolverStatus status = solver.solve();
    
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
}
