#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include <array>
#include <iostream>
#include <cmath>

using namespace minisolver;

// ==========================================
// 1. Interface Model (Benchmark)
// Uses built-in L1/L2 soft constraint logic
// ==========================================
struct InterfaceModel {
    static const int NX=1;
    static const int NU=1;
    static const int NC=1;
    static const int NP=0;

    static std::array<double, NC> constraint_weights; 
    static std::array<int, NC> constraint_types;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        return x + u * dt; 
    }

    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0,0) = 1.0; kp.B(0,0) = dt;
    }

    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        // Constraint: x <= 5.0
        kp.g_val(0) = kp.x(0) - 5.0; 
        kp.C(0,0) = 1.0; kp.D(0,0) = 0.0;
    }

    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T diff = kp.x(0) - 10.0; // Target x = 10
        kp.cost = diff*diff + 1e-4 * kp.u(0)*kp.u(0); 
        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.Q(0,0) = 2.0; kp.R(0,0) = 2e-4;
    }
    
    template<typename T> static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_exact(kp); }
    template<typename T> static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_exact(kp); }
    template<typename T> static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt); compute_constraints(kp); compute_cost(kp);
    }
};

std::array<double, 1> InterfaceModel::constraint_weights = {0.0};
std::array<int, 1> InterfaceModel::constraint_types = {0};


// ==========================================
// 2. Manual L1 Model
// Explicitly adds 'slk' as a control variable
// ==========================================
struct ManualL1Model {
    static const int NX=1;
    static const int NU=2; // [u, slk]
    static const int NC=2; // [g-slk, -slk]
    static const int NP=0;
    
    static constexpr std::array<double, NC> constraint_weights = {0.0, 0.0};
    static constexpr std::array<int, NC> constraint_types = {0, 0};
    
    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u", "slk"};
    static constexpr std::array<const char*, NP> param_names = {};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt; // slk (u[1]) does not affect dynamics
        return x_next;
    }

    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0,0) = 1.0; 
        kp.B(0,0) = dt; kp.B(0,1) = 0.0; 
    }

    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x = kp.x(0);
        T slk = kp.u(1);
        
        // 1. x - 5 - slk <= 0
        kp.g_val(0) = x - 5.0 - slk;
        kp.C(0,0) = 1.0; 
        kp.D(0,0) = 0.0; kp.D(0,1) = -1.0; 
        
        // 2. -slk <= 0 (Non-negative slack)
        kp.g_val(1) = -slk;
        kp.C(1,0) = 0.0;
        kp.D(1,0) = 0.0; kp.D(1,1) = -1.0;
    }

    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T diff = kp.x(0) - 10.0;
        T slk = kp.u(1);
        double w = 1.0; // Fixed L1 Weight
        
        // Cost: (x-10)^2 + w*slk
        kp.cost = diff*diff + 1e-4*kp.u(0)*kp.u(0) + w * slk;
        
        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.r(1) = w; // Gradient w.r.t slk is constant w
        
        kp.Q(0,0) = 2.0;
        kp.R(0,0) = 2e-4; 
        kp.R(1,1) = 0.0; // Linear cost -> 0 Hessian
        kp.R(0,1) = 0.0; kp.R(1,0) = 0.0;
    }
    
    template<typename T> static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_exact(kp); }
    template<typename T> static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_exact(kp); }
    template<typename T> static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt); compute_constraints(kp); compute_cost(kp);
    }
};

// ==========================================
// 3. Manual L2 Model
// Uses 0.5 * w * slk^2 to match Interface
// ==========================================
struct ManualL2Model {
    static const int NX=1;
    static const int NU=2; 
    static const int NC=1; 
    static const int NP=0;
    
    static constexpr std::array<double, NC> constraint_weights = {0.0};
    static constexpr std::array<int, NC> constraint_types = {0};
    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u", "slk"};
    static constexpr std::array<const char*, NP> param_names = {};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt; 
        return x_next;
    }

    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0,0) = 1.0; 
        kp.B(0,0) = dt; kp.B(0,1) = 0.0; 
    }

    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x = kp.x(0);
        T slk = kp.u(1);
        // x - 5 - slk <= 0
        kp.g_val(0) = x - 5.0 - slk;
        kp.C(0,0) = 1.0; 
        kp.D(0,0) = 0.0; kp.D(0,1) = -1.0; 
    }

    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T diff = kp.x(0) - 10.0;
        T slk = kp.u(1);
        double w = 1.0; // Fixed L2 Weight
        
        // Cost: 0.5 * w * slk^2 (Matches MiniSolver Interface L2 formulation)
        kp.cost = diff*diff + 1e-4*kp.u(0)*kp.u(0) + 0.5 * w * slk * slk;
        
        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.r(1) = w * slk; 
        
        kp.Q(0,0) = 2.0;
        kp.R(0,0) = 2e-4; 
        kp.R(1,1) = w; 
        kp.R(0,1) = 0.0; kp.R(1,0) = 0.0;
    }
    
    template<typename T> static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_exact(kp); }
    template<typename T> static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_exact(kp); }
    template<typename T> static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt); compute_constraints(kp); compute_cost(kp);
    }
};


// ==========================================
// 4. Comparison Tests
// ==========================================

TEST(ComparisonTest, L1_SoftConstraint) {
    SolverConfig config;
    config.tol_con = 1e-4;
    config.max_iters = 100;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    
    // We use N=2 to ensure 'slk' (as a control input) is optimized at the intermediate node.
    // k=0: x0 -> u0 -> x1
    // k=1: x1 -> u1(slk) -> x2. Constraint applies here.
    int N = 2;

    // --- 1. Interface (Ground Truth) ---
    InterfaceModel::constraint_types[0] = 1; // L1
    InterfaceModel::constraint_weights[0] = 1.0; 
    
    MiniSolver<InterfaceModel, 5> solver_if(N, Backend::CPU_SERIAL, config);
    solver_if.set_dt(1.0);
    solver_if.set_initial_state("x", 0.0);
    solver_if.set_control_guess(0, "u", 10.0); // Reach 10 at k=1
    solver_if.set_control_guess(1, "u", 0.0);
    solver_if.rollout_dynamics();
    solver_if.solve();
    double x_if = solver_if.get_state(1, 0); // Check x at k=1
    
    // --- 2. Manual Model ---
    MiniSolver<ManualL1Model, 5> solver_man(N, Backend::CPU_SERIAL, config);
    solver_man.set_dt(1.0);
    solver_man.set_initial_state("x", 0.0);
    solver_man.set_control_guess(0, "u", 10.0);
    solver_man.set_control_guess(1, "u", 0.0);
    
    // Manual Initialization for k=1 (where x=10 approx)
    // x ~ 10. Constraint x - 5 - slk <= 0.
    // To satisfy, slk needs to be 5.
    double slk_init = 5.0; 
    solver_man.set_control_guess(1, "slk", slk_init);
    
    solver_man.rollout_dynamics();
    
    // [Init Dual Variables]
    // For L1: Stationarity implies Lambda = Weight = 1.0
    auto& traj = solver_man.trajectory.active();
    traj[1].s(0) = 0.01;  // Small slack for active constraint
    traj[1].lam(0) = 1.0; // L1 Dual = Weight
    
    traj[1].s(1) = slk_init; // Inactive non-negative constraint
    traj[1].lam(1) = 0.01;
    
    solver_man.is_warm_started = true;
    
    solver_man.solve();
    double x_man = solver_man.get_state(1, 0);
    
    std::cout << "[L1 Comparison N=2] Interface x1: " << x_if << " vs Manual x1: " << x_man << std::endl;
    
    // Theoretical Opt for L1: x = 9.5
    EXPECT_NEAR(x_if, 9.5, 1e-3);
    EXPECT_NEAR(x_if, x_man, 1e-3);
}

TEST(ComparisonTest, L2_SoftConstraint) {
    SolverConfig config;
    config.tol_con = 1e-4;
    config.max_iters = 100;
    config.integrator = IntegratorType::EULER_EXPLICIT; 
    config.barrier_strategy = BarrierStrategy::MEHROTRA; 

    int N = 2;

    // --- 1. Interface ---
    InterfaceModel::constraint_types[0] = 2; // L2
    InterfaceModel::constraint_weights[0] = 1.0; 
    
    MiniSolver<InterfaceModel, 5> solver_if(N, Backend::CPU_SERIAL, config);
    solver_if.set_dt(1.0);
    solver_if.set_initial_state("x", 0.0);
    solver_if.set_control_guess(0, "u", 10.0);
    solver_if.set_control_guess(1, "u", 0.0);
    solver_if.rollout_dynamics();
    solver_if.solve();
    double x_if = solver_if.get_state(1, 0);

    // --- 2. Manual Model ---
    MiniSolver<ManualL2Model, 5> solver_man(N, Backend::CPU_SERIAL, config);
    solver_man.set_dt(1.0);
    solver_man.set_initial_state("x", 0.0);
    solver_man.set_control_guess(0, "u", 10.0);
    solver_man.set_control_guess(1, "u", 0.0);
    
    // Initialize slk at k=1
    double slk_init = 5.0; 
    solver_man.set_control_guess(1, "slk", slk_init);
    
    solver_man.rollout_dynamics();
    
    // [Init Dual Variables]
    // For L2: Stationarity implies Lambda = Weight * Slack
    // Lam = 1.0 * 5.0 = 5.0
    auto& traj = solver_man.trajectory.active();
    traj[1].s(0) = 0.01;
    traj[1].lam(0) = 5.0; 
    
    solver_man.is_warm_started = true;
    
    solver_man.solve();
    double x_man = solver_man.get_state(1, 0);
    
    std::cout << "[L2 Comparison N=2] Interface x1: " << x_if << " vs Manual x1: " << x_man << std::endl;
    
    // Theoretical Opt for L2: x = 25/3 = 8.333...
    EXPECT_NEAR(x_if, 8.333, 1e-3);
    EXPECT_NEAR(x_if, x_man, 1e-3);
}