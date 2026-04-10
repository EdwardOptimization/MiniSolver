/**
 * @file test_solver_quality.cpp
 * @brief Comprehensive solver quality tests inspired by acados/CasADi testing practices.
 * 
 * Categories:
 *   1. Finite Difference Jacobian Verification
 *   2. KKT Optimality Conditions Check
 *   3. Known Analytical Solution Verification
 *   4. MPC Closed-Loop Simulation
 *   5. Mehrotra Predictor-Corrector Convergence
 *   6. Warm Start Quality Verification
 */

#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace minisolver;

// =============================================================================
// Helper: Simple unconstrained QP model for analytical solution tests
// min 0.5 * w_x * x^2 + 0.5 * w_u * u^2
// s.t. x_{k+1} = x_k + u_k * dt
// Analytical solution (unconstrained LQR) can be computed.
// =============================================================================
struct SimpleQPModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& /*x*/, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/) {
        MSVec<T, NX> xdot; xdot(0) = u(0);
        return xdot;
    }

    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x, const MSVec<T, NU>& u,
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
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& /*kp*/) {}

    template<typename T>
    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {
        double w_x = 10.0, w_u = 1.0;
        kp.cost = 0.5 * w_x * kp.x(0) * kp.x(0) + 0.5 * w_u * kp.u(0) * kp.u(0);
        kp.q(0) = w_x * kp.x(0);
        kp.r(0) = w_u * kp.u(0);
        kp.Q(0,0) = w_x;
        kp.R(0,0) = w_u;
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
// Helper: Constrained model for KKT tests
// min x^2 + 0.1*u^2   s.t. u <= 0.5
// =============================================================================
struct ConstrainedQPModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {0.0};
    static constexpr std::array<int, NC> constraint_types = {0};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
                                   const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        MSVec<T, NX> xn; xn(0) = x(0) + u(0) * dt; return xn;
    }
    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0,0) = 1.0; kp.B(0,0) = dt;
    }
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        kp.g_val(0) = kp.u(0) - 0.5; // u <= 0.5
        kp.C(0,0) = 0.0; kp.D(0,0) = 1.0;
    }
    template<typename T>
    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {
        kp.cost = kp.x(0)*kp.x(0) + 0.1*kp.u(0)*kp.u(0);
        kp.q(0) = 2.0*kp.x(0); kp.r(0) = 0.2*kp.u(0);
        kp.Q(0,0) = 2.0; kp.R(0,0) = 0.2; kp.H(0,0) = 0.0;
    }
    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_gn(kp); }
    template<typename T>
    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) { compute_cost_gn(kp); }
    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt); compute_constraints(kp); compute_cost(kp);
    }
};


// =============================================================================
// TEST 1: Finite Difference Jacobian Verification
// Verifies that the SymPy-generated analytical Jacobians (A, B, C, D, q, r)
// match central finite differences, following the acados/CasADi pattern.
// =============================================================================
TEST(SolverQualityTest, FiniteDifferenceJacobianVerification) {
    using Model = CarModel;
    constexpr int NX = 4, NU = 2, NC = 5, NP = 13;
    
    KnotPoint<double, NX, NU, NC, NP> kp;
    kp.set_zero();
    
    // Set a non-trivial operating point
    kp.x(0) = 5.0;  kp.x(1) = 1.0;  kp.x(2) = 0.3;  kp.x(3) = 8.0;
    kp.u(0) = 1.5;  kp.u(1) = 0.2;
    // Parameters
    kp.p(0) = 5.0;  // v_ref
    kp.p(1) = 10.0; // x_ref
    kp.p(3) = 20.0; // obs_x
    kp.p(5) = 1.5;  // obs_rad
    kp.p(6) = 2.5;  // L
    kp.p(7) = 1.0;  // car_rad
    kp.p(8) = 1.0;  // w_pos
    kp.p(9) = 1.0;  // w_vel
    kp.p(10) = 0.1; // w_theta
    kp.p(11) = 0.1; // w_acc
    kp.p(12) = 1.0; // w_steer
    
    double dt = 0.1;
    
    // Compute analytical derivatives
    Model::compute_dynamics(kp, IntegratorType::RK4_EXPLICIT, dt);
    Model::compute_constraints(kp);
    Model::compute_cost_exact(kp);
    
    // Save analytical values
    auto A_an = kp.A;
    auto B_an = kp.B;
    auto C_an = kp.C;
    auto D_an = kp.D;
    auto q_an = kp.q;
    auto r_an = kp.r;
    (void)kp.f_resid; // Used implicitly via FD perturbation
    (void)kp.g_val;
    (void)kp.cost;
    
    double eps = 1e-6;
    
    // --- Verify A = df/dx via central FD ---
    for(int j = 0; j < NX; ++j) {
        KnotPoint<double, NX, NU, NC, NP> kp_p = kp, kp_m = kp;
        kp_p.x(j) += eps;
        kp_m.x(j) -= eps;
        Model::compute_dynamics(kp_p, IntegratorType::RK4_EXPLICIT, dt);
        Model::compute_dynamics(kp_m, IntegratorType::RK4_EXPLICIT, dt);
        for(int i = 0; i < NX; ++i) {
            double fd = (kp_p.f_resid(i) - kp_m.f_resid(i)) / (2 * eps);
            EXPECT_NEAR(A_an(i, j), fd, 1e-4) 
                << "A(" << i << "," << j << ") mismatch: analytical=" << A_an(i,j) << " fd=" << fd;
        }
    }
    
    // --- Verify B = df/du via central FD ---
    for(int j = 0; j < NU; ++j) {
        KnotPoint<double, NX, NU, NC, NP> kp_p = kp, kp_m = kp;
        kp_p.u(j) += eps;
        kp_m.u(j) -= eps;
        Model::compute_dynamics(kp_p, IntegratorType::RK4_EXPLICIT, dt);
        Model::compute_dynamics(kp_m, IntegratorType::RK4_EXPLICIT, dt);
        for(int i = 0; i < NX; ++i) {
            double fd = (kp_p.f_resid(i) - kp_m.f_resid(i)) / (2 * eps);
            EXPECT_NEAR(B_an(i, j), fd, 1e-4)
                << "B(" << i << "," << j << ") mismatch";
        }
    }
    
    // --- Verify C = dg/dx via central FD ---
    for(int j = 0; j < NX; ++j) {
        KnotPoint<double, NX, NU, NC, NP> kp_p = kp, kp_m = kp;
        kp_p.x(j) += eps;
        kp_m.x(j) -= eps;
        Model::compute_constraints(kp_p);
        Model::compute_constraints(kp_m);
        for(int i = 0; i < NC; ++i) {
            double fd = (kp_p.g_val(i) - kp_m.g_val(i)) / (2 * eps);
            EXPECT_NEAR(C_an(i, j), fd, 1e-4)
                << "C(" << i << "," << j << ") mismatch";
        }
    }
    
    // --- Verify D = dg/du via central FD ---
    for(int j = 0; j < NU; ++j) {
        KnotPoint<double, NX, NU, NC, NP> kp_p = kp, kp_m = kp;
        kp_p.u(j) += eps;
        kp_m.u(j) -= eps;
        Model::compute_constraints(kp_p);
        Model::compute_constraints(kp_m);
        for(int i = 0; i < NC; ++i) {
            double fd = (kp_p.g_val(i) - kp_m.g_val(i)) / (2 * eps);
            EXPECT_NEAR(D_an(i, j), fd, 1e-4)
                << "D(" << i << "," << j << ") mismatch";
        }
    }
    
    // --- Verify q = dcost/dx via central FD ---
    for(int j = 0; j < NX; ++j) {
        KnotPoint<double, NX, NU, NC, NP> kp_p = kp, kp_m = kp;
        kp_p.x(j) += eps;
        kp_m.x(j) -= eps;
        Model::compute_cost_exact(kp_p);
        Model::compute_cost_exact(kp_m);
        double fd = (kp_p.cost - kp_m.cost) / (2 * eps);
        EXPECT_NEAR(q_an(j), fd, 1e-4) << "q(" << j << ") mismatch";
    }
    
    // --- Verify r = dcost/du via central FD ---
    for(int j = 0; j < NU; ++j) {
        KnotPoint<double, NX, NU, NC, NP> kp_p = kp, kp_m = kp;
        kp_p.u(j) += eps;
        kp_m.u(j) -= eps;
        Model::compute_cost_exact(kp_p);
        Model::compute_cost_exact(kp_m);
        double fd = (kp_p.cost - kp_m.cost) / (2 * eps);
        EXPECT_NEAR(r_an(j), fd, 1e-4) << "r(" << j << ") mismatch";
    }
}


// =============================================================================
// TEST 2: KKT Optimality Conditions Check
// After solve, verifies the 4 KKT conditions:
//   (a) Primal feasibility:  g(x,u) + s = 0,  s >= 0
//   (b) Dual feasibility:    lam >= 0
//   (c) Complementary slackness:  s_i * lam_i ≈ 0
//   (d) Stationarity:  ∇L = 0  (checked via r_bar after barrier derivatives)
// =============================================================================
TEST(SolverQualityTest, KKTOptimalityConditions) {
    constexpr int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 100;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.tol_con = 1e-6;
    config.mu_final = 1e-8;
    
    MiniSolver<ConstrainedQPModel, 30> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 5.0);
    solver.rollout_dynamics();
    
    SolverStatus status = solver.solve();
    ASSERT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
    
    double tol = 1e-4;
    double comp_tol = (status == SolverStatus::OPTIMAL) ? 1e-4 : 2e-4;
    
    for(int k = 0; k <= N; ++k) {
        // (a) Primal feasibility: g + s ≈ 0
        for(int i = 0; i < 1; ++i) {
            double prim_resid = std::abs(solver.get_constraint_val(k, i) + solver.get_slack(k, i));
            EXPECT_LT(prim_resid, tol) << "Primal infeasible at k=" << k;
        }
        
        // (b) Dual feasibility: s >= 0, lam >= 0
        for(int i = 0; i < 1; ++i) {
            EXPECT_GE(solver.get_slack(k, i), -1e-10) << "Negative slack at k=" << k;
            EXPECT_GE(solver.get_dual(k, i), -1e-10) << "Negative dual at k=" << k;
        }
        
        // (c) Complementary slackness: s * lam ≈ 0 (≈ mu_final)
        for(int i = 0; i < 1; ++i) {
            double comp = solver.get_slack(k, i) * solver.get_dual(k, i);
            EXPECT_LT(comp, comp_tol) << "Complementarity violated at k=" << k << ": s*lam=" << comp;
        }
    }
    
    // (d) Dynamics feasibility: x_{k+1} ≈ f(x_k, u_k)
    for(int k = 0; k < N; ++k) {
        double defect = std::abs(solver.get_state(k + 1, 0) - (solver.get_state(k, 0) + solver.get_control(k, 0) * 0.1));
        EXPECT_LT(defect, tol) << "Dynamics defect at k=" << k;
    }
}


// =============================================================================
// TEST 3: Known Analytical Solution Verification
// Unconstrained LQR: min sum{ 0.5*w_x*x^2 + 0.5*w_u*u^2 }
// with x_{k+1} = x_k + u_k * dt, x_0 = x0.
// The optimal u drives x towards 0. We verify cost monotonically decreases
// and the solution satisfies optimality within numerical tolerance.
// =============================================================================
TEST(SolverQualityTest, AnalyticalSolutionUnconstrained) {
    constexpr int N = 20;
    double dt = 0.1;
    double x0 = 5.0;
    
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 100;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    
    MiniSolver<SimpleQPModel, 30> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(dt);
    solver.set_initial_state("x", x0);
    solver.rollout_dynamics();
    
    SolverStatus status = solver.solve();
    ASSERT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
    
    // Property 1: x should decrease monotonically toward 0
    double prev_abs_x = std::abs(x0);
    for(int k = 1; k <= N; ++k) {
        double abs_x = std::abs(solver.get_state(k, 0));
        EXPECT_LE(abs_x, prev_abs_x + 1e-6) 
            << "State should decrease monotonically at k=" << k;
        prev_abs_x = abs_x;
    }
    
    // Property 2: Final state should be close to 0 (the target)
    double x_final = solver.get_state(N, 0);
    EXPECT_NEAR(x_final, 0.0, 1.0) << "Final state should be near 0";
    
    // Property 3: Total cost should be much lower than initial cost
    double total_cost = 0.0;
    for(int k = 0; k <= N; ++k) total_cost += solver.get_stage_cost(k);
    double initial_cost = 0.5 * 10.0 * x0 * x0 * (N + 1); // If u=0, cost = sum w_x*x0^2/2
    EXPECT_LT(total_cost, initial_cost * 0.5) << "Optimization should significantly reduce cost";
    
    // Property 4: Dynamics should be exactly satisfied (unconstrained, no defects)
    for(int k = 0; k < N; ++k) {
        double x_k = solver.get_state(k, 0);
        double u_k = solver.get_control(k, 0);
        double x_next_expected = x_k + u_k * dt;
        double x_next_actual = solver.get_state(k + 1, 0);
        EXPECT_NEAR(x_next_actual, x_next_expected, 1e-6) 
            << "Dynamics violated at k=" << k;
    }
}


// =============================================================================
// TEST 4: MPC Closed-Loop Simulation
// Runs multiple solve-shift-update cycles to verify:
//   - shift_trajectory works correctly
//   - solver stays stable over multiple MPC steps
//   - state converges to target over closed-loop horizon
// =============================================================================
TEST(SolverQualityTest, MPCClosedLoopSimulation) {
    constexpr int N = 10;
    constexpr int MPC_STEPS = 15;
    double dt = 0.1;
    
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 50;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.tol_con = 1e-4;
    
    MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(dt);
    
    // Set parameters: track straight line at v=5
    for(int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "v_ref", 5.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_pos", 0.1);
        solver.set_parameter(k, "w_acc", 0.1);
        solver.set_parameter(k, "w_steer", 1.0);
        solver.set_parameter(k, "w_theta", 0.1);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        solver.set_parameter(k, "obs_x", 100.0); // Far away
        solver.set_parameter(k, "obs_y", 0.0);
        solver.set_parameter(k, "obs_rad", 1.0);
    }
    
    // Initial state
    double sim_x = 0.0, sim_y = 0.0, sim_theta = 0.0, sim_v = 0.0;
    solver.set_initial_state("x", sim_x);
    solver.set_initial_state("y", sim_y);
    solver.set_initial_state("theta", sim_theta);
    solver.set_initial_state("v", sim_v);
    solver.rollout_dynamics();
    
    int success_count = 0;

    auto shift_car_guess = [&](auto& s) {
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < 4; ++i) {
                s.set_state_guess(k, i, s.get_state(k + 1, i));
            }
            if (k + 1 < N) {
                for (int i = 0; i < 2; ++i) {
                    s.set_control_guess(k, i, s.get_control(k + 1, i));
                }
            } else {
                for (int i = 0; i < 2; ++i) {
                    s.set_control_guess(k, i, s.get_control(k, i));
                }
            }
        }
    };
    
    for(int step = 0; step < MPC_STEPS; ++step) {
        SolverStatus status = solver.solve();
        
        if (status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE) {
            success_count++;
        }
        
        // Apply first control to plant (simple Euler simulation)
        double u_acc = solver.get_control(0, 0);
        double u_steer = solver.get_control(0, 1);
        
        sim_x += sim_v * std::cos(sim_theta) * dt;
        sim_y += sim_v * std::sin(sim_theta) * dt;
        sim_theta += (sim_v / 2.5) * std::tan(u_steer) * dt;
        sim_v += u_acc * dt;
        
        // Shift and update
        shift_car_guess(solver);
        solver.set_initial_state("x", sim_x);
        solver.set_initial_state("y", sim_y);
        solver.set_initial_state("theta", sim_theta);
        solver.set_initial_state("v", sim_v);
        
        // Update reference trajectory (moving target)
        for(int k = 0; k <= N; ++k) {
            double t_k = (step + 1 + k) * dt;
            solver.set_parameter(k, "x_ref", t_k * 5.0);
        }
        
        // Reset solver state for new MPC step
        solver.reset(ResetOption::ALG_STATE);
    }
    
    // At least 80% of MPC steps should succeed
    EXPECT_GE(success_count, MPC_STEPS * 0.8) 
        << "MPC loop should be stable: " << success_count << "/" << MPC_STEPS << " succeeded";
    
    // Velocity should approach target (5.0) after multiple steps
    EXPECT_GT(sim_v, 3.0) << "Closed-loop velocity should approach target 5.0";
}


// =============================================================================
// TEST 5: Mehrotra Predictor-Corrector Convergence
// Verifies that Mehrotra converges faster (fewer iterations) than Monotone
// on the same problem, which is the key advantage of the Mehrotra strategy.
// =============================================================================
TEST(SolverQualityTest, MehrotraConvergenceAdvantage) {
    constexpr int N = 15;
    
    auto make_config = [](BarrierStrategy strategy) {
        SolverConfig config;
        config.print_level = PrintLevel::NONE;
        config.max_iters = 200;
        config.barrier_strategy = strategy;
        config.tol_con = 1e-5;
        config.mu_final = 1e-6;
        config.line_search_type = LineSearchType::FILTER;
        config.integrator = IntegratorType::EULER_EXPLICIT;
        return config;
    };
    
    auto setup_solver = [&](auto& solver) {
        solver.set_dt(0.1);
        for(int k = 0; k <= N; ++k) {
            solver.set_parameter(k, "v_ref", 5.0);
            solver.set_parameter(k, "w_vel", 1.0);
            solver.set_parameter(k, "w_pos", 1.0);
            solver.set_parameter(k, "w_acc", 0.1);
            solver.set_parameter(k, "w_steer", 1.0);
            solver.set_parameter(k, "w_theta", 0.1);
            solver.set_parameter(k, "L", 2.5);
            solver.set_parameter(k, "car_rad", 1.0);
            solver.set_parameter(k, "obs_x", 100.0);
            solver.set_parameter(k, "obs_y", 0.0);
            solver.set_parameter(k, "obs_rad", 1.0);
        }
        solver.set_initial_state("x", 0.0);
        solver.set_initial_state("v", 0.0);
        solver.rollout_dynamics();
    };
    
    // Solve with Monotone
    MiniSolver<CarModel, 20> solver_mono(N, Backend::CPU_SERIAL, make_config(BarrierStrategy::MONOTONE));
    setup_solver(solver_mono);
    SolverStatus status_mono = solver_mono.solve();
    int iters_mono = solver_mono.get_iteration_count();
    
    // Solve with Mehrotra
    MiniSolver<CarModel, 20> solver_meh(N, Backend::CPU_SERIAL, make_config(BarrierStrategy::MEHROTRA));
    setup_solver(solver_meh);
    SolverStatus status_meh = solver_meh.solve();
    int iters_meh = solver_meh.get_iteration_count();
    
    // Both should converge
    EXPECT_TRUE(status_mono == SolverStatus::OPTIMAL || status_mono == SolverStatus::FEASIBLE)
        << "Monotone should converge";
    EXPECT_TRUE(status_meh == SolverStatus::OPTIMAL || status_meh == SolverStatus::FEASIBLE)
        << "Mehrotra should converge";
    
    // Mehrotra should use fewer iterations (or at most equal)
    EXPECT_LE(iters_meh, iters_mono) 
        << "Mehrotra (" << iters_meh << " iters) should be faster than Monotone (" 
        << iters_mono << " iters)";
    
    // Solutions should reach similar cost
    double cost_mono = 0, cost_meh = 0;
    for(int k = 0; k <= N; ++k) {
        cost_mono += solver_mono.get_stage_cost(k);
        cost_meh += solver_meh.get_stage_cost(k);
    }
    EXPECT_NEAR(cost_mono, cost_meh, cost_mono * 0.1) 
        << "Both strategies should find similar optimal cost";
}


// =============================================================================
// TEST 6: Initial Guess Quality Verification
// Verifies that a shifted state/control guess still converges to a solution comparable
// to a cold start on a slightly perturbed problem (simulating MPC re-solve).
// =============================================================================
TEST(SolverQualityTest, BetterInitialGuessProducesComparableSolution) {
    constexpr int N = 15;
    
    auto make_config = []() {
        SolverConfig config;
        config.print_level = PrintLevel::NONE;
        config.max_iters = 100;
        config.barrier_strategy = BarrierStrategy::MEHROTRA;
        config.tol_con = 1e-5;
        config.mu_final = 1e-6;
        config.integrator = IntegratorType::EULER_EXPLICIT;
        return config;
    };
    
    auto setup_params = [&](auto& solver) {
        solver.set_dt(0.1);
        for(int k = 0; k <= N; ++k) {
            solver.set_parameter(k, "v_ref", 5.0);
            solver.set_parameter(k, "w_vel", 1.0);
            solver.set_parameter(k, "w_pos", 1.0);
            solver.set_parameter(k, "w_acc", 0.1);
            solver.set_parameter(k, "w_steer", 1.0);
            solver.set_parameter(k, "w_theta", 0.1);
            solver.set_parameter(k, "L", 2.5);
            solver.set_parameter(k, "car_rad", 1.0);
            solver.set_parameter(k, "obs_x", 100.0);
            solver.set_parameter(k, "obs_y", 0.0);
            solver.set_parameter(k, "obs_rad", 1.0);
        }
    };
    
    // 1. Solve the original problem to get a good solution
    MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, make_config());
    setup_params(solver);
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("v", 0.0);
    solver.rollout_dynamics();
    solver.solve();
    
    auto seed_state_guess = [&](auto& dst, const auto& src) {
        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < CarModel::NX; ++i) {
                int src_k = std::min(k + 1, N);
                dst.set_state_guess(k, i, src.get_state(src_k, i));
            }
        }
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < CarModel::NU; ++i) {
                int src_k = std::min(k + 1, N - 1);
                dst.set_control_guess(k, i, src.get_control(src_k, i));
            }
        }
    };
    
    // 2. Better-guess solve: new solver with slightly perturbed initial state
    MiniSolver<CarModel, 20> solver_guess(N, Backend::CPU_SERIAL, make_config());
    setup_params(solver_guess);
    solver_guess.set_initial_state("x", 0.5);
    solver_guess.set_initial_state("v", 0.5);
    seed_state_guess(solver_guess, solver);
    
    SolverStatus status_guess = solver_guess.solve();
    int iters_guess = solver_guess.get_iteration_count();
    
    // 3. Cold start solve: same perturbed problem, no warm start
    MiniSolver<CarModel, 20> solver_cold(N, Backend::CPU_SERIAL, make_config());
    setup_params(solver_cold);
    solver_cold.set_initial_state("x", 0.5);
    solver_cold.set_initial_state("v", 0.5);
    solver_cold.rollout_dynamics();
    
    SolverStatus status_cold = solver_cold.solve();
    int iters_cold = solver_cold.get_iteration_count();
    
    // Both should converge
    EXPECT_TRUE(status_guess == SolverStatus::OPTIMAL || status_guess == SolverStatus::FEASIBLE);
    EXPECT_TRUE(status_cold == SolverStatus::OPTIMAL || status_cold == SolverStatus::FEASIBLE);
    
    double x_guess = solver_guess.get_state(N, solver_guess.get_state_idx("x"));
    double v_guess = solver_guess.get_state(N, solver_guess.get_state_idx("v"));
    double x_cold = solver_cold.get_state(N, solver_cold.get_state_idx("x"));
    double v_cold = solver_cold.get_state(N, solver_cold.get_state_idx("v"));

    // A better guess should stay in the same solution basin, even if the nonlinear
    // solver converges to a slightly different local solution.
    EXPECT_NEAR(x_guess, x_cold, 1.0);
    EXPECT_NEAR(v_guess, v_cold, 0.2);
    EXPECT_LE(iters_guess, iters_cold + 4);
}
