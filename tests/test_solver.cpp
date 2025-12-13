#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include <vector>

using namespace minisolver;

// Integration test: Run the solver on the CarModel and check if it converges
TEST(SolverTest, FullConvergence) {
    int N = 20; // Shorter horizon for quick test
    
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG; // Enable debug logs
    config.tol_con = 1e-3;
    config.max_iters = 50;
    
    // Robust Config
    // Use Monotone strategy as Adaptive might be unstable for cold start without tuning
    config.barrier_strategy = BarrierStrategy::MONOTONE; 
    config.line_search_type = LineSearchType::FILTER;
    
    MiniSolver<CarModel, 50> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    
    // Set Target
    for(int k=0; k<=N; ++k) {
        solver.set_parameter(k, "v_ref", 5.0);
        solver.set_parameter(k, "w_vel", 1.0);
        // Put obstacle far away to avoid complexity in this basic test
        solver.set_parameter(k, "obs_x", 100.0);
        solver.set_parameter(k, "obs_rad", 1.0);
        // [FIX] Set physical parameters to avoid div-by-zero (L=0)
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
    }
    
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("v", 0.0);
    
    solver.rollout_dynamics();
    
    SolverStatus status = solver.solve();
    
    EXPECT_EQ(status, SolverStatus::OPTIMAL);
    
    // Check if velocity reached target
    // N*dt = 2.0s. Acc limit is 3.0. 
    // v = a*t = 3*2 = 6 > 5. So it should reach 5.0.
    double v_final = solver.get_state(N, 3); // index 3 is v
    EXPECT_NEAR(v_final, 5.0, 0.1);
}

TEST(SolverTest, InfeasibleStartRecovery) {
    // Start with a state that violates constraints immediately?
    // Or set initial guess that is very bad.
    int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::ITER;
    config.enable_feasibility_restoration = true;
    config.max_iters = 500;
    
    // Robust Config
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::FILTER;

    MiniSolver<CarModel, 50> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    
    // Initialize with random/bad values
    for(int k=0; k<=N; ++k) {
        solver.set_parameter(k, "v_ref", 5.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_steer", 0.1); // Add steer weight
        solver.set_parameter(k, "w_acc", 0.1);   // Add acc weight
        solver.set_parameter(k, "obs_x", 100.0);
        solver.set_parameter(k, "obs_rad", 1.0);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        solver.set_state_guess(k, "v", 20.0); // Bad guess (20.0 >> 0.0), but recoverable
    }
    
    // Solve
    SolverStatus status = solver.solve();
    // It should handle the bad guess and converge
    EXPECT_EQ(status, SolverStatus::OPTIMAL);
}

