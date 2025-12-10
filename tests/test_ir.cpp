#include <gtest/gtest.h>
#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

// Simple test to ensure Iterative Refinement runs without crashing and maintains/improves convergence
TEST(IterativeRefinementTest, RunsWithoutCrash) {
    int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.integrator = IntegratorType::RK4_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.inertia_strategy = InertiaStrategy::REGULARIZATION;
    
    // Enable IR
    config.enable_iterative_refinement = true;
    
    // Create Solver
    MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, config);
    
    // Init
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("v", 0.0);
    
    // Params
    for(int k=0; k<=N; ++k) {
        solver.set_parameter(k, "v_ref", 1.0);
        solver.set_parameter(k, "x_ref", k * 0.1); 
        solver.set_parameter(k, "y_ref", 0.0);
        solver.set_parameter(k, "obs_x", 100.0); // Far away
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
    
    EXPECT_EQ(status, SolverStatus::SOLVED);
}

