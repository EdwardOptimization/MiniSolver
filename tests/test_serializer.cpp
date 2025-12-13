#include <gtest/gtest.h>
#include "minisolver/core/serializer.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include <cstdio> // for remove()

using namespace minisolver;

TEST(SerializerTest, CaptureAndSaveAndLoad) {
    int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    // Set some non-default config values to verify they are saved correctly
    config.mu_init = 0.5;
    config.tol_con = 1e-4;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    using MySolver = MiniSolver<CarModel, 50>;
    using Serializer = SolverSerializer<CarModel, 50>;

    MySolver solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    // Set parameters and initial state
    for(int k=0; k<=N; ++k) {
        solver.set_parameter(k, "v_ref", 5.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "obs_x", 100.0);
        solver.set_parameter(k, "obs_rad", 1.0);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        
        // Set some dummy state values manually to verify trajectory save/load
        // (Instead of solving, we just fill data to check exact IO reproduction)
        solver.trajectory.active()[k].x.fill(k * 1.0); // x = [k, k, k...]
        solver.trajectory.active()[k].u.fill(k * 0.5); // u = [k/2, ...]
        solver.trajectory.active()[k].s.fill(0.1);     // slacks
        solver.trajectory.active()[k].lam.fill(0.2);   // duals
        solver.trajectory.active()[k].cost = k * 10.0;
    }
    solver.current_iter = 15;

    // 1. Capture State (Test New Interface)
    auto snapshot = Serializer::capture_state(solver, SolverStatus::FEASIBLE);
    
    EXPECT_EQ(snapshot.N, N);
    EXPECT_EQ(snapshot.config.mu_init, 0.5);
    EXPECT_EQ(snapshot.status, SolverStatus::FEASIBLE);
    EXPECT_EQ(snapshot.iterations, 15);
    // Total cost verification
    double expected_cost = 0;
    for(int k=0; k<=N; ++k) expected_cost += k * 10.0;
    EXPECT_DOUBLE_EQ(snapshot.total_cost, expected_cost);

    // 2. Save to Disk
    std::string filename = "test_snapshot.dat";
    bool save_ok = Serializer::save_state(filename, snapshot);
    EXPECT_TRUE(save_ok);

    // 3. Load into new solver
    MySolver solver2(10, Backend::CPU_SERIAL); // Initialize with different N
    bool load_ok = Serializer::load_case(filename, solver2);
    EXPECT_TRUE(load_ok);

    // 4. Verify Loaded Data
    EXPECT_EQ(solver2.N, N);
    EXPECT_EQ(solver2.config.mu_init, 0.5);
    EXPECT_EQ(solver2.config.barrier_strategy, BarrierStrategy::MEHROTRA);
    EXPECT_EQ(solver2.current_iter, 0); // Note: load_case resets iter to 0 usually? 
                                        // Let's check implementation. 
                                        // SolverSerializer::load_case sets config/mu/reg/dt/traj. 
                                        // It does NOT explicitly restore 'current_iter' to solver object 
                                        // (because solver object usually starts fresh solve).
                                        // The iteration count is just metadata in the file for user info.
    
    // Verify Trajectory
    const auto& traj2 = solver2.trajectory.active();
    for(int k=0; k<=N; ++k) {
        // X
        for(int i=0; i<CarModel::NX; ++i) {
            EXPECT_DOUBLE_EQ(traj2[k].x(i), k * 1.0);
        }
        // U
        for(int i=0; i<CarModel::NU; ++i) {
            EXPECT_DOUBLE_EQ(traj2[k].u(i), k * 0.5);
        }
        // P
        // Check a parameter (v_ref index = 5 in car_model usually, but let's check values)
        // param_names: "obs_x", "obs_y", "obs_rad", "v_ref", "car_rad", "L", "w_vel", "w_steer", "w_acc"
        EXPECT_EQ(solver2.get_parameter(k, solver2.get_param_idx("v_ref")), 5.0);
        EXPECT_EQ(solver2.get_parameter(k, solver2.get_param_idx("L")), 2.5);

        // Slacks/Duals
        for(int i=0; i<CarModel::NC; ++i) {
            EXPECT_DOUBLE_EQ(traj2[k].s(i), 0.1);
            EXPECT_DOUBLE_EQ(traj2[k].lam(i), 0.2);
        }
    }

    // Cleanup
    std::remove(filename.c_str());
}

