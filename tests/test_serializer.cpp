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
    // Note: load_case restores mu/reg from file now (updated behavior)
    
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

TEST(SerializerTest, FullRoundTrip) {
    int N = 5;
    SolverConfig config;
    
    // 1. Create and Run Solver A
    MiniSolver<CarModel, 10> solverA(N, Backend::CPU_SERIAL, config);
    solverA.set_dt(0.1);
    solverA.set_initial_state("x", 1.0);
    solverA.set_parameter(0, "v_ref", 5.0);
    
    // Run 1 step to populate internal state (s, lam, etc.)
    solverA.config.max_iters = 1;
    solverA.solve();
    
    // 2. Serialize to File
    std::string filename = "test_roundtrip.bin";
    minisolver::SolverSerializer<CarModel, 10>::save_case(filename, solverA);
    
    // 3. Deserialize to Solver B
    MiniSolver<CarModel, 10> solverB(N, Backend::CPU_SERIAL, config);
    bool load_ok = minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solverB);
    EXPECT_TRUE(load_ok);
    
    // 4. Cleanup
    std::remove(filename.c_str());
    
    // 5. Compare State (Bit-Exact)
    EXPECT_EQ(solverA.N, solverB.N);
    EXPECT_EQ(solverA.mu, solverB.mu);
    EXPECT_EQ(solverA.reg, solverB.reg);
    
    auto& trajA = solverA.trajectory.active();
    auto& trajB = solverB.trajectory.active();
    
    for(int k=0; k<=N; ++k) {
        for(int i=0; i<CarModel::NX; ++i) EXPECT_EQ(trajA[k].x(i), trajB[k].x(i));
        for(int i=0; i<CarModel::NU; ++i) EXPECT_EQ(trajA[k].u(i), trajB[k].u(i));
        for(int i=0; i<CarModel::NC; ++i) {
            EXPECT_EQ(trajA[k].s(i), trajB[k].s(i));
            EXPECT_EQ(trajA[k].lam(i), trajB[k].lam(i));
        }
        for(int i=0; i<CarModel::NP; ++i) EXPECT_EQ(trajA[k].p(i), trajB[k].p(i));
    }
}
