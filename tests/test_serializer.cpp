#include <gtest/gtest.h>
#include "minisolver/core/serializer.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include <cstdio> // for remove()
#include <fstream>
#include <iterator>
#include <vector>

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
        for (int i = 0; i < CarModel::NX; ++i) solver.set_state_guess(k, i, k * 1.0);
        if (k < N) {
            for (int i = 0; i < CarModel::NU; ++i) solver.set_control_guess(k, i, k * 0.5);
        }
        for (int i = 0; i < CarModel::NC; ++i) {
            solver.set_slack_guess(k, i, 0.1);
            solver.set_dual_guess(k, i, 0.2);
        }
    }

    // 1. Capture State (Test New Interface)
    auto snapshot = Serializer::capture_state(solver, SolverStatus::FEASIBLE);
    snapshot.iterations = 7;
    snapshot.mu = 0.3;
    snapshot.reg = 1.2;
    
    EXPECT_EQ(snapshot.N, N);
    EXPECT_EQ(snapshot.config.mu_init, 0.5);
    EXPECT_EQ(snapshot.status, SolverStatus::FEASIBLE);
    EXPECT_EQ(snapshot.iterations, 7);
    EXPECT_DOUBLE_EQ(snapshot.mu, 0.3);
    EXPECT_DOUBLE_EQ(snapshot.reg, 1.2);
    EXPECT_DOUBLE_EQ(snapshot.total_cost, 0.0);

    // 2. Save to Disk
    std::string filename = "test_snapshot.dat";
    bool save_ok = Serializer::save_state(filename, snapshot);
    EXPECT_TRUE(save_ok);

    // 3. Load into new solver
    MySolver solver2(10, Backend::CPU_SERIAL); // Initialize with different N
    bool load_ok = Serializer::load_case(filename, solver2);
    EXPECT_TRUE(load_ok);

    // 4. Verify Loaded Data
    EXPECT_EQ(solver2.get_horizon(), N);
    EXPECT_EQ(solver2.get_config().mu_init, 0.5);
    EXPECT_EQ(solver2.get_config().barrier_strategy, BarrierStrategy::MEHROTRA);
    auto loaded_snapshot = Serializer::capture_state(solver2);
    EXPECT_EQ(loaded_snapshot.iterations, 7);
    EXPECT_DOUBLE_EQ(loaded_snapshot.mu, 0.3);
    EXPECT_DOUBLE_EQ(loaded_snapshot.reg, 1.2);
    
    // Verify Trajectory
    for(int k=0; k<=N; ++k) {
        // X
        for(int i=0; i<CarModel::NX; ++i) {
            EXPECT_DOUBLE_EQ(solver2.get_state(k, i), k * 1.0);
        }
        // U
        if (k < N) {
            for(int i=0; i<CarModel::NU; ++i) {
                EXPECT_DOUBLE_EQ(solver2.get_control(k, i), k * 0.5);
            }
        }
        // P
        EXPECT_EQ(solver2.get_parameter(k, solver2.get_param_idx("v_ref")), 5.0);
        EXPECT_EQ(solver2.get_parameter(k, solver2.get_param_idx("L")), 2.5);

        // Slacks/Duals
        for(int i=0; i<CarModel::NC; ++i) {
            EXPECT_DOUBLE_EQ(solver2.get_slack(k, i), 0.1);
            EXPECT_DOUBLE_EQ(solver2.get_dual(k, i), 0.2);
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
    SolverConfig cfgA = solverA.get_config();
    cfgA.max_iters = 1;
    solverA.set_config(cfgA);
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
    EXPECT_EQ(solverA.get_horizon(), solverB.get_horizon());
    EXPECT_EQ(solverA.get_iteration_count(), solverB.get_iteration_count());

    auto stateA = minisolver::SolverSerializer<CarModel, 10>::capture_state(solverA);
    auto stateB = minisolver::SolverSerializer<CarModel, 10>::capture_state(solverB);
    EXPECT_DOUBLE_EQ(stateA.mu, stateB.mu);
    EXPECT_DOUBLE_EQ(stateA.reg, stateB.reg);
    EXPECT_DOUBLE_EQ(stateA.total_cost, stateB.total_cost);
    
    for(int k=0; k<=N; ++k) {
        for(int i=0; i<CarModel::NX; ++i) EXPECT_EQ(solverA.get_state(k, i), solverB.get_state(k, i));
        if (k < N) {
            for(int i=0; i<CarModel::NU; ++i) EXPECT_EQ(solverA.get_control(k, i), solverB.get_control(k, i));
        }
        for(int i=0; i<CarModel::NC; ++i) {
            EXPECT_EQ(solverA.get_slack(k, i), solverB.get_slack(k, i));
            EXPECT_EQ(solverA.get_dual(k, i), solverB.get_dual(k, i));
        }
        for(int i=0; i<CarModel::NP; ++i) EXPECT_EQ(solverA.get_parameter(k, i), solverB.get_parameter(k, i));
    }
}

TEST(SerializerTest, TruncatedFileRejected) {
    int N = 3;
    MiniSolver<CarModel, 10> solver(N, Backend::CPU_SERIAL);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 1.0);
    solver.set_parameter(0, "v_ref", 5.0);
    solver.solve();

    std::string filename = "test_truncated.bin";
    ASSERT_TRUE((minisolver::SolverSerializer<CarModel, 10>::save_case(filename, solver)));

    std::ifstream in(filename, std::ios::binary);
    ASSERT_TRUE(in.good());
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    ASSERT_GT(bytes.size(), 8u);
    bytes.resize(bytes.size() - 8);

    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good());
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();

    SolverConfig cfg2;
    cfg2.mu_init = 0.123;
    cfg2.barrier_strategy = BarrierStrategy::MONOTONE;
    MiniSolver<CarModel, 10> solver2(1, Backend::CPU_SERIAL, cfg2);
    EXPECT_EQ(solver2.get_horizon(), 1);
    EXPECT_DOUBLE_EQ(solver2.get_config().mu_init, 0.123);

    EXPECT_FALSE((minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solver2)));

    // Load should be atomic: on failure, the solver state should remain unchanged.
    EXPECT_EQ(solver2.get_horizon(), 1);
    EXPECT_DOUBLE_EQ(solver2.get_config().mu_init, 0.123);

    std::remove(filename.c_str());
}

TEST(SerializerTest, OversizeHorizonRejected) {
    // Save with MAX_N=50 and N=20, then attempt to load with MAX_N=10.
    // The loader should refuse to truncate.
    int N = 20;
    MiniSolver<CarModel, 50> solver_big(N, Backend::CPU_SERIAL);
    solver_big.set_dt(0.1);

    std::string filename = "test_oversize.bin";
    ASSERT_TRUE((minisolver::SolverSerializer<CarModel, 50>::save_case(filename, solver_big)));

    MiniSolver<CarModel, 10> solver_small(5, Backend::CPU_SERIAL);
    EXPECT_FALSE((minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solver_small)));

    std::remove(filename.c_str());
}
