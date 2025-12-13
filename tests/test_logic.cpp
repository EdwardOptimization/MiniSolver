#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"

using namespace minisolver;

TEST(LogicTest, ParameterPersistenceCheck) {
    int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.max_iters = 1; // Run 1 iter to force buffer swap
    
    MiniSolver<CarModel, 10> solver(N, Backend::CPU_SERIAL, config);
    
    // 1. Set a Parameter (e.g. x_ref at k=2)
    double magic_val = 123.456;
    solver.set_parameter(2, "x_ref", magic_val);
    
    // Verify initial set
    EXPECT_DOUBLE_EQ(solver.get_parameter(2, "x_ref"), magic_val);
    EXPECT_DOUBLE_EQ(solver.trajectory.active()[2].p(1), magic_val);
    
    // 2. Run Solve
    // This triggers step(), linear solve, line search.
    // Line search creates 'candidate' trajectory.
    // If successful, swap() happens.
    // We need to ensure parameters are carried over during candidate preparation and swap.
    solver.solve();
    
    // 3. Verify persistence in Active
    // If the bug exists (candidate buffer had zero params and overwrote active), this will be 0.
    double val_after = solver.get_parameter(2, "x_ref");
    EXPECT_DOUBLE_EQ(val_after, magic_val) << "Parameter lost after solve() iteration (Ghost Cost Bug)";
    
    // 4. Verify persistence in Candidate (should be synced)
    EXPECT_DOUBLE_EQ(solver.trajectory.candidate()[2].p(1), magic_val) << "Candidate buffer parameter out of sync";
}

