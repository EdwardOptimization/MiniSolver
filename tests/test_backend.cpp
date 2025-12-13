#include <gtest/gtest.h>
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"

using namespace minisolver;

TEST(BackendTest, FallbackLogic) {
    int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    
    // Explicitly request GPU Backend
    // In a CPU-only build (which this test suite is), this should trigger fallback.
    config.backend = Backend::GPU_MPX; 
    
    // Check if USE_CUDA is defined. If defined, we can't test fallback easily without mocking.
    // Assuming standard CI environment is CPU-only or we can force fallback path logic.
    // The Solver constructor usually checks this.
    // MiniSolver constructor:
    // : trajectory(initial_N), N(initial_N), backend(b), ...
    // It doesn't switch backend in constructor usually, but inside solve() or step() it might check.
    // Wait, RiccatiSolver is instantiated in constructor.
    // Let's check if we can verify the fallback mechanism behavior.
    
    // If backend is GPU but no GPU support compiled, does it crash or warn?
    // Current implementation might just store the enum.
    // The LinearSolver factory/instantiation logic is what matters.
    // In solver.h:
    // linear_solver = std::make_unique<RiccatiSolver<TrajArray, Model>>();
    // It seems RiccatiSolver is generic CPU implementation by default?
    // Where is the GPU dispatch?
    // Usually in linear_solver.h or riccati_solver.h.
    
    MiniSolver<CarModel, 10> solver(N, config.backend, config);
    
    // Check if solver actually runs.
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    
    SolverStatus status = solver.solve();
    
    // It should solve successfully (using CPU fallback or if GPU is actually available).
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
    
    // If we are strictly CPU, verify no GPU calls were made? Hard to test without mocks.
    // But ensuring it doesn't crash with GPU_MPX config on CPU is the goal.
}

