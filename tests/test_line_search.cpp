#include <gtest/gtest.h>
#include "minisolver/algorithms/line_search.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/algorithms/riccati_solver.h"

using namespace minisolver;

// Mock LinearSolver that always returns success and a fixed step
class MockLinearSolver : public LinearSolver<Trajectory<KnotPoint<double, 4, 2, 5, 13>, 10>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 4, 2, 5, 13>, 10>::TrajArray;
    
    bool solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/, InertiaStrategy /*strategy*/, 
              const SolverConfig& /*config*/, const TrajArray* /*affine_traj*/ = nullptr) override {
        // Mock: set dx = -0.1 * x (descent direction)
        for(int k=0; k<=N; ++k) {
            traj[k].dx = -0.1 * traj[k].x;
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
        }
        return true;
    }
};

TEST(LineSearchTest, FilterAcceptance) {
    SolverConfig config;
    config.line_search_type = LineSearchType::FILTER;
    
    // N=10
    constexpr int N = 10;
    using Model = CarModel;
    using Strategy = FilterLineSearch<Model, N>;
    
    Strategy ls;
    MockLinearSolver linear_solver;
    
    Trajectory<KnotPoint<double, 4, 2, 5, 13>, N> trajectory(N);
    std::array<double, N> dts;
    dts.fill(0.1);
    
    // Set initial state high cost
    for(int k=0; k<=N; ++k) {
        trajectory.active()[k].set_zero();
        trajectory.active()[k].x.fill(10.0); // Far from zero
        trajectory.active()[k].cost = 1000.0;
        trajectory.active()[k].g_val.fill(0.0); // Feasible
    }
    
    // [FIX] Must compute search direction first, as LineSearch assumes dx is ready
    linear_solver.solve(trajectory.active(), N, 0.1, 1e-6, InertiaStrategy::REGULARIZATION, config);
    
    // Perform search
    double alpha = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    
    // Since mock linear solver gives descent, it should accept step
    EXPECT_GT(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    
    // Check if state updated
    // New x = 10 - alpha * 0.1 * 10 = 10 * (1 - 0.1 alpha) < 10
    EXPECT_LT(trajectory.active()[0].x(0), 10.0);
}

