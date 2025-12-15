#include <gtest/gtest.h>
#include "minisolver/algorithms/line_search.h"
#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/algorithms/riccati_solver.h"

using namespace minisolver;

// Mock LinearSolver that always returns success and a fixed step
class MockLinearSolver : public LinearSolver<Trajectory<KnotPointV2<double, 4, 2, 5, 13>, 10>> {
public:
    using TrajType = Trajectory<KnotPointV2<double, 4, 2, 5, 13>, 10>;
    
    bool solve(TrajType& traj, int N, double /*mu*/, double /*reg*/, InertiaStrategy /*strategy*/, 
              const SolverConfig& /*config*/, const TrajType* /*affine_traj*/ = nullptr) override {
        // Mock: set dx = -0.1 * x (descent direction)
        auto* state = traj.get_active_state();
        auto* workspace = traj.get_workspace();
        
        for(int k=0; k<=N; ++k) {
            workspace[k].dx = -0.1 * state[k].x;
            workspace[k].du.setZero();
            workspace[k].ds.setZero();
            workspace[k].dlam.setZero();
            workspace[k].dsoft_s.setZero();
        }
        return true;
    }
    
    bool solve_soc(TrajType& /*traj*/, const TrajType& /*soc_rhs_traj*/, int /*N*/, double /*mu*/, double /*reg*/, 
                   InertiaStrategy /*strategy*/, const SolverConfig& /*config*/) override {
        return false;
    }
    
    bool refine(TrajType& /*traj*/, const TrajType& /*original_system*/, int /*N*/, double /*mu*/, double /*reg*/, 
                const SolverConfig& /*config*/) override {
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
    
    Trajectory<KnotPointV2<double, 4, 2, 5, 13>, N> trajectory(N);
    std::array<double, N> dts;
    dts.fill(0.1);
    
    auto* state = trajectory.get_active_state();
    auto* model = trajectory.get_model_data();
    
    // Set initial state high cost
    for(int k=0; k<=N; ++k) {
        state[k].x.fill(10.0); // Far from zero
        state[k].u.setZero();
        state[k].s.setConstant(1e-3);
        state[k].lam.setConstant(1e-3);
        state[k].soft_s.setConstant(1e-3);
        state[k].p.setZero();
        state[k].g_val.fill(0.0); // Feasible
        state[k].cost = 1000.0;
        
        model[k].Q.setIdentity();
        model[k].R.setIdentity();
        model[k].q.setZero();
        model[k].r.setZero();
        model[k].H.setZero();
        model[k].A.setIdentity();
        model[k].B.setZero();
        model[k].C.setZero();
        model[k].D.setZero();
        model[k].f_resid.setZero();
    }
    
    // [FIX] Must compute search direction first, as LineSearch assumes dx is ready
    linear_solver.solve(trajectory, N, 0.1, 1e-6, InertiaStrategy::REGULARIZATION, config);
    
    // Perform search
    double alpha = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    
    // Since mock linear solver gives descent, it should accept step
    EXPECT_GT(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    
    // Check if state updated
    // New x = 10 - alpha * 0.1 * 10 = 10 * (1 - 0.1 alpha) < 10
    auto* state_after = trajectory.get_active_state();
    EXPECT_LT(state_after[0].x(0), 10.0);
}

