#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"

using namespace minisolver;

// Mock LinearSolver that fails first step but returns success on SOC step
class SocMockLinearSolver : public LinearSolver<Trajectory<KnotPoint<double, 4, 2, 5, 13>, 5>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 4, 2, 5, 13>, 5>::TrajArray;
    int solve_count = 0;
    
    bool solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/, InertiaStrategy /*strategy*/, 
              const SolverConfig& /*config*/, const TrajArray* /*affine_traj*/ = nullptr) override {
        solve_count++;
        // Standard solve: produce a step that gets rejected (e.g. too aggressive)
        // dx = -10.0 (if x=10, goes to 0)
        // But let's pretend this step causes constraint violation or cost increase
        for(int k=0; k<=N; ++k) {
            traj[k].dx.fill(-10.0);
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
        }
        return true;
    }
    
    bool solve_soc(TrajArray& traj, const TrajArray& /*soc_rhs_traj*/, int N, double /*mu*/, double /*reg*/, InertiaStrategy /*strategy*/,
                   const SolverConfig& /*config*/) override {
        solve_count++;
        // SOC solve: produce a correction step
        // dx = 1.0 (corrects back slightly)
        for(int k=0; k<=N; ++k) {
            traj[k].dx.fill(1.0);
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
        }
        return true;
    }
};

// We need a custom LineSearch that we can inspect, or just use FilterLineSearch
// and verify SOC was triggered by checking solve count?
// But FilterLineSearch owns the linear solver? No, it takes reference.

TEST(AdvancedFeaturesTest, SOCLogic) {
    constexpr int N = 5;
    using Model = CarModel; // Reuse CarModel dimensions
    
    SolverConfig config;
    config.line_search_type = LineSearchType::FILTER;
    config.enable_soc = true;
    config.soc_trigger_alpha = 0.5; // Trigger if alpha > 0.5 step failed? 
    // Wait, logic is: if !accepted && alpha > trigger.
    // We want first step (alpha=1.0) to fail.
    
    SocMockLinearSolver linear_solver;
    FilterLineSearch<Model, N> ls;
    
    Trajectory<KnotPoint<double, 4, 2, 5, 13>, N> trajectory(N);
    std::array<double, N> dts; dts.fill(0.1);
    
    // Setup initial state
    auto& active = trajectory.active();
    for(int k=0; k<=N; ++k) {
        active[k].set_zero();
        active[k].x.fill(10.0);
        active[k].cost = 0.0;
        active[k].g_val.fill(-1.0); // Feasible
    }
    
    // First solve (outside LineSearch)
    linear_solver.solve(active, N, 0.1, 1e-6, InertiaStrategy::REGULARIZATION, config);
    // dx = -10. Candidate x = 0.
    
    // We need to make Candidate (x=0) UNACCEPTABLE.
    // Filter accepts if phi decreases or theta decreases.
    // Active: phi=0, theta=0.
    // Candidate (x=0): we need to make cost/viol high.
    // But `Model::compute` is called on candidate.
    // CarModel with 0 parameters -> cost 0.
    // So candidate will have cost 0.
    // Filter accepts equality.
    
    // We need to inject logic to make candidate fail?
    // We can't easily mock Model::compute inside template.
    // But we can set up CarModel parameters such that x=0 has HIGH cost.
    // E.g. x_ref = 10. w_pos = 100.
    // Active x=10 -> Cost 0.
    // Candidate x=0 -> Cost 10000.
    // So Phi increases.
    
    for(int k=0; k<=N; ++k) {
        active[k].p(1) = 10.0; // x_ref
        active[k].p(8) = 100.0; // w_pos
    }
    // Recompute active metrics (cost=0)
    // Actually search computes m_0.
    
    // Run search
    double alpha = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    (void)alpha; // Unused variable
    
    // Expectations:
    // 1. First step (alpha=1, dx=-10 -> x=0) rejected because cost increases (0 -> 10000).
    // 2. SOC triggered (alpha > 0.5).
    // 3. solve_soc called. returns dx=1.
    // 4. New candidate x = 0 + 1 = 1.
    // 5. Cost at x=1: 100 * (1-10)^2 = 8100.
    // 6. Still > 0. Rejected?
    // 7. Backtrack alpha=0.5. dx=-5. x=5. Cost 2500. Rejected.
    // ...
    // Eventually alpha small enough?
    
    // We just want to verify solve_soc was called.
    // solve_count should be 1 (initial) + 1 (soc) + ... wait, search doesn't call solve().
    // search calls solve_soc().
    // So solve_count inside search starts at 1 (from our manual call).
    // If soc triggered, count becomes 2.
    
    EXPECT_GE(linear_solver.solve_count, 2);
}

