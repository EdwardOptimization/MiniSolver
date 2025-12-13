#include <gtest/gtest.h>
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/solver/solver.h"
#include <array>

using namespace minisolver;

// Model: Flat Cost, Feasible
// Min 0.000001 * x^2
// s.t. x >= 0
// Start at x=0. Cost=0. Stays at 0.
// Or start at x=1. Cost=1e-6.
// Cost reduction is very small.
struct FlatCostModel {
    static const int NX=1;
    static const int NU=1; // Dummy
    static const int NC=1;
    static const int NP=0;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {0.0};
    static constexpr std::array<int, NC> constraint_types = {0}; // Hard

    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& /*u*/,
        const MSVec<T, NP>& /*p*/,
        double /*dt*/,
        IntegratorType /*type*/)
    {
        return x; // Static
    }

    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double /*dt*/) {
        kp.f_resid(0) = kp.x(0);
        kp.A(0,0) = 1.0;
        kp.B(0,0) = 0.0;
        
        // Very small cost
        double w = 1e-7;
        kp.cost = w * kp.x(0) * kp.x(0);
        kp.Q(0,0) = 2*w;
        kp.q(0) = 2*w * kp.x(0);
        kp.R.setIdentity(); // Regularize u
        
        // Feasible Constraint x >= -10
        kp.g_val(0) = -10.0 - kp.x(0); // -10 - x <= 0 -> x >= -10
        kp.C(0,0) = -1.0;
    }
    
    template<typename T>
    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) { compute(kp, type, dt); }
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
};

TEST(FeaturesTest, CostStagnationTermination) {
    int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    
    // Set tight tolerances so it doesn't converge by KKT easily
    config.tol_grad = 1e-12;
    config.tol_con = 1e-12;
    config.tol_mu = 1e-12;
    config.mu_min = 1e-9;
    
    // Set Cost Tolerance larger than actual change
    config.tol_cost = 1e-5; 
    
    config.max_iters = 50; 
    
    MiniSolver<FlatCostModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0); // Cost ~ 1e-7
    
    SolverStatus status = solver.solve();
    
    // It should terminate due to Cost Stagnation, but return OPTIMAL or FEASIBLE if valid.
    // The "Stagnation" check returns SOLVED/OPTIMAL if feasible.
    EXPECT_EQ(status, SolverStatus::OPTIMAL);
    
    // Check it didn't run full iterations
    EXPECT_LT(solver.current_iter, config.max_iters);
    
    // Verify log message manually if needed, but status check implies it stopped early.
}

