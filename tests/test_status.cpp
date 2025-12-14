#include <gtest/gtest.h>
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/solver/solver.h"
#include <array>

using namespace minisolver;

// Model: x in R^1
// Cost: x^2
// Con: x >= 2 AND x <= 1 (Impossible)
struct InfeasibleModel {
    static const int NX=1;
    static const int NU=1; // Dummy
    static const int NC=2;
    static const int NP=0;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {0.0, 0.0};
    static constexpr std::array<int, NC> constraint_types = {0, 0}; // Hard

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
    static void compute(KnotPointV2<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double /*dt*/) {
        // Dynamics: x' = x (Static)
        kp.f_resid(0) = kp.x(0);
        kp.A(0,0) = 1.0;
        kp.B(0,0) = 0.0;
        
        // Cost: x^2
        kp.cost = kp.x(0)*kp.x(0);
        kp.Q(0,0) = 2.0;
        kp.q(0) = 2.0 * kp.x(0);
        
        // Constraints
        // 1. x >= 2  => 2 - x <= 0  => g0 = 2 - x
        // 2. x <= 1  => x - 1 <= 0  => g1 = x - 1
        kp.g_val(0) = 2.0 - kp.x(0);
        kp.g_val(1) = kp.x(0) - 1.0;
        
        // C = dg/dx
        kp.C(0,0) = -1.0;
        kp.C(1,0) = 1.0;
        
        kp.D.setZero();
    }
    
    // Explicit Exact/GN mapping
    template<typename T>
    static void compute_cost_gn(KnotPointV2<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T>
    static void compute_cost_exact(KnotPointV2<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T>
    static void compute_dynamics(KnotPointV2<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) { compute(kp, type, dt); }
    template<typename T>
    static void compute_constraints(KnotPointV2<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
};

TEST(StatusTest, InfeasibilityDetection) {
    int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.max_iters = 50;
    config.enable_feasibility_restoration = true; // Give it a chance to try restoration
    
    MiniSolver<InfeasibleModel, 10> solver(N, Backend::CPU_SERIAL, config);
    
    // Initial State: x=1.5 (Violates both slightly? No, 1.5 >= 1 (ok for g1?), 1.5 <= 2 (ok for g0? wait)
    // g0 = 2 - 1.5 = 0.5 > 0 (Violated)
    // g1 = 1.5 - 1 = 0.5 > 0 (Violated)
    // Wait, x=1.5 violates x>=2 (needs x>=2) and x<=1 (needs x<=1).
    // Yes, 1.5 is in (1, 2), so it violates both constraints.
    solver.set_initial_state("x", 1.5); 
    
    SolverStatus status = solver.solve();
    
    // Should detect infeasibility
    // It might return PRIMAL_INFEASIBLE (mapped to INFEASIBLE now)
    // Or MAX_ITER if it cycles (but we want it to detect infeasibility if restoration fails)
    
    EXPECT_EQ(status, SolverStatus::INFEASIBLE);
}

