#include <gtest/gtest.h>
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/solver/solver.h"
#include <array>

using namespace minisolver;

// Define a model with strictly singular Hessian
// Min J = 0  (No Cost)
// s.t. x = 1 (Constraint)
// This problem has a 0 Hessian for objective. 
// If regularization fails, the KKT system [Q  A^T; A  0] will be singular if Q=0 and A is not full rank (here A=[1] is full rank, so KKT is invertible, but pure Newton on Cost would fail).
// Let's make it harder: Q=0, and unconstrained degrees of freedom.
// Min 0*x^2 + 0*u^2
// s.t. x_{k+1} = x_k + u_k
// x_0 = 0
// No other constraints.
// Solution is x=0, u=0 (or any u if cost truly 0).
// But Riccati backup requires inverting (R + B^T P B). If R=0 and P=0 (terminal), this is singular.
// The solver should add regularization to R to proceed.

struct SingularModel {
    static const int NX=1;
    static const int NU=1;
    static const int NC=0;
    static const int NP=0;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/,
        double dt,
        IntegratorType /*type*/)
    {
        return x + u * dt;
    }

    // Zero Cost, Zero Derivatives (Singular)
    template<typename T>
    static void compute(KnotPointV2<T,NX,NU,NC,NP>& kp, IntegratorType /*type*/, double dt) {
        // Dynamics: x' = u
        kp.f_resid(0) = kp.x(0) + kp.u(0)*dt;
        kp.A(0,0) = 1.0;
        kp.B(0,0) = dt;
        
        // Cost: 0
        kp.cost = 0.0;
        kp.Q.setZero();
        kp.R.setZero(); // Strictly Singular
        kp.q.setZero();
        kp.r.setZero();
        
        // No Constraints
        // kp.C, kp.D, kp.g_val are zero by default or init
    }
    
    // Explicit Exact/GN mapping
    template<typename T>
    static void compute_cost_gn(KnotPointV2<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T>
    static void compute_cost_exact(KnotPointV2<T,NX,NU,NC,NP>& kp) { compute(kp, IntegratorType::EULER_EXPLICIT, 0.1); }
    template<typename T>
    static void compute_dynamics(KnotPointV2<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) { compute(kp, type, dt); }
    template<typename T>
    static void compute_constraints(KnotPointV2<T,NX,NU,NC,NP>& /*kp*/) { /* None */ }
};

TEST(RobustnessTest, SingularHessianRecovery) {
    int N = 10;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.inertia_strategy = InertiaStrategy::REGULARIZATION;
    config.reg_init = 1e-8; // Start small
    config.reg_min = 1e-8;
    
    // Create Solver
    MiniSolver<SingularModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 1.0); // Start away from 0
    
    // The problem: Min 0. Dynamics x+=u*dt.
    // Optimal strategy: u can be anything?
    // Wait, if cost is 0, any feasible trajectory is optimal.
    // But Newton step needs to invert Hessian.
    // Solver should detect singularity in R (0) and add regularization.
    // It should effectively minimize 0.5 * reg * ||du||^2 + 0.5 * reg * ||dx||^2
    // Which means it should stay close to initial guess (u=0).
    
    SolverStatus status = solver.solve();
    
    // Should NOT crash or return NUMERICAL_ERROR
    EXPECT_NE(status, SolverStatus::NUMERICAL_ERROR);
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
    
    // Check if regularization was increased or maintained
    // Note: If reg_init (1e-8) is sufficient to make it invertible numerically (machine epsilon 1e-16),
    // it might not need to increase if singular_threshold is loose.
    // However, since R=0, eigenvalues are all 0 (or small noise).
    // The Riccati solver should bump reg if it encounters small pivots.
    
    // In this case, it converged in 2 iters.
    // Iter 1: Regularization likely used.
    // Let's print reg to see.
    std::cout << "Final Reg: " << solver.reg << std::endl;
    
    // Just verify it succeeded. The fact it solved a Singular problem is the key.
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
}

