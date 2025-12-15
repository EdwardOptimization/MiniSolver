#include <gtest/gtest.h>
#include "minisolver/solver/solver.h"
#include <array>
#include <iostream>

using namespace minisolver;

// Define SoftModel with mutable constraint configuration
struct SoftModel {
    static const int NX=1;
    static const int NU=1;
    static const int NC=1;
    static const int NP=0;

    // Mutable for testing purposes
    static std::array<double, NC> constraint_weights; 
    static std::array<int, NC> constraint_types;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};

    template<typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/) {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt;
        return x_next;
    }

    template<typename T>
    static void compute_dynamics(
        const StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model,
        IntegratorType /*type*/,
        double dt)
    {
        T x = state.x(0); T u = state.u(0);
        model.f_resid(0) = x + u * dt;
        model.A(0,0) = 1.0;
        model.B(0,0) = dt;
    }

    template<typename T>
    static void compute_constraints(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
        T x = state.x(0);
        // x <= 5 -> x - 5 <= 0
        state.g_val(0) = x - 5.0;
        model.C(0,0) = 1.0;
        model.D(0,0) = 0.0;
    }

    template<typename T>
    static void compute_cost_impl(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
        T x = state.x(0); T u = state.u(0);
        // Cost: (x - 10)^2 + 1e-4 * u^2 (small regularization)
        T diff = x - 10.0;
        state.cost = diff*diff + 1e-4 * u*u;
        
        model.q(0) = 2 * diff;
        model.r(0) = 2e-4 * u;
        
        model.Q(0,0) = 2.0;
        model.R(0,0) = 2e-4;
        model.H.setZero();
    }
    
    template<typename T>
    static void compute_cost_gn(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
        compute_cost_impl(state, model);
    }
    
    template<typename T>
    static void compute_cost_exact(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
        compute_cost_impl(state, model);
    }
    
    template<typename T>
    static void compute(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model,
        IntegratorType type,
        double dt)
    {
        compute_dynamics(state, model, type, dt);
        compute_constraints(state, model);
        compute_cost_impl(state, model);
    }
};

// Define static members
std::array<double, SoftModel::NC> SoftModel::constraint_weights = {0.0};
std::array<int, SoftModel::NC> SoftModel::constraint_types = {0};

TEST(SoftConstraintTest, L1_Convergence) {
    // Setup L1
    SoftModel::constraint_types[0] = 1; // L1
    SoftModel::constraint_weights[0] = 1.0; // w=1
    
    SolverConfig config;
    //config.print_level = PrintLevel::DEBUG;
    config.tol_con = 1e-4;
    config.tol_dual = 1e-4;
    config.max_iters = 50;
    
    // N=1
    MiniSolver<SoftModel, 5> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0); // dt=1
    
    // Initial guess x=0, u=10 (to reach 10)
    solver.set_initial_state("x", 0.0);
    solver.set_control_guess(0, "u", 10.0);
    solver.rollout_dynamics();

    // Theoretical L1 Opt: min (x-10)^2 + 1*max(0, x-5)
    // At x=5: cost 25.
    // For x > 5: cost (x-10)^2 + (x-5). deriv 2(x-10)+1 = 2x-19=0 -> x=9.5. Cost 0.25+4.5 = 4.75.
    // For x <= 5: cost (x-10)^2. min at x=5 (constrained).
    // Global min x=9.5.
    
    solver.solve();
    
    double x_final = solver.get_state(1, 0);
    std::cout << "L1 Final X: " << x_final << std::endl;
    
    
    EXPECT_NEAR(x_final, 9.5, 1.0e-3); 
}

TEST(SoftConstraintTest, L2_Convergence) {
    // Setup L2
    SoftModel::constraint_types[0] = 2; // L2
    SoftModel::constraint_weights[0] = 1.0; // w=1
    
    SolverConfig config;
    config.tol_con = 1e-4;
    config.max_iters = 50;
    
    MiniSolver<SoftModel, 5> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0); 
    // Start at x=10 (Cost min, but Feasibility violation)
    solver.set_initial_state("x", 10.0);
    solver.set_control_guess(0, "u", 0.0);
    solver.rollout_dynamics();
    
    // Theoretical L2 Opt: x = 8.333
    // From x=10, cost increases, but violation decreases.
    // If phi doesn't include penalty, cost increases -> phi increases.
    // theta decreases.
    // Filter accepts if theta decreases enough.
    
    solver.solve();
    
    double x_final = solver.get_state(1, 0);
    std::cout << "L2 Final X (from x=10): " << x_final << std::endl;
    
    EXPECT_NEAR(x_final, 8.333, 1.0e-3);
}

