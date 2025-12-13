#include <gtest/gtest.h>
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/matrix_defs.h"
#include <cmath>
#include <array>

using namespace minisolver;

// Define a simple nonlinear model: dx/dt = -x^2
// Exact solution: x(t) = 1 / (1/x0 + t)
struct NonlinearDecayModel {
    static const int NX=1;
    static const int NU=1; // Dummy
    static const int NC=0;
    static const int NP=0;

    static constexpr std::array<const char*, NX> state_names = {"x"};
    static constexpr std::array<const char*, NU> control_names = {"u"};
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& /*u_in*/,
        const MSVec<T, NP>& /*p_in*/) 
    {
        T x = x_in(0);
        MSVec<T, NX> xdot;
        xdot(0) = -x * x;
        return xdot;
    }

    // Standard Integrator Implementation (copied from generated code pattern)
    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& u,
        const MSVec<T, NP>& p,
        double dt,
        IntegratorType type)
    {
        switch(type) {
            case IntegratorType::EULER_EXPLICIT: 
                return x + dynamics_continuous(x, u, p) * dt;
                
            case IntegratorType::RK2_EXPLICIT: 
            {
               auto k1 = dynamics_continuous(x, u, p);
               auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u, p);
               return x + k2 * dt;
            }

            case IntegratorType::EULER_IMPLICIT:
            {
                // Simple Fixed-Point Iteration
                MSVec<T, NX> x_next = x; // Guess
                for(int i=0; i<10; ++i) {
                    x_next = x + dynamics_continuous(x_next, u, p) * dt;
                }
                return x_next;
            }

            case IntegratorType::RK2_IMPLICIT:
            {
                // Implicit Midpoint
                MSVec<T, NX> k = dynamics_continuous(x, u, p); // Guess k0
                for(int i=0; i<10; ++i) {
                    k = dynamics_continuous<T>(x + k * (0.5 * dt), u, p);
                }
                return x + k * dt;
            }

            case IntegratorType::RK4_EXPLICIT:
            default:
            {
               auto k1 = dynamics_continuous(x, u, p);
               auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u, p);
               auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u, p);
               auto k4 = dynamics_continuous<T>(x + k3 * dt, u, p);
               return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }
            
            case IntegratorType::RK4_IMPLICIT:
            {
                // Gauss-Legendre RK4 (Implicit) is complex to implement generically without Butcher tableau.
                // The generated code usually maps RK4_IMPLICIT to RK4_EXPLICIT or a specific implicit scheme.
                // For this test, we assume it falls back to explicit or uses a simple implementation if available.
                // In CarModel it was mapped to same block as Explicit.
                // Let's copy explicit logic here as placeholder if that's what generator does.
                auto k1 = dynamics_continuous(x, u, p);
               auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u, p);
               auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u, p);
               auto k4 = dynamics_continuous<T>(x + k3 * dt, u, p);
               return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }
        }
    }
};

TEST(IntegratorTest, AccuracyComparison) {
    double dt = 0.1;
    double x0_val = 1.0;
    double t_end = 1.0;
    int steps = static_cast<int>(t_end / dt);
    
    MSVec<double, 1> u; u.setZero();
    MSVec<double, 0> p;
    
    // 1. Exact Solution
    double x_exact = 1.0 / (1.0/x0_val + t_end); // 1 / (1 + 1) = 0.5
    
    // 2. Euler Explicit
    MSVec<double, 1> x_ee; x_ee(0) = x0_val;
    for(int k=0; k<steps; ++k) x_ee = NonlinearDecayModel::integrate(x_ee, u, p, dt, IntegratorType::EULER_EXPLICIT);
    double err_ee = std::abs(x_ee(0) - x_exact);
    
    // 3. RK4 Explicit
    MSVec<double, 1> x_rk4; x_rk4(0) = x0_val;
    for(int k=0; k<steps; ++k) x_rk4 = NonlinearDecayModel::integrate(x_rk4, u, p, dt, IntegratorType::RK4_EXPLICIT);
    double err_rk4 = std::abs(x_rk4(0) - x_exact);
    
    // 4. Euler Implicit
    MSVec<double, 1> x_ei; x_ei(0) = x0_val;
    for(int k=0; k<steps; ++k) x_ei = NonlinearDecayModel::integrate(x_ei, u, p, dt, IntegratorType::EULER_IMPLICIT);
    double err_ei = std::abs(x_ei(0) - x_exact);
    
    // Verify Accuracy Hierarchy: RK4 > Euler
    EXPECT_LT(err_rk4, err_ee);
    EXPECT_LT(err_rk4, 1e-5); // RK4 should be very accurate
    
    // Implicit vs Explicit Euler on this stable decaying system
    // Euler explicit: x_{k+1} = x_k - dt*x_k^2
    // Euler implicit: x_{k+1} = x_k - dt*x_{k+1}^2  => dt*x^2 + x - x_k = 0
    // Implicit usually has different error characteristics.
    // For x' = -x^2, Implicit is actually slightly less accurate than explicit in early steps but more stable?
    // Let's just check they are reasonable.
    EXPECT_LT(err_ee, 0.1);
    EXPECT_LT(err_ei, 0.1);
    
    // RK2 Explicit
    MSVec<double, 1> x_rk2; x_rk2(0) = x0_val;
    for(int k=0; k<steps; ++k) x_rk2 = NonlinearDecayModel::integrate(x_rk2, u, p, dt, IntegratorType::RK2_EXPLICIT);
    double err_rk2 = std::abs(x_rk2(0) - x_exact);
    
    EXPECT_LT(err_rk2, err_ee);
    EXPECT_LT(err_rk4, err_rk2);
}

