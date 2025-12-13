#pragma once
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/matrix_defs.h"
#include <cmath>
#include <string>
#include <array>

namespace minisolver {

struct LineSearchBenchmarkModel {
    // --- Constants ---
    static const int NX=2;
    static const int NU=2;
    static const int NC=4;
    static const int NP=3;

    static constexpr std::array<double, NC> constraint_weights = {0.0, 0.0, 0.0, 0.0};
    static constexpr std::array<int, NC> constraint_types = {0, 0, 0, 0};


    // --- Name Arrays (for Map Construction) ---
    static constexpr std::array<const char*, NX> state_names = {
        "x",
        "y",
    };

    static constexpr std::array<const char*, NU> control_names = {
        "vx",
        "vy",
    };

    static constexpr std::array<const char*, NP> param_names = {
        "a_param",
        "b_param",
        "w_u",
    };


    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in) 
    {
        (void)x_in;
        T vx = u_in(0);
        T vy = u_in(1);
        (void)p_in;

        MSVec<T, NX> xdot;
        xdot(0) = vx;
        xdot(1) = vy;
        return xdot;

    }

    // --- Integrator Interface ---
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
                // Simple Fixed-Point Iteration for x_next = x + f(x_next, u) * dt
                MSVec<T, NX> x_next = x; // Guess
                for(int i=0; i<5; ++i) {
                    x_next = x + dynamics_continuous(x_next, u, p) * dt;
                }
                return x_next;
            }

            case IntegratorType::RK2_IMPLICIT:
            {
                // Implicit Midpoint: k = f(x + 0.5*dt*k). x_next = x + dt*k
                MSVec<T, NX> k = dynamics_continuous(x, u, p); // Guess k0
                for(int i=0; i<5; ++i) {
                    k = dynamics_continuous<T>(x + k * (0.5 * dt), u, p);
                }
                return x + k * dt;
            }

            // Fallback for others to RK4 or appropriate handling
            default: // RK4 Explicit (Default)
            {
               auto k1 = dynamics_continuous(x, u, p);
               auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u, p);
               auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u, p);
               auto k4 = dynamics_continuous<T>(x + k3 * dt, u, p);
               return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }
        }
    }

    // --- 1. Compute Dynamics (f_resid, A, B) ---
    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        T x = kp.x(0);
        T y = kp.x(1);
        T vx = kp.u(0);
        T vy = kp.u(1);

        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
            case IntegratorType::EULER_IMPLICIT:
            {
                kp.f_resid(0) = dt*vx + x;
                kp.f_resid(1) = dt*vy + y;
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(1,1) = 1;
                kp.B.setZero();
                kp.B(0,0) = dt;
                kp.B(1,1) = dt;
                break;
            }
            case IntegratorType::RK2_EXPLICIT:
            case IntegratorType::RK2_IMPLICIT:
            {
                kp.f_resid(0) = dt*vx + x;
                kp.f_resid(1) = dt*vy + y;
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(1,1) = 1;
                kp.B.setZero();
                kp.B(0,0) = dt;
                kp.B(1,1) = dt;
                break;
            }
            case IntegratorType::RK4_EXPLICIT:
            case IntegratorType::RK4_IMPLICIT:
            {
                T tmp_d0 = 1.0*dt;
                kp.f_resid(0) = tmp_d0*vx + x;
                kp.f_resid(1) = tmp_d0*vy + y;
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(1,1) = 1;
                kp.B.setZero();
                kp.B(0,0) = tmp_d0;
                kp.B(1,1) = tmp_d0;
                break;
            }
        }
    }

    // --- 2. Compute Constraints (g_val, C, D) ---
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T vx = kp.u(0);
        T vy = kp.u(1);

        // --- Special Constraints Pre-Calculation ---


        // g_val
        kp.g_val(0,0) = vx - 5.0;
        kp.g_val(1,0) = -vx - 5.0;
        kp.g_val(2,0) = vy - 5.0;
        kp.g_val(3,0) = -vy - 5.0;

        // C
        kp.C(0,0) = 0;
        kp.C(0,1) = 0;
        kp.C(1,0) = 0;
        kp.C(1,1) = 0;
        kp.C(2,0) = 0;
        kp.C(2,1) = 0;
        kp.C(3,0) = 0;
        kp.C(3,1) = 0;

        // D
        kp.D(0,0) = 1;
        kp.D(0,1) = 0;
        kp.D(1,0) = -1;
        kp.D(1,1) = 0;
        kp.D(2,0) = 0;
        kp.D(2,1) = 1;
        kp.D(3,0) = 0;
        kp.D(3,1) = -1;

    }

    // --- 3. Compute Cost (Implemented via template for Exact/GN) ---
    template<typename T, int Mode>
    static void compute_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x = kp.x(0);
        T y = kp.x(1);
        T vx = kp.u(0);
        T vy = kp.u(1);
        T a_param = kp.p(0);
        T b_param = kp.p(1);
        T w_u = kp.p(2);

        T tmp_j0 = pow(x, 2);
        T tmp_j1 = -tmp_j0 + y;
        T tmp_j2 = 4*b_param;
        T tmp_j3 = tmp_j1*tmp_j2;
        T tmp_j4 = 2*w_u;
        T tmp_j5 = -tmp_j2*x;

        // q
        kp.q(0,0) = -2*a_param - tmp_j3*x + 2*x;
        kp.q(1,0) = b_param*(-2*tmp_j0 + 2*y);

        // r
        kp.r(0,0) = tmp_j4*vx;
        kp.r(1,0) = tmp_j4*vy;

        // Q (Mode 0=GN, 1=Exact)
        kp.Q(0,0) = 8*b_param*tmp_j0 - tmp_j3 + 2;
        kp.Q(0,1) = tmp_j5;
        kp.Q(1,0) = tmp_j5;
        kp.Q(1,1) = 2*b_param;

        // R (Mode 0=GN, 1=Exact)
        kp.R(0,0) = tmp_j4;
        kp.R(0,1) = 0;
        kp.R(1,0) = 0;
        kp.R(1,1) = tmp_j4;

        // H (Mode 0=GN, 1=Exact)
        kp.H(0,0) = 0;
        kp.H(0,1) = 0;
        kp.H(1,0) = 0;
        kp.H(1,1) = 0;

        kp.cost = b_param*pow(tmp_j1, 2) + w_u*(pow(vx, 2) + pow(vy, 2)) + pow(a_param - x, 2);
    }

template<typename T>
    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_cost_impl<T, 0>(kp);
    }

    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_cost_impl<T, 1>(kp);
    }

    template<typename T>
    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_cost_impl<T, 1>(kp);
    }


    // --- 4. Compute All (Convenience) ---
    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp); // Default GN
    }

    template<typename T>
    static void compute_exact(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost_exact(kp); // Exact Hessian
    }

    // --- 5. Sparse Kernels (Generated) ---
    
    // --- 6. Fused Riccati Kernel (Generated) ---
    // Updates kp.Q_bar, R_bar, H_bar, q_bar, r_bar in one go.
    // Uses Vxx, Vx from next step.
    template<typename T>
    static void compute_fused_riccati_step(
        const MSMat<T, NX, NX>& Vxx, 
        const MSVec<T, NX>& Vx,
        KnotPoint<T,NX,NU,NC,NP>& kp) 
    {
        T P_0_0 = Vxx(0,0);
        T P_0_1 = Vxx(0,1);
        T P_1_1 = Vxx(1,1);
        T p_0 = Vx(0);
        T p_1 = Vx(1);
        T A_0_0 = kp.A(0,0);
        T A_1_1 = kp.A(1,1);
        T B_0_0 = kp.B(0,0);
        T B_1_1 = kp.B(1,1);

        // CSE Intermediate Variables
        T tmp_ric0 = A_1_1*P_0_1;
        T tmp_ric1 = B_1_1*P_0_1;

        // Accumulate Results
        kp.Q_bar(0,0) += pow(A_0_0, 2)*P_0_0;
        kp.Q_bar(0,1) += A_0_0*tmp_ric0;
        kp.Q_bar(1,1) += pow(A_1_1, 2)*P_1_1;
        kp.R_bar(0,0) += pow(B_0_0, 2)*P_0_0;
        kp.R_bar(0,1) += B_0_0*tmp_ric1;
        kp.R_bar(1,1) += pow(B_1_1, 2)*P_1_1;
        kp.H_bar(0,0) += A_0_0*B_0_0*P_0_0;
        kp.H_bar(0,1) += B_0_0*tmp_ric0;
        kp.H_bar(1,0) += A_0_0*tmp_ric1;
        kp.H_bar(1,1) += A_1_1*B_1_1*P_1_1;
        kp.q_bar(0,0) += A_0_0*p_0;
        kp.q_bar(1,0) += A_1_1*p_1;
        kp.r_bar(0,0) += B_0_0*p_0;
        kp.r_bar(1,0) += B_1_1*p_1;

        // Fill Lower Triangles (Symmetry)
        kp.Q_bar(1,0) = kp.Q_bar(0,1);
        kp.R_bar(1,0) = kp.R_bar(0,1);

    }
    
};
}
