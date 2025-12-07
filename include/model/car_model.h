#pragma once
#include "core/types.h"
#include "core/solver_options.h"
#include "core/matrix_defs.h"
#include <cmath>
#include <string>
#include <array>

namespace minisolver {

struct CarModel {
    // --- Constants ---
    static const int NX=4;
    static const int NU=2;
    static const int NC=5;
    static const int NP=13;

    // --- Name Arrays (for Map Construction) ---
    static constexpr std::array<const char*, NX> state_names = {
        "x",
        "y",
        "theta",
        "v",
    };

    static constexpr std::array<const char*, NU> control_names = {
        "acc",
        "steer",
    };

    static constexpr std::array<const char*, NP> param_names = {
        "v_ref",
        "x_ref",
        "y_ref",
        "obs_x",
        "obs_y",
        "obs_rad",
        "L",
        "car_rad",
        "w_pos",
        "w_vel",
        "w_theta",
        "w_acc",
        "w_steer",
    };


    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in) 
    {
                T x = x_in(0);
        T y = x_in(1);
        T theta = x_in(2);
        T v = x_in(3);
        T acc = u_in(0);
        T steer = u_in(1);
        T v_ref = p_in(0);
        T x_ref = p_in(1);
        T y_ref = p_in(2);
        T obs_x = p_in(3);
        T obs_y = p_in(4);
        T obs_rad = p_in(5);
        T L = p_in(6);
        T car_rad = p_in(7);
        T w_pos = p_in(8);
        T w_vel = p_in(9);
        T w_theta = p_in(10);
        T w_acc = p_in(11);
        T w_steer = p_in(12);

        MSVec<T, NX> xdot;
        xdot(0) = v*cos(theta);
        xdot(1) = v*sin(theta);
        xdot(2) = v*tan(steer)/L;
        xdot(3) = acc;
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
        T theta = kp.x(2);
        T v = kp.x(3);
        T acc = kp.u(0);
        T steer = kp.u(1);
        T v_ref = kp.p(0);
        T x_ref = kp.p(1);
        T y_ref = kp.p(2);
        T obs_x = kp.p(3);
        T obs_y = kp.p(4);
        T obs_rad = kp.p(5);
        T L = kp.p(6);
        T car_rad = kp.p(7);
        T w_pos = kp.p(8);
        T w_vel = kp.p(9);
        T w_theta = kp.p(10);
        T w_acc = kp.p(11);
        T w_steer = kp.p(12);

        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
            case IntegratorType::EULER_IMPLICIT:
            {
                T tmp_d0 = dt*cos(theta);
                T tmp_d1 = tmp_d0*v;
                T tmp_d2 = dt*sin(theta);
                T tmp_d3 = tmp_d2*v;
                T tmp_d4 = tan(steer);
                T tmp_d5 = dt/L;
                T tmp_d6 = tmp_d4*tmp_d5;
                kp.f_resid(0) = tmp_d1 + x;
                kp.f_resid(1) = tmp_d3 + y;
                kp.f_resid(2) = theta + tmp_d6*v;
                kp.f_resid(3) = acc*dt + v;
                kp.A(0,0) = 1;
                kp.A(0,1) = 0;
                kp.A(0,2) = -tmp_d3;
                kp.A(0,3) = tmp_d0;
                kp.A(1,0) = 0;
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d1;
                kp.A(1,3) = tmp_d2;
                kp.A(2,0) = 0;
                kp.A(2,1) = 0;
                kp.A(2,2) = 1;
                kp.A(2,3) = tmp_d6;
                kp.A(3,0) = 0;
                kp.A(3,1) = 0;
                kp.A(3,2) = 0;
                kp.A(3,3) = 1;
                kp.B(0,0) = 0;
                kp.B(0,1) = 0;
                kp.B(1,0) = 0;
                kp.B(1,1) = 0;
                kp.B(2,0) = 0;
                kp.B(2,1) = tmp_d5*v*(pow(tmp_d4, 2) + 1);
                kp.B(3,0) = dt;
                kp.B(3,1) = 0;
                break;
            }
            case IntegratorType::RK2_EXPLICIT:
            case IntegratorType::RK2_IMPLICIT:
            {
                T tmp_d0 = acc*dt;
                T tmp_d1 = 0.5*tmp_d0 + v;
                T tmp_d2 = 1.0/L;
                T tmp_d3 = tan(steer);
                T tmp_d4 = tmp_d2*tmp_d3;
                T tmp_d5 = dt*tmp_d4;
                T tmp_d6 = tmp_d1*tmp_d5;
                T tmp_d7 = theta + 0.5*tmp_d6;
                T tmp_d8 = cos(tmp_d7);
                T tmp_d9 = dt*tmp_d8;
                T tmp_d10 = tmp_d1*tmp_d9;
                T tmp_d11 = sin(tmp_d7);
                T tmp_d12 = dt*tmp_d11;
                T tmp_d13 = tmp_d1*tmp_d12;
                T tmp_d14 = 0.5*pow(dt, 2);
                T tmp_d15 = tmp_d11*tmp_d14;
                T tmp_d16 = tmp_d1*tmp_d4;
                T tmp_d17 = tmp_d14*tmp_d8;
                T tmp_d18 = 0.25*pow(dt, 3)*tmp_d16;
                T tmp_d19 = tmp_d2*(pow(tmp_d3, 2) + 1);
                T tmp_d20 = pow(tmp_d1, 2)*tmp_d19;
                kp.f_resid(0) = tmp_d10 + x;
                kp.f_resid(1) = tmp_d13 + y;
                kp.f_resid(2) = theta + tmp_d6;
                kp.f_resid(3) = tmp_d0 + v;
                kp.A(0,0) = 1;
                kp.A(0,1) = 0;
                kp.A(0,2) = -tmp_d13;
                kp.A(0,3) = -tmp_d15*tmp_d16 + tmp_d9;
                kp.A(1,0) = 0;
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d10;
                kp.A(1,3) = tmp_d12 + tmp_d16*tmp_d17;
                kp.A(2,0) = 0;
                kp.A(2,1) = 0;
                kp.A(2,2) = 1;
                kp.A(2,3) = tmp_d5;
                kp.A(3,0) = 0;
                kp.A(3,1) = 0;
                kp.A(3,2) = 0;
                kp.A(3,3) = 1;
                kp.B(0,0) = -tmp_d11*tmp_d18 + tmp_d17;
                kp.B(0,1) = -tmp_d15*tmp_d20;
                kp.B(1,0) = tmp_d15 + tmp_d18*tmp_d8;
                kp.B(1,1) = tmp_d17*tmp_d20;
                kp.B(2,0) = tmp_d14*tmp_d4;
                kp.B(2,1) = dt*tmp_d1*tmp_d19;
                kp.B(3,0) = dt;
                kp.B(3,1) = 0;
                break;
            }
            case IntegratorType::RK4_EXPLICIT:
            case IntegratorType::RK4_IMPLICIT:
            {
                T tmp_d0 = cos(theta);
                T tmp_d1 = acc*dt;
                T tmp_d2 = tmp_d1 + v;
                T tmp_d3 = 1.5*tmp_d1 + v;
                T tmp_d4 = 1.0/L;
                T tmp_d5 = tan(steer);
                T tmp_d6 = tmp_d4*tmp_d5;
                T tmp_d7 = dt*tmp_d6;
                T tmp_d8 = theta + tmp_d3*tmp_d7;
                T tmp_d9 = cos(tmp_d8);
                T tmp_d10 = tmp_d2*tmp_d9;
                T tmp_d11 = 0.5*tmp_d1 + v;
                T tmp_d12 = 0.5*tmp_d7;
                T tmp_d13 = theta + tmp_d11*tmp_d12;
                T tmp_d14 = cos(tmp_d13);
                T tmp_d15 = 2*tmp_d14;
                T tmp_d16 = 1.0*tmp_d1 + v;
                T tmp_d17 = theta + tmp_d12*tmp_d16;
                T tmp_d18 = cos(tmp_d17);
                T tmp_d19 = 2*tmp_d18;
                T tmp_d20 = 0.16666666666666666*dt;
                T tmp_d21 = tmp_d20*(tmp_d0*v + tmp_d10 + tmp_d11*tmp_d15 + tmp_d11*tmp_d19);
                T tmp_d22 = sin(theta);
                T tmp_d23 = sin(tmp_d8);
                T tmp_d24 = tmp_d2*tmp_d23;
                T tmp_d25 = sin(tmp_d13);
                T tmp_d26 = 2*tmp_d25;
                T tmp_d27 = sin(tmp_d17);
                T tmp_d28 = 2*tmp_d27;
                T tmp_d29 = tmp_d11*tmp_d26 + tmp_d11*tmp_d28 + tmp_d22*v + tmp_d24;
                T tmp_d30 = 4*tmp_d11;
                T tmp_d31 = 1.0*dt;
                T tmp_d32 = tmp_d25*tmp_d31;
                T tmp_d33 = tmp_d11*tmp_d6;
                T tmp_d34 = tmp_d27*tmp_d31;
                T tmp_d35 = tmp_d14*tmp_d31;
                T tmp_d36 = tmp_d18*tmp_d31;
                T tmp_d37 = pow(dt, 2)*tmp_d6;
                T tmp_d38 = 1.5*tmp_d37;
                T tmp_d39 = 0.5*tmp_d37;
                T tmp_d40 = tmp_d11*tmp_d39;
                T tmp_d41 = 1.0*tmp_d11*tmp_d37;
                T tmp_d42 = tmp_d4*(pow(tmp_d5, 2) + 1);
                T tmp_d43 = pow(tmp_d11, 2)*tmp_d42;
                T tmp_d44 = dt*tmp_d3*tmp_d42;
                T tmp_d45 = tmp_d11*tmp_d16*tmp_d42;
                kp.f_resid(0) = tmp_d21 + x;
                kp.f_resid(1) = tmp_d20*tmp_d29 + y;
                kp.f_resid(2) = theta + tmp_d20*(tmp_d2*tmp_d6 + tmp_d30*tmp_d6 + tmp_d6*v);
                kp.f_resid(3) = tmp_d16;
                kp.A(0,0) = 1;
                kp.A(0,1) = 0;
                kp.A(0,2) = -tmp_d20*tmp_d29;
                kp.A(0,3) = tmp_d20*(tmp_d0 + tmp_d15 + tmp_d19 - tmp_d24*tmp_d7 - tmp_d32*tmp_d33 - tmp_d33*tmp_d34 + tmp_d9);
                kp.A(1,0) = 0;
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d21;
                kp.A(1,3) = tmp_d20*(tmp_d10*tmp_d7 + tmp_d22 + tmp_d23 + tmp_d26 + tmp_d28 + tmp_d33*tmp_d35 + tmp_d33*tmp_d36);
                kp.A(2,0) = 0;
                kp.A(2,1) = 0;
                kp.A(2,2) = 1;
                kp.A(2,3) = tmp_d31*tmp_d6;
                kp.A(3,0) = 0;
                kp.A(3,1) = 0;
                kp.A(3,2) = 0;
                kp.A(3,3) = 1;
                kp.B(0,0) = tmp_d20*(dt*tmp_d9 - tmp_d24*tmp_d38 - tmp_d25*tmp_d40 - tmp_d27*tmp_d41 + tmp_d35 + tmp_d36);
                kp.B(0,1) = tmp_d20*(-tmp_d24*tmp_d44 - tmp_d32*tmp_d43 - tmp_d34*tmp_d45);
                kp.B(1,0) = tmp_d20*(dt*tmp_d23 + tmp_d10*tmp_d38 + tmp_d14*tmp_d40 + tmp_d18*tmp_d41 + tmp_d32 + tmp_d34);
                kp.B(1,1) = tmp_d20*(tmp_d10*tmp_d44 + tmp_d35*tmp_d43 + tmp_d36*tmp_d45);
                kp.B(2,0) = tmp_d39;
                kp.B(2,1) = tmp_d20*(tmp_d2*tmp_d42 + tmp_d30*tmp_d42 + tmp_d42*v);
                kp.B(3,0) = tmp_d31;
                kp.B(3,1) = 0;
                break;
            }
        }
    }

    // --- 2. Compute Constraints (g_val, C, D) ---
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
                T x = kp.x(0);
        T y = kp.x(1);
        T theta = kp.x(2);
        T v = kp.x(3);
        T acc = kp.u(0);
        T steer = kp.u(1);
        T v_ref = kp.p(0);
        T x_ref = kp.p(1);
        T y_ref = kp.p(2);
        T obs_x = kp.p(3);
        T obs_y = kp.p(4);
        T obs_rad = kp.p(5);
        T L = kp.p(6);
        T car_rad = kp.p(7);
        T w_pos = kp.p(8);
        T w_vel = kp.p(9);
        T w_theta = kp.p(10);
        T w_acc = kp.p(11);
        T w_steer = kp.p(12);

        T tmp_c0 = -obs_x + x;
        T tmp_c1 = -obs_y + y;
        T tmp_c2 = sqrt(pow(tmp_c0, 2) + pow(tmp_c1, 2) + 9.9999999999999995e-7);
        T tmp_c3 = 1.0/tmp_c2;

        // g_val
        kp.g_val(0,0) = acc - 3.0;
        kp.g_val(1,0) = -acc - 3.0;
        kp.g_val(2,0) = steer - 0.5;
        kp.g_val(3,0) = -steer - 0.5;
        kp.g_val(4,0) = car_rad + obs_rad - tmp_c2;

        // C
        kp.C(0,0) = 0;
        kp.C(0,1) = 0;
        kp.C(0,2) = 0;
        kp.C(0,3) = 0;
        kp.C(1,0) = 0;
        kp.C(1,1) = 0;
        kp.C(1,2) = 0;
        kp.C(1,3) = 0;
        kp.C(2,0) = 0;
        kp.C(2,1) = 0;
        kp.C(2,2) = 0;
        kp.C(2,3) = 0;
        kp.C(3,0) = 0;
        kp.C(3,1) = 0;
        kp.C(3,2) = 0;
        kp.C(3,3) = 0;
        kp.C(4,0) = -tmp_c0*tmp_c3;
        kp.C(4,1) = -tmp_c1*tmp_c3;
        kp.C(4,2) = 0;
        kp.C(4,3) = 0;

        // D
        kp.D(0,0) = 1;
        kp.D(0,1) = 0;
        kp.D(1,0) = -1;
        kp.D(1,1) = 0;
        kp.D(2,0) = 0;
        kp.D(2,1) = 1;
        kp.D(3,0) = 0;
        kp.D(3,1) = -1;
        kp.D(4,0) = 0;
        kp.D(4,1) = 0;

    }

    // --- 3. Compute Cost (Implemented via template for Exact/GN) ---
    template<typename T, bool Exact>
    static void compute_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x = kp.x(0);
        T y = kp.x(1);
        T theta = kp.x(2);
        T v = kp.x(3);
        T acc = kp.u(0);
        T steer = kp.u(1);
        T v_ref = kp.p(0);
        T x_ref = kp.p(1);
        T y_ref = kp.p(2);
        T obs_x = kp.p(3);
        T obs_y = kp.p(4);
        T obs_rad = kp.p(5);
        T L = kp.p(6);
        T car_rad = kp.p(7);
        T w_pos = kp.p(8);
        T w_vel = kp.p(9);
        T w_theta = kp.p(10);
        T w_acc = kp.p(11);
        T w_steer = kp.p(12);
        T lam_0 = kp.lam(0);
        T lam_1 = kp.lam(1);
        T lam_2 = kp.lam(2);
        T lam_3 = kp.lam(3);
        T lam_4 = kp.lam(4);

        T tmp_j0 = 2*w_theta;
        T tmp_j1 = 2*w_acc;
        T tmp_j2 = 2*w_steer;
        T tmp_j3 = 2*w_pos;
        T tmp_j4 = obs_x - x;
        T tmp_j5 = -tmp_j4;
        T tmp_j6 = obs_y - y;
        T tmp_j7 = -tmp_j6;
        T tmp_j8 = pow(tmp_j5, 2) + pow(tmp_j7, 2) + 9.9999999999999995e-7;
        T tmp_j9 = pow(tmp_j8, -1.0/2.0);
        T tmp_j10 = pow(tmp_j8, -3.0/2.0);
        T tmp_j11 = tmp_j10*tmp_j5;
        T tmp_j12 = -lam_4*tmp_j11*tmp_j6;

        // q
        kp.q(0,0) = w_pos*(2*x - 2*x_ref);
        kp.q(1,0) = w_pos*(2*y - 2*y_ref);
        kp.q(2,0) = theta*tmp_j0;
        kp.q(3,0) = w_vel*(2*v - 2*v_ref);

        // r
        kp.r(0,0) = acc*tmp_j1;
        kp.r(1,0) = steer*tmp_j2;

        // Q (Conditionally Exact)
        kp.Q(0,0) = tmp_j3;
        if constexpr (Exact) kp.Q(0,0) += lam_4*(-tmp_j11*tmp_j4 - tmp_j9);
        kp.Q(0,1) = 0;
        if constexpr (Exact) kp.Q(0,1) += tmp_j12;
        kp.Q(0,2) = 0;
        kp.Q(0,3) = 0;
        kp.Q(1,0) = 0;
        if constexpr (Exact) kp.Q(1,0) += tmp_j12;
        kp.Q(1,1) = tmp_j3;
        if constexpr (Exact) kp.Q(1,1) += lam_4*(-tmp_j10*tmp_j6*tmp_j7 - tmp_j9);
        kp.Q(1,2) = 0;
        kp.Q(1,3) = 0;
        kp.Q(2,0) = 0;
        kp.Q(2,1) = 0;
        kp.Q(2,2) = tmp_j0;
        kp.Q(2,3) = 0;
        kp.Q(3,0) = 0;
        kp.Q(3,1) = 0;
        kp.Q(3,2) = 0;
        kp.Q(3,3) = 2*w_vel;

        // R (Conditionally Exact)
        kp.R(0,0) = tmp_j1;
        kp.R(0,1) = 0;
        kp.R(1,0) = 0;
        kp.R(1,1) = tmp_j2;

        // H (Conditionally Exact)
        kp.H(0,0) = 0;
        kp.H(0,1) = 0;
        kp.H(0,2) = 0;
        kp.H(0,3) = 0;
        kp.H(1,0) = 0;
        kp.H(1,1) = 0;
        kp.H(1,2) = 0;
        kp.H(1,3) = 0;

        kp.cost = pow(acc, 2)*w_acc + pow(steer, 2)*w_steer + pow(theta, 2)*w_theta + w_pos*pow(x - x_ref, 2) + w_pos*pow(y - y_ref, 2) + w_vel*pow(v - v_ref, 2);
    }

template<typename T>
    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_cost_impl<T, false>(kp);
    }

    template<typename T>
    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_cost_impl<T, true>(kp);
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
};
}
