#pragma once
#include "core/types.h"
#include "core/solver_options.h"
#include "core/matrix_defs.h"
#include <cmath>
#include <string>
#include <array>

namespace minisolver {

struct BicycleExtModel {
    // --- Constants ---
    static const int NX=6;
    static const int NU=2;
    static const int NC=10;
    static const int NP=15;

    // --- Name Arrays (for Map Construction) ---
    static constexpr std::array<const char*, NX> state_names = {
        "x",
        "y",
        "theta",
        "kappa",
        "v",
        "a",
    };

    static constexpr std::array<const char*, NU> control_names = {
        "dkappa",
        "jerk",
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
        "w_kappa",
        "w_a",
        "w_dkappa",
        "w_jerk",
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
        T kappa = x_in(3);
        T v = x_in(4);
        T a = x_in(5);
        T dkappa = u_in(0);
        T jerk = u_in(1);
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
        T w_kappa = p_in(11);
        T w_a = p_in(12);
        T w_dkappa = p_in(13);
        T w_jerk = p_in(14);

        MSVec<T, NX> xdot;
        xdot(0) = v*cos(theta);
        xdot(1) = v*sin(theta);
        xdot(2) = kappa*v;
        xdot(3) = dkappa;
        xdot(4) = a;
        xdot(5) = jerk;
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
            case IntegratorType::EULER_EXPLICIT: return x + dynamics_continuous(x, u, p) * dt;
            default: // RK4 Explicit
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
        T kappa = kp.x(3);
        T v = kp.x(4);
        T a = kp.x(5);
        T dkappa = kp.u(0);
        T jerk = kp.u(1);
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
        T w_kappa = kp.p(11);
        T w_a = kp.p(12);
        T w_dkappa = kp.p(13);
        T w_jerk = kp.p(14);

        T tmp_d0 = cos(theta);
        T tmp_d1 = 0.5*dt;
        T tmp_d2 = a*tmp_d1 + v;
        T tmp_d3 = kappa*tmp_d2;
        T tmp_d4 = theta + tmp_d1*tmp_d3;
        T tmp_d5 = cos(tmp_d4);
        T tmp_d6 = 2*tmp_d5;
        T tmp_d7 = dt*(a + jerk*tmp_d1);
        T tmp_d8 = tmp_d7 + v;
        T tmp_d9 = 1.5*tmp_d7 + v;
        T tmp_d10 = dkappa*dt;
        T tmp_d11 = kappa + 0.5*tmp_d10;
        T tmp_d12 = dt*tmp_d11;
        T tmp_d13 = theta + tmp_d12*tmp_d9;
        T tmp_d14 = cos(tmp_d13);
        T tmp_d15 = tmp_d14*tmp_d8;
        T tmp_d16 = 0.5*tmp_d7;
        T tmp_d17 = tmp_d16 + v;
        T tmp_d18 = tmp_d16 + tmp_d2;
        T tmp_d19 = theta + tmp_d1*tmp_d11*tmp_d18;
        T tmp_d20 = cos(tmp_d19);
        T tmp_d21 = 2*tmp_d20;
        T tmp_d22 = 0.16666666666666666*dt;
        T tmp_d23 = tmp_d22*(tmp_d0*v + tmp_d15 + tmp_d17*tmp_d21 + tmp_d2*tmp_d6);
        T tmp_d24 = sin(theta);
        T tmp_d25 = sin(tmp_d4);
        T tmp_d26 = 2*tmp_d25;
        T tmp_d27 = sin(tmp_d13);
        T tmp_d28 = tmp_d27*tmp_d8;
        T tmp_d29 = sin(tmp_d19);
        T tmp_d30 = 2*tmp_d29;
        T tmp_d31 = tmp_d17*tmp_d30 + tmp_d2*tmp_d26 + tmp_d24*v + tmp_d28;
        T tmp_d32 = 2*tmp_d11;
        T tmp_d33 = kappa + tmp_d10;
        T tmp_d34 = 1.0*dt;
        T tmp_d35 = pow(tmp_d2, 2);
        T tmp_d36 = tmp_d25*tmp_d34;
        T tmp_d37 = dt*tmp_d9;
        T tmp_d38 = tmp_d29*tmp_d34;
        T tmp_d39 = tmp_d17*tmp_d38;
        T tmp_d40 = pow(dt, 2);
        T tmp_d41 = 0.5*tmp_d40;
        T tmp_d42 = tmp_d3*tmp_d41;
        T tmp_d43 = tmp_d11*tmp_d40;
        T tmp_d44 = 1.5*tmp_d43;
        T tmp_d45 = tmp_d17*tmp_d29;
        T tmp_d46 = 1.0*tmp_d43;
        T tmp_d47 = tmp_d34*tmp_d5;
        T tmp_d48 = tmp_d17*tmp_d20*tmp_d34;
        T tmp_d49 = tmp_d17*tmp_d20;
        T tmp_d50 = tmp_d41*tmp_d9;
        T tmp_d51 = tmp_d29*tmp_d41;
        T tmp_d52 = tmp_d17*tmp_d18;
        T tmp_d53 = pow(dt, 3)*tmp_d11;
        T tmp_d54 = 0.75*tmp_d53;
        T tmp_d55 = 0.25*tmp_d53;

        // f_resid
        kp.f_resid(0,0) = tmp_d23 + x;
        kp.f_resid(1,0) = tmp_d22*tmp_d31 + y;
        kp.f_resid(2,0) = theta + tmp_d22*(kappa*v + tmp_d17*tmp_d32 + tmp_d2*tmp_d32 + tmp_d33*tmp_d8);
        kp.f_resid(3,0) = kappa + 1.0*tmp_d10;
        kp.f_resid(4,0) = tmp_d22*(6*a + 3.0*dt*jerk) + v;
        kp.f_resid(5,0) = a + jerk*tmp_d34;

        // A
        kp.A(0,0) = 1;
        kp.A(0,1) = 0;
        kp.A(0,2) = -tmp_d22*tmp_d31;
        kp.A(0,3) = tmp_d22*(-tmp_d18*tmp_d39 - tmp_d28*tmp_d37 - tmp_d35*tmp_d36);
        kp.A(0,4) = tmp_d22*(tmp_d0 - tmp_d11*tmp_d39 - tmp_d12*tmp_d28 + tmp_d14 + tmp_d21 - tmp_d3*tmp_d36 + tmp_d6);
        kp.A(0,5) = tmp_d22*(dt*tmp_d14 + 1.0*dt*tmp_d20 + 1.0*dt*tmp_d5 - tmp_d25*tmp_d42 - tmp_d28*tmp_d44 - tmp_d45*tmp_d46);
        kp.A(1,0) = 0;
        kp.A(1,1) = 1;
        kp.A(1,2) = tmp_d23;
        kp.A(1,3) = tmp_d22*(tmp_d15*tmp_d37 + tmp_d18*tmp_d48 + tmp_d35*tmp_d47);
        kp.A(1,4) = tmp_d22*(tmp_d11*tmp_d48 + tmp_d12*tmp_d15 + tmp_d24 + tmp_d26 + tmp_d27 + tmp_d3*tmp_d47 + tmp_d30);
        kp.A(1,5) = tmp_d22*(dt*tmp_d27 + tmp_d15*tmp_d44 + tmp_d36 + tmp_d38 + tmp_d42*tmp_d5 + tmp_d46*tmp_d49);
        kp.A(2,0) = 0;
        kp.A(2,1) = 0;
        kp.A(2,2) = 1;
        kp.A(2,3) = tmp_d22*(a*tmp_d34 + 2.0*tmp_d7 + 6*v);
        kp.A(2,4) = tmp_d22*(6*kappa + 3.0*tmp_d10);
        kp.A(2,5) = tmp_d22*(dt*tmp_d33 + 2.0*tmp_d12);
        kp.A(3,0) = 0;
        kp.A(3,1) = 0;
        kp.A(3,2) = 0;
        kp.A(3,3) = 1;
        kp.A(3,4) = 0;
        kp.A(3,5) = 0;
        kp.A(4,0) = 0;
        kp.A(4,1) = 0;
        kp.A(4,2) = 0;
        kp.A(4,3) = 0;
        kp.A(4,4) = 1;
        kp.A(4,5) = tmp_d34;
        kp.A(5,0) = 0;
        kp.A(5,1) = 0;
        kp.A(5,2) = 0;
        kp.A(5,3) = 0;
        kp.A(5,4) = 0;
        kp.A(5,5) = 1;

        // B
        kp.B(0,0) = tmp_d22*(-tmp_d28*tmp_d50 - tmp_d51*tmp_d52);
        kp.B(0,1) = tmp_d22*(0.5*tmp_d14*tmp_d40 + 0.5*tmp_d20*tmp_d40 - tmp_d28*tmp_d54 - tmp_d45*tmp_d55);
        kp.B(1,0) = tmp_d22*(tmp_d15*tmp_d50 + tmp_d20*tmp_d41*tmp_d52);
        kp.B(1,1) = tmp_d22*(tmp_d15*tmp_d54 + tmp_d27*tmp_d41 + tmp_d49*tmp_d55 + tmp_d51);
        kp.B(2,0) = tmp_d22*(dt*tmp_d8 + tmp_d17*tmp_d34 + tmp_d2*tmp_d34);
        kp.B(2,1) = tmp_d22*(tmp_d11*tmp_d41 + tmp_d33*tmp_d41);
        kp.B(3,0) = tmp_d34;
        kp.B(3,1) = 0;
        kp.B(4,0) = 0;
        kp.B(4,1) = tmp_d41;
        kp.B(5,0) = 0;
        kp.B(5,1) = tmp_d34;

    }

    // --- 2. Compute Constraints (g_val, C, D) ---
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
                T x = kp.x(0);
        T y = kp.x(1);
        T theta = kp.x(2);
        T kappa = kp.x(3);
        T v = kp.x(4);
        T a = kp.x(5);
        T dkappa = kp.u(0);
        T jerk = kp.u(1);
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
        T w_kappa = kp.p(11);
        T w_a = kp.p(12);
        T w_dkappa = kp.p(13);
        T w_jerk = kp.p(14);


        // g_val
        kp.g_val(0,0) = v - 15.0;
        kp.g_val(1,0) = -v;
        kp.g_val(2,0) = a - 5.0;
        kp.g_val(3,0) = -a - 5.0;
        kp.g_val(4,0) = kappa - 0.5;
        kp.g_val(5,0) = -kappa - 0.5;
        kp.g_val(6,0) = jerk - 50.0;
        kp.g_val(7,0) = -jerk - 50.0;
        kp.g_val(8,0) = dkappa - 2.0;
        kp.g_val(9,0) = -dkappa - 2.0;

        // C
        kp.C(0,0) = 0;
        kp.C(0,1) = 0;
        kp.C(0,2) = 0;
        kp.C(0,3) = 0;
        kp.C(0,4) = 1;
        kp.C(0,5) = 0;
        kp.C(1,0) = 0;
        kp.C(1,1) = 0;
        kp.C(1,2) = 0;
        kp.C(1,3) = 0;
        kp.C(1,4) = -1;
        kp.C(1,5) = 0;
        kp.C(2,0) = 0;
        kp.C(2,1) = 0;
        kp.C(2,2) = 0;
        kp.C(2,3) = 0;
        kp.C(2,4) = 0;
        kp.C(2,5) = 1;
        kp.C(3,0) = 0;
        kp.C(3,1) = 0;
        kp.C(3,2) = 0;
        kp.C(3,3) = 0;
        kp.C(3,4) = 0;
        kp.C(3,5) = -1;
        kp.C(4,0) = 0;
        kp.C(4,1) = 0;
        kp.C(4,2) = 0;
        kp.C(4,3) = 1;
        kp.C(4,4) = 0;
        kp.C(4,5) = 0;
        kp.C(5,0) = 0;
        kp.C(5,1) = 0;
        kp.C(5,2) = 0;
        kp.C(5,3) = -1;
        kp.C(5,4) = 0;
        kp.C(5,5) = 0;
        kp.C(6,0) = 0;
        kp.C(6,1) = 0;
        kp.C(6,2) = 0;
        kp.C(6,3) = 0;
        kp.C(6,4) = 0;
        kp.C(6,5) = 0;
        kp.C(7,0) = 0;
        kp.C(7,1) = 0;
        kp.C(7,2) = 0;
        kp.C(7,3) = 0;
        kp.C(7,4) = 0;
        kp.C(7,5) = 0;
        kp.C(8,0) = 0;
        kp.C(8,1) = 0;
        kp.C(8,2) = 0;
        kp.C(8,3) = 0;
        kp.C(8,4) = 0;
        kp.C(8,5) = 0;
        kp.C(9,0) = 0;
        kp.C(9,1) = 0;
        kp.C(9,2) = 0;
        kp.C(9,3) = 0;
        kp.C(9,4) = 0;
        kp.C(9,5) = 0;

        // D
        kp.D(0,0) = 0;
        kp.D(0,1) = 0;
        kp.D(1,0) = 0;
        kp.D(1,1) = 0;
        kp.D(2,0) = 0;
        kp.D(2,1) = 0;
        kp.D(3,0) = 0;
        kp.D(3,1) = 0;
        kp.D(4,0) = 0;
        kp.D(4,1) = 0;
        kp.D(5,0) = 0;
        kp.D(5,1) = 0;
        kp.D(6,0) = 0;
        kp.D(6,1) = 1;
        kp.D(7,0) = 0;
        kp.D(7,1) = -1;
        kp.D(8,0) = 1;
        kp.D(8,1) = 0;
        kp.D(9,0) = -1;
        kp.D(9,1) = 0;

    }

    // --- 3. Compute Cost (Implemented via template for Exact/GN) ---
    template<typename T, bool Exact>
    static void compute_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x = kp.x(0);
        T y = kp.x(1);
        T theta = kp.x(2);
        T kappa = kp.x(3);
        T v = kp.x(4);
        T a = kp.x(5);
        T dkappa = kp.u(0);
        T jerk = kp.u(1);
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
        T w_kappa = kp.p(11);
        T w_a = kp.p(12);
        T w_dkappa = kp.p(13);
        T w_jerk = kp.p(14);
        T lam_0 = kp.lam(0);
        T lam_1 = kp.lam(1);
        T lam_2 = kp.lam(2);
        T lam_3 = kp.lam(3);
        T lam_4 = kp.lam(4);
        T lam_5 = kp.lam(5);
        T lam_6 = kp.lam(6);
        T lam_7 = kp.lam(7);
        T lam_8 = kp.lam(8);
        T lam_9 = kp.lam(9);

        T tmp_j0 = 2*w_theta;
        T tmp_j1 = 2*w_kappa;
        T tmp_j2 = 2*w_a;
        T tmp_j3 = 2*w_dkappa;
        T tmp_j4 = 2*w_jerk;
        T tmp_j5 = 2*w_pos;

        // q
        kp.q(0,0) = w_pos*(2*x - 2*x_ref);
        kp.q(1,0) = w_pos*(2*y - 2*y_ref);
        kp.q(2,0) = theta*tmp_j0;
        kp.q(3,0) = kappa*tmp_j1;
        kp.q(4,0) = w_vel*(2*v - 2*v_ref);
        kp.q(5,0) = a*tmp_j2;

        // r
        kp.r(0,0) = dkappa*tmp_j3;
        kp.r(1,0) = jerk*tmp_j4;

        // Q (Conditionally Exact)
        kp.Q(0,0) = tmp_j5;
        kp.Q(0,1) = 0;
        kp.Q(0,2) = 0;
        kp.Q(0,3) = 0;
        kp.Q(0,4) = 0;
        kp.Q(0,5) = 0;
        kp.Q(1,0) = 0;
        kp.Q(1,1) = tmp_j5;
        kp.Q(1,2) = 0;
        kp.Q(1,3) = 0;
        kp.Q(1,4) = 0;
        kp.Q(1,5) = 0;
        kp.Q(2,0) = 0;
        kp.Q(2,1) = 0;
        kp.Q(2,2) = tmp_j0;
        kp.Q(2,3) = 0;
        kp.Q(2,4) = 0;
        kp.Q(2,5) = 0;
        kp.Q(3,0) = 0;
        kp.Q(3,1) = 0;
        kp.Q(3,2) = 0;
        kp.Q(3,3) = tmp_j1;
        kp.Q(3,4) = 0;
        kp.Q(3,5) = 0;
        kp.Q(4,0) = 0;
        kp.Q(4,1) = 0;
        kp.Q(4,2) = 0;
        kp.Q(4,3) = 0;
        kp.Q(4,4) = 2*w_vel;
        kp.Q(4,5) = 0;
        kp.Q(5,0) = 0;
        kp.Q(5,1) = 0;
        kp.Q(5,2) = 0;
        kp.Q(5,3) = 0;
        kp.Q(5,4) = 0;
        kp.Q(5,5) = tmp_j2;

        // R (Conditionally Exact)
        kp.R(0,0) = tmp_j3;
        kp.R(0,1) = 0;
        kp.R(1,0) = 0;
        kp.R(1,1) = tmp_j4;

        // H (Conditionally Exact)
        kp.H(0,0) = 0;
        kp.H(0,1) = 0;
        kp.H(0,2) = 0;
        kp.H(0,3) = 0;
        kp.H(0,4) = 0;
        kp.H(0,5) = 0;
        kp.H(1,0) = 0;
        kp.H(1,1) = 0;
        kp.H(1,2) = 0;
        kp.H(1,3) = 0;
        kp.H(1,4) = 0;
        kp.H(1,5) = 0;

        kp.cost = pow(a, 2)*w_a + pow(dkappa, 2)*w_dkappa + pow(jerk, 2)*w_jerk + pow(kappa, 2)*w_kappa + pow(theta, 2)*w_theta + w_pos*(pow(x - x_ref, 2) + pow(y - y_ref, 2)) + w_vel*pow(v - v_ref, 2);
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

