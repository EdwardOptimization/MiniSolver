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
    static const int NP=6;

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
    };


    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in) 
    {
                T x = x_in(0);
        T y = x_in(1);
        T theta = x_in(2);
        T v = x_in(3);
        T acc = u_in(0);
        T steer = u_in(1);

        MSVec<T, NX> xdot;
        xdot(0) = v*cos(theta);
        xdot(1) = v*sin(theta);
        xdot(2) = 0.40000000000000002*v*tan(steer);
        xdot(3) = acc;
        return xdot;
    }

    // --- Integrator Interface ---
    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& u,
        double dt,
        IntegratorType type)
    {
        switch(type) {
            case IntegratorType::EULER_EXPLICIT: return x + dynamics_continuous(x, u) * dt;
            default: // RK4 Explicit
            {
               auto k1 = dynamics_continuous(x, u);
               auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u);
               auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u);
               auto k4 = dynamics_continuous<T>(x + k3 * dt, u);
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

        T tmp_d0 = cos(theta);
        T tmp_d1 = acc*dt;
        T tmp_d2 = tmp_d1 + v;
        T tmp_d3 = 1.5*tmp_d1 + v;
        T tmp_d4 = tan(steer);
        T tmp_d5 = 0.40000000000000002*tmp_d4;
        T tmp_d6 = dt*tmp_d5;
        T tmp_d7 = theta + tmp_d3*tmp_d6;
        T tmp_d8 = cos(tmp_d7);
        T tmp_d9 = tmp_d2*tmp_d8;
        T tmp_d10 = 0.5*tmp_d1 + v;
        T tmp_d11 = 0.20000000000000001*tmp_d4;
        T tmp_d12 = dt*tmp_d11;
        T tmp_d13 = theta + tmp_d10*tmp_d12;
        T tmp_d14 = cos(tmp_d13);
        T tmp_d15 = 2*tmp_d14;
        T tmp_d16 = 1.0*tmp_d1 + v;
        T tmp_d17 = theta + tmp_d12*tmp_d16;
        T tmp_d18 = cos(tmp_d17);
        T tmp_d19 = 2*tmp_d18;
        T tmp_d20 = 0.16666666666666666*dt;
        T tmp_d21 = tmp_d20*(tmp_d0*v + tmp_d10*tmp_d15 + tmp_d10*tmp_d19 + tmp_d9);
        T tmp_d22 = sin(theta);
        T tmp_d23 = sin(tmp_d7);
        T tmp_d24 = tmp_d2*tmp_d23;
        T tmp_d25 = sin(tmp_d13);
        T tmp_d26 = 2*tmp_d25;
        T tmp_d27 = sin(tmp_d17);
        T tmp_d28 = 2*tmp_d27;
        T tmp_d29 = tmp_d10*tmp_d26 + tmp_d10*tmp_d28 + tmp_d22*v + tmp_d24;
        T tmp_d30 = 1.6000000000000001*tmp_d10;
        T tmp_d31 = tmp_d10*tmp_d6;
        T tmp_d32 = pow(dt, 2);
        T tmp_d33 = 0.60000000000000009*tmp_d32*tmp_d4;
        T tmp_d34 = tmp_d11*tmp_d32;
        T tmp_d35 = tmp_d10*tmp_d34;
        T tmp_d36 = tmp_d10*tmp_d27;
        T tmp_d37 = tmp_d32*tmp_d5;
        T tmp_d38 = pow(tmp_d4, 2) + 1;
        T tmp_d39 = 0.40000000000000002*tmp_d38;
        T tmp_d40 = dt*tmp_d39;
        T tmp_d41 = pow(tmp_d10, 2)*tmp_d40;
        T tmp_d42 = tmp_d3*tmp_d40;
        T tmp_d43 = tmp_d16*tmp_d40;
        T tmp_d44 = 1.0*dt;
        T tmp_d45 = tmp_d10*tmp_d18;

        // f_resid
        kp.f_resid(0,0) = tmp_d21 + x;
        kp.f_resid(1,0) = tmp_d20*tmp_d29 + y;
        kp.f_resid(2,0) = theta + tmp_d20*(tmp_d2*tmp_d5 + tmp_d30*tmp_d4 + tmp_d5*v);
        kp.f_resid(3,0) = tmp_d16;

        // A
        kp.A(0,0) = 1;
        kp.A(0,1) = 0;
        kp.A(0,2) = -tmp_d20*tmp_d29;
        kp.A(0,3) = tmp_d20*(tmp_d0 + tmp_d15 + tmp_d19 - tmp_d24*tmp_d6 - tmp_d25*tmp_d31 - tmp_d27*tmp_d31 + tmp_d8);
        kp.A(1,0) = 0;
        kp.A(1,1) = 1;
        kp.A(1,2) = tmp_d21;
        kp.A(1,3) = tmp_d20*(tmp_d14*tmp_d31 + tmp_d18*tmp_d31 + tmp_d22 + tmp_d23 + tmp_d26 + tmp_d28 + tmp_d6*tmp_d9);
        kp.A(2,0) = 0;
        kp.A(2,1) = 0;
        kp.A(2,2) = 1;
        kp.A(2,3) = tmp_d6;
        kp.A(3,0) = 0;
        kp.A(3,1) = 0;
        kp.A(3,2) = 0;
        kp.A(3,3) = 1;

        // B
        kp.B(0,0) = tmp_d20*(1.0*dt*tmp_d14 + 1.0*dt*tmp_d18 + dt*tmp_d8 - tmp_d24*tmp_d33 - tmp_d25*tmp_d35 - tmp_d36*tmp_d37);
        kp.B(0,1) = tmp_d20*(-tmp_d24*tmp_d42 - tmp_d25*tmp_d41 - tmp_d36*tmp_d43);
        kp.B(1,0) = tmp_d20*(dt*tmp_d23 + tmp_d14*tmp_d35 + tmp_d25*tmp_d44 + tmp_d27*tmp_d44 + tmp_d33*tmp_d9 + tmp_d37*tmp_d45);
        kp.B(1,1) = tmp_d20*(tmp_d14*tmp_d41 + tmp_d42*tmp_d9 + tmp_d43*tmp_d45);
        kp.B(2,0) = tmp_d34;
        kp.B(2,1) = tmp_d20*(tmp_d2*tmp_d39 + tmp_d30*tmp_d38 + tmp_d39*v);
        kp.B(3,0) = tmp_d44;
        kp.B(3,1) = 0;

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

        T tmp_c0 = -obs_x + x;
        T tmp_c1 = -obs_y + y;
        T tmp_c2 = sqrt(pow(tmp_c0, 2) + pow(tmp_c1, 2) + 9.9999999999999995e-7);
        T tmp_c3 = 1.0/tmp_c2;

        // g_val
        kp.g_val(0,0) = acc - 3.0;
        kp.g_val(1,0) = -acc - 3.0;
        kp.g_val(2,0) = steer - 0.5;
        kp.g_val(3,0) = -steer - 0.5;
        kp.g_val(4,0) = obs_rad - tmp_c2 + 1.0;

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

    // --- 3. Compute Cost (q, r, Q, R, H, cost) ---
    template<typename T>
    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) {
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


        // q
        kp.q(0,0) = 2.0*x - 2.0*x_ref;
        kp.q(1,0) = 2.0*y - 2.0*y_ref;
        kp.q(2,0) = 0.20000000000000001*theta;
        kp.q(3,0) = 2.0*v - 2.0*v_ref;

        // r
        kp.r(0,0) = 0.20000000000000001*acc;
        kp.r(1,0) = 2.0*steer;

        // Q
        kp.Q(0,0) = 2.0;
        kp.Q(0,1) = 0;
        kp.Q(0,2) = 0;
        kp.Q(0,3) = 0;
        kp.Q(1,0) = 0;
        kp.Q(1,1) = 2.0;
        kp.Q(1,2) = 0;
        kp.Q(1,3) = 0;
        kp.Q(2,0) = 0;
        kp.Q(2,1) = 0;
        kp.Q(2,2) = 0.20000000000000001;
        kp.Q(2,3) = 0;
        kp.Q(3,0) = 0;
        kp.Q(3,1) = 0;
        kp.Q(3,2) = 0;
        kp.Q(3,3) = 2.0;

        // R
        kp.R(0,0) = 0.20000000000000001;
        kp.R(0,1) = 0;
        kp.R(1,0) = 0;
        kp.R(1,1) = 2.0;

        // H
        kp.H(0,0) = 0;
        kp.H(0,1) = 0;
        kp.H(0,2) = 0;
        kp.H(0,3) = 0;
        kp.H(1,0) = 0;
        kp.H(1,1) = 0;
        kp.H(1,2) = 0;
        kp.H(1,3) = 0;

        kp.cost = 0.10000000000000001*pow(acc, 2) + 1.0*pow(steer, 2) + 0.10000000000000001*pow(theta, 2) + 1.0*pow(v - v_ref, 2) + 1.0*pow(x - x_ref, 2) + 1.0*pow(y - y_ref, 2);

    }

    // --- 4. Compute All (Convenience) ---
    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};
}

