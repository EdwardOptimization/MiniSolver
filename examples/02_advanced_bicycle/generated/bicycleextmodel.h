#pragma once
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/matrix_defs.h"
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

    static constexpr std::array<double, NC> constraint_weights = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    static constexpr std::array<int, NC> constraint_types = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


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
        T theta = x_in(2);
        T kappa = x_in(3);
        T v = x_in(4);
        T a = x_in(5);
        T dkappa = u_in(0);
        T jerk = u_in(1);
        (void)p_in;

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
        T kappa = kp.x(3);
        T v = kp.x(4);
        T a = kp.x(5);
        T dkappa = kp.u(0);
        T jerk = kp.u(1);

        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
            case IntegratorType::EULER_IMPLICIT:
            {
                T tmp_d0 = dt*cos(theta);
                T tmp_d1 = tmp_d0*v;
                T tmp_d2 = dt*sin(theta);
                T tmp_d3 = tmp_d2*v;
                T tmp_d4 = dt*v;
                kp.f_resid(0) = tmp_d1 + x;
                kp.f_resid(1) = tmp_d3 + y;
                kp.f_resid(2) = kappa*tmp_d4 + theta;
                kp.f_resid(3) = dkappa*dt + kappa;
                kp.f_resid(4) = a*dt + v;
                kp.f_resid(5) = a + dt*jerk;
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(0,2) = -tmp_d3;
                kp.A(0,4) = tmp_d0;
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d1;
                kp.A(1,4) = tmp_d2;
                kp.A(2,2) = 1;
                kp.A(2,3) = tmp_d4;
                kp.A(2,4) = dt*kappa;
                kp.A(3,3) = 1;
                kp.A(4,4) = 1;
                kp.A(4,5) = dt;
                kp.A(5,5) = 1;
                kp.B.setZero();
                kp.B(3,0) = dt;
                kp.B(5,1) = dt;
                break;
            }
            case IntegratorType::RK2_EXPLICIT:
            case IntegratorType::RK2_IMPLICIT:
            {
                T tmp_d0 = 0.5*a*dt + v;
                T tmp_d1 = dt*tmp_d0;
                T tmp_d2 = 0.5*kappa*tmp_d1 + theta;
                T tmp_d3 = cos(tmp_d2);
                T tmp_d4 = dt*tmp_d0*tmp_d3;
                T tmp_d5 = sin(tmp_d2);
                T tmp_d6 = dt*tmp_d5;
                T tmp_d7 = tmp_d0*tmp_d6;
                T tmp_d8 = dkappa*dt;
                T tmp_d9 = kappa + 0.5*tmp_d8;
                T tmp_d10 = dt*jerk;
                T tmp_d11 = pow(tmp_d0, 2);
                T tmp_d12 = pow(dt, 2);
                T tmp_d13 = 0.5*tmp_d12;
                T tmp_d14 = tmp_d13*tmp_d5;
                T tmp_d15 = kappa*tmp_d0;
                T tmp_d16 = 0.25*pow(dt, 3)*tmp_d15;
                T tmp_d17 = tmp_d13*tmp_d3;
                kp.f_resid(0) = tmp_d4 + x;
                kp.f_resid(1) = tmp_d7 + y;
                kp.f_resid(2) = theta + tmp_d1*tmp_d9;
                kp.f_resid(3) = kappa + tmp_d8;
                kp.f_resid(4) = dt*(a + 0.5*tmp_d10) + v;
                kp.f_resid(5) = a + tmp_d10;
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(0,2) = -tmp_d7;
                kp.A(0,3) = -tmp_d11*tmp_d14;
                kp.A(0,4) = dt*tmp_d3 - tmp_d14*tmp_d15;
                kp.A(0,5) = 0.5*tmp_d12*tmp_d3 - tmp_d16*tmp_d5;
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d4;
                kp.A(1,3) = tmp_d11*tmp_d17;
                kp.A(1,4) = tmp_d15*tmp_d17 + tmp_d6;
                kp.A(1,5) = tmp_d14 + tmp_d16*tmp_d3;
                kp.A(2,2) = 1;
                kp.A(2,3) = tmp_d1;
                kp.A(2,4) = dt*tmp_d9;
                kp.A(2,5) = tmp_d13*tmp_d9;
                kp.A(3,3) = 1;
                kp.A(4,4) = 1;
                kp.A(4,5) = dt;
                kp.A(5,5) = 1;
                kp.B.setZero();
                kp.B(2,0) = tmp_d0*tmp_d13;
                kp.B(3,0) = dt;
                kp.B(4,1) = tmp_d13;
                kp.B(5,1) = dt;
                break;
            }
            case IntegratorType::RK4_EXPLICIT:
            case IntegratorType::RK4_IMPLICIT:
            {
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
                kp.f_resid(0) = tmp_d23 + x;
                kp.f_resid(1) = tmp_d22*tmp_d31 + y;
                kp.f_resid(2) = theta + tmp_d22*(kappa*v + tmp_d17*tmp_d32 + tmp_d2*tmp_d32 + tmp_d33*tmp_d8);
                kp.f_resid(3) = kappa + 1.0*tmp_d10;
                kp.f_resid(4) = tmp_d22*(6*a + 3.0*dt*jerk) + v;
                kp.f_resid(5) = a + jerk*tmp_d34;
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(0,2) = -tmp_d22*tmp_d31;
                kp.A(0,3) = tmp_d22*(-tmp_d18*tmp_d39 - tmp_d28*tmp_d37 - tmp_d35*tmp_d36);
                kp.A(0,4) = tmp_d22*(tmp_d0 - tmp_d11*tmp_d39 - tmp_d12*tmp_d28 + tmp_d14 + tmp_d21 - tmp_d3*tmp_d36 + tmp_d6);
                kp.A(0,5) = tmp_d22*(dt*tmp_d14 + 1.0*dt*tmp_d20 + 1.0*dt*tmp_d5 - tmp_d25*tmp_d42 - tmp_d28*tmp_d44 - tmp_d45*tmp_d46);
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d23;
                kp.A(1,3) = tmp_d22*(tmp_d15*tmp_d37 + tmp_d18*tmp_d48 + tmp_d35*tmp_d47);
                kp.A(1,4) = tmp_d22*(tmp_d11*tmp_d48 + tmp_d12*tmp_d15 + tmp_d24 + tmp_d26 + tmp_d27 + tmp_d3*tmp_d47 + tmp_d30);
                kp.A(1,5) = tmp_d22*(dt*tmp_d27 + tmp_d15*tmp_d44 + tmp_d36 + tmp_d38 + tmp_d42*tmp_d5 + tmp_d46*tmp_d49);
                kp.A(2,2) = 1;
                kp.A(2,3) = tmp_d22*(a*tmp_d34 + 2.0*tmp_d7 + 6*v);
                kp.A(2,4) = tmp_d22*(6*kappa + 3.0*tmp_d10);
                kp.A(2,5) = tmp_d22*(dt*tmp_d33 + 2.0*tmp_d12);
                kp.A(3,3) = 1;
                kp.A(4,4) = 1;
                kp.A(4,5) = tmp_d34;
                kp.A(5,5) = 1;
                kp.B.setZero();
                kp.B(0,0) = tmp_d22*(-tmp_d28*tmp_d50 - tmp_d51*tmp_d52);
                kp.B(0,1) = tmp_d22*(0.5*tmp_d14*tmp_d40 + 0.5*tmp_d20*tmp_d40 - tmp_d28*tmp_d54 - tmp_d45*tmp_d55);
                kp.B(1,0) = tmp_d22*(tmp_d15*tmp_d50 + tmp_d20*tmp_d41*tmp_d52);
                kp.B(1,1) = tmp_d22*(tmp_d15*tmp_d54 + tmp_d27*tmp_d41 + tmp_d49*tmp_d55 + tmp_d51);
                kp.B(2,0) = tmp_d22*(dt*tmp_d8 + tmp_d17*tmp_d34 + tmp_d2*tmp_d34);
                kp.B(2,1) = tmp_d22*(tmp_d11*tmp_d41 + tmp_d33*tmp_d41);
                kp.B(3,0) = tmp_d34;
                kp.B(4,1) = tmp_d41;
                kp.B(5,1) = tmp_d34;
                break;
            }
        }
    }

    // --- 2. Compute Constraints (g_val, C, D) ---
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T kappa = kp.x(3);
        T v = kp.x(4);
        T a = kp.x(5);
        T dkappa = kp.u(0);
        T jerk = kp.u(1);

        // --- Special Constraints Pre-Calculation ---


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
    template<typename T, int Mode>
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
        T w_pos = kp.p(8);
        T w_vel = kp.p(9);
        T w_theta = kp.p(10);
        T w_kappa = kp.p(11);
        T w_a = kp.p(12);
        T w_dkappa = kp.p(13);
        T w_jerk = kp.p(14);

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

        // Q (Mode 0=GN, 1=Exact)
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

        // R (Mode 0=GN, 1=Exact)
        kp.R(0,0) = tmp_j3;
        kp.R(0,1) = 0;
        kp.R(1,0) = 0;
        kp.R(1,1) = tmp_j4;

        // H (Mode 0=GN, 1=Exact)
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
        T P_0_2 = Vxx(0,2);
        T P_0_3 = Vxx(0,3);
        T P_0_4 = Vxx(0,4);
        T P_0_5 = Vxx(0,5);
        T P_1_1 = Vxx(1,1);
        T P_1_2 = Vxx(1,2);
        T P_1_3 = Vxx(1,3);
        T P_1_4 = Vxx(1,4);
        T P_1_5 = Vxx(1,5);
        T P_2_2 = Vxx(2,2);
        T P_2_3 = Vxx(2,3);
        T P_2_4 = Vxx(2,4);
        T P_2_5 = Vxx(2,5);
        T P_3_3 = Vxx(3,3);
        T P_3_4 = Vxx(3,4);
        T P_3_5 = Vxx(3,5);
        T P_4_4 = Vxx(4,4);
        T P_4_5 = Vxx(4,5);
        T P_5_5 = Vxx(5,5);
        T p_0 = Vx(0);
        T p_1 = Vx(1);
        T p_2 = Vx(2);
        T p_3 = Vx(3);
        T p_4 = Vx(4);
        T p_5 = Vx(5);
        T A_0_0 = kp.A(0,0);
        T A_0_2 = kp.A(0,2);
        T A_0_3 = kp.A(0,3);
        T A_0_4 = kp.A(0,4);
        T A_0_5 = kp.A(0,5);
        T A_1_1 = kp.A(1,1);
        T A_1_2 = kp.A(1,2);
        T A_1_3 = kp.A(1,3);
        T A_1_4 = kp.A(1,4);
        T A_1_5 = kp.A(1,5);
        T A_2_2 = kp.A(2,2);
        T A_2_3 = kp.A(2,3);
        T A_2_4 = kp.A(2,4);
        T A_2_5 = kp.A(2,5);
        T A_3_3 = kp.A(3,3);
        T A_4_4 = kp.A(4,4);
        T A_4_5 = kp.A(4,5);
        T A_5_5 = kp.A(5,5);
        T B_0_0 = kp.B(0,0);
        T B_0_1 = kp.B(0,1);
        T B_1_0 = kp.B(1,0);
        T B_1_1 = kp.B(1,1);
        T B_2_0 = kp.B(2,0);
        T B_2_1 = kp.B(2,1);
        T B_3_0 = kp.B(3,0);
        T B_4_1 = kp.B(4,1);
        T B_5_1 = kp.B(5,1);

        // CSE Intermediate Variables
        T tmp_ric0 = A_0_2*P_0_0;
        T tmp_ric1 = A_1_2*P_0_1;
        T tmp_ric2 = A_2_2*P_0_2;
        T tmp_ric3 = A_0_3*P_0_0;
        T tmp_ric4 = A_1_3*P_0_1;
        T tmp_ric5 = A_2_3*P_0_2;
        T tmp_ric6 = A_3_3*P_0_3;
        T tmp_ric7 = A_0_4*P_0_0;
        T tmp_ric8 = A_1_4*P_0_1;
        T tmp_ric9 = A_2_4*P_0_2;
        T tmp_ric10 = A_4_4*P_0_4;
        T tmp_ric11 = A_0_5*P_0_0;
        T tmp_ric12 = A_1_5*P_0_1;
        T tmp_ric13 = A_2_5*P_0_2;
        T tmp_ric14 = A_4_5*P_0_4;
        T tmp_ric15 = A_5_5*P_0_5;
        T tmp_ric16 = A_0_2*P_0_1;
        T tmp_ric17 = A_1_2*P_1_1;
        T tmp_ric18 = A_2_2*P_1_2;
        T tmp_ric19 = A_0_3*P_0_1;
        T tmp_ric20 = A_1_3*P_1_1;
        T tmp_ric21 = A_2_3*P_1_2;
        T tmp_ric22 = A_3_3*P_1_3;
        T tmp_ric23 = A_0_4*P_0_1;
        T tmp_ric24 = A_1_4*P_1_1;
        T tmp_ric25 = A_2_4*P_1_2;
        T tmp_ric26 = A_4_4*P_1_4;
        T tmp_ric27 = A_0_5*P_0_1;
        T tmp_ric28 = A_1_5*P_1_1;
        T tmp_ric29 = A_2_5*P_1_2;
        T tmp_ric30 = A_4_5*P_1_4;
        T tmp_ric31 = A_5_5*P_1_5;
        T tmp_ric32 = tmp_ric0 + tmp_ric1 + tmp_ric2;
        T tmp_ric33 = tmp_ric16 + tmp_ric17 + tmp_ric18;
        T tmp_ric34 = A_0_2*P_0_2 + A_1_2*P_1_2 + A_2_2*P_2_2;
        T tmp_ric35 = A_0_2*P_0_4 + A_1_2*P_1_4 + A_2_2*P_2_4;
        T tmp_ric36 = tmp_ric3 + tmp_ric4 + tmp_ric5 + tmp_ric6;
        T tmp_ric37 = tmp_ric19 + tmp_ric20 + tmp_ric21 + tmp_ric22;
        T tmp_ric38 = A_0_3*P_0_2 + A_1_3*P_1_2 + A_2_3*P_2_2 + A_3_3*P_2_3;
        T tmp_ric39 = A_0_3*P_0_4 + A_1_3*P_1_4 + A_2_3*P_2_4 + A_3_3*P_3_4;
        T tmp_ric40 = tmp_ric10 + tmp_ric7 + tmp_ric8 + tmp_ric9;
        T tmp_ric41 = tmp_ric23 + tmp_ric24 + tmp_ric25 + tmp_ric26;
        T tmp_ric42 = A_0_4*P_0_2 + A_1_4*P_1_2 + A_2_4*P_2_2 + A_4_4*P_2_4;
        T tmp_ric43 = A_0_4*P_0_4 + A_1_4*P_1_4 + A_2_4*P_2_4 + A_4_4*P_4_4;
        T tmp_ric44 = B_0_0*P_0_0 + B_1_0*P_0_1 + B_2_0*P_0_2 + B_3_0*P_0_3;
        T tmp_ric45 = B_0_0*P_0_1 + B_1_0*P_1_1 + B_2_0*P_1_2 + B_3_0*P_1_3;
        T tmp_ric46 = B_0_0*P_0_2 + B_1_0*P_1_2 + B_2_0*P_2_2 + B_3_0*P_2_3;
        T tmp_ric47 = B_0_0*P_0_3 + B_1_0*P_1_3 + B_2_0*P_2_3 + B_3_0*P_3_3;
        T tmp_ric48 = B_0_0*P_0_4 + B_1_0*P_1_4 + B_2_0*P_2_4 + B_3_0*P_3_4;
        T tmp_ric49 = B_0_0*P_0_5 + B_1_0*P_1_5 + B_2_0*P_2_5 + B_3_0*P_3_5;
        T tmp_ric50 = B_0_1*P_0_0 + B_1_1*P_0_1 + B_2_1*P_0_2 + B_4_1*P_0_4 + B_5_1*P_0_5;
        T tmp_ric51 = B_0_1*P_0_1 + B_1_1*P_1_1 + B_2_1*P_1_2 + B_4_1*P_1_4 + B_5_1*P_1_5;
        T tmp_ric52 = B_0_1*P_0_2 + B_1_1*P_1_2 + B_2_1*P_2_2 + B_4_1*P_2_4 + B_5_1*P_2_5;
        T tmp_ric53 = B_0_1*P_0_4 + B_1_1*P_1_4 + B_2_1*P_2_4 + B_4_1*P_4_4 + B_5_1*P_4_5;
        T tmp_ric54 = B_0_1*P_0_5 + B_1_1*P_1_5 + B_2_1*P_2_5 + B_4_1*P_4_5 + B_5_1*P_5_5;

        // Accumulate Results
        kp.Q_bar(0,0) += pow(A_0_0, 2)*P_0_0;
        kp.Q_bar(0,1) += A_0_0*A_1_1*P_0_1;
        kp.Q_bar(0,2) += A_0_0*tmp_ric0 + A_0_0*tmp_ric1 + A_0_0*tmp_ric2;
        kp.Q_bar(0,3) += A_0_0*tmp_ric3 + A_0_0*tmp_ric4 + A_0_0*tmp_ric5 + A_0_0*tmp_ric6;
        kp.Q_bar(0,4) += A_0_0*tmp_ric10 + A_0_0*tmp_ric7 + A_0_0*tmp_ric8 + A_0_0*tmp_ric9;
        kp.Q_bar(0,5) += A_0_0*tmp_ric11 + A_0_0*tmp_ric12 + A_0_0*tmp_ric13 + A_0_0*tmp_ric14 + A_0_0*tmp_ric15;
        kp.Q_bar(1,1) += pow(A_1_1, 2)*P_1_1;
        kp.Q_bar(1,2) += A_1_1*tmp_ric16 + A_1_1*tmp_ric17 + A_1_1*tmp_ric18;
        kp.Q_bar(1,3) += A_1_1*tmp_ric19 + A_1_1*tmp_ric20 + A_1_1*tmp_ric21 + A_1_1*tmp_ric22;
        kp.Q_bar(1,4) += A_1_1*tmp_ric23 + A_1_1*tmp_ric24 + A_1_1*tmp_ric25 + A_1_1*tmp_ric26;
        kp.Q_bar(1,5) += A_1_1*tmp_ric27 + A_1_1*tmp_ric28 + A_1_1*tmp_ric29 + A_1_1*tmp_ric30 + A_1_1*tmp_ric31;
        kp.Q_bar(2,2) += A_0_2*tmp_ric32 + A_1_2*tmp_ric33 + A_2_2*tmp_ric34;
        kp.Q_bar(2,3) += A_0_3*tmp_ric32 + A_1_3*tmp_ric33 + A_2_3*tmp_ric34 + A_3_3*(A_0_2*P_0_3 + A_1_2*P_1_3 + A_2_2*P_2_3);
        kp.Q_bar(2,4) += A_0_4*tmp_ric32 + A_1_4*tmp_ric33 + A_2_4*tmp_ric34 + A_4_4*tmp_ric35;
        kp.Q_bar(2,5) += A_0_5*tmp_ric32 + A_1_5*tmp_ric33 + A_2_5*tmp_ric34 + A_4_5*tmp_ric35 + A_5_5*(A_0_2*P_0_5 + A_1_2*P_1_5 + A_2_2*P_2_5);
        kp.Q_bar(3,3) += A_0_3*tmp_ric36 + A_1_3*tmp_ric37 + A_2_3*tmp_ric38 + A_3_3*(A_0_3*P_0_3 + A_1_3*P_1_3 + A_2_3*P_2_3 + A_3_3*P_3_3);
        kp.Q_bar(3,4) += A_0_4*tmp_ric36 + A_1_4*tmp_ric37 + A_2_4*tmp_ric38 + A_4_4*tmp_ric39;
        kp.Q_bar(3,5) += A_0_5*tmp_ric36 + A_1_5*tmp_ric37 + A_2_5*tmp_ric38 + A_4_5*tmp_ric39 + A_5_5*(A_0_3*P_0_5 + A_1_3*P_1_5 + A_2_3*P_2_5 + A_3_3*P_3_5);
        kp.Q_bar(4,4) += A_0_4*tmp_ric40 + A_1_4*tmp_ric41 + A_2_4*tmp_ric42 + A_4_4*tmp_ric43;
        kp.Q_bar(4,5) += A_0_5*tmp_ric40 + A_1_5*tmp_ric41 + A_2_5*tmp_ric42 + A_4_5*tmp_ric43 + A_5_5*(A_0_4*P_0_5 + A_1_4*P_1_5 + A_2_4*P_2_5 + A_4_4*P_4_5);
        kp.Q_bar(5,5) += A_0_5*(tmp_ric11 + tmp_ric12 + tmp_ric13 + tmp_ric14 + tmp_ric15) + A_1_5*(tmp_ric27 + tmp_ric28 + tmp_ric29 + tmp_ric30 + tmp_ric31) + A_2_5*(A_0_5*P_0_2 + A_1_5*P_1_2 + A_2_5*P_2_2 + A_4_5*P_2_4 + A_5_5*P_2_5) + A_4_5*(A_0_5*P_0_4 + A_1_5*P_1_4 + A_2_5*P_2_4 + A_4_5*P_4_4 + A_5_5*P_4_5) + A_5_5*(A_0_5*P_0_5 + A_1_5*P_1_5 + A_2_5*P_2_5 + A_4_5*P_4_5 + A_5_5*P_5_5);
        kp.R_bar(0,0) += B_0_0*tmp_ric44 + B_1_0*tmp_ric45 + B_2_0*tmp_ric46 + B_3_0*tmp_ric47;
        kp.R_bar(0,1) += B_0_1*tmp_ric44 + B_1_1*tmp_ric45 + B_2_1*tmp_ric46 + B_4_1*tmp_ric48 + B_5_1*tmp_ric49;
        kp.R_bar(1,1) += B_0_1*tmp_ric50 + B_1_1*tmp_ric51 + B_2_1*tmp_ric52 + B_4_1*tmp_ric53 + B_5_1*tmp_ric54;
        kp.H_bar(0,0) += A_0_0*tmp_ric44;
        kp.H_bar(0,1) += A_1_1*tmp_ric45;
        kp.H_bar(0,2) += A_0_2*tmp_ric44 + A_1_2*tmp_ric45 + A_2_2*tmp_ric46;
        kp.H_bar(0,3) += A_0_3*tmp_ric44 + A_1_3*tmp_ric45 + A_2_3*tmp_ric46 + A_3_3*tmp_ric47;
        kp.H_bar(0,4) += A_0_4*tmp_ric44 + A_1_4*tmp_ric45 + A_2_4*tmp_ric46 + A_4_4*tmp_ric48;
        kp.H_bar(0,5) += A_0_5*tmp_ric44 + A_1_5*tmp_ric45 + A_2_5*tmp_ric46 + A_4_5*tmp_ric48 + A_5_5*tmp_ric49;
        kp.H_bar(1,0) += A_0_0*tmp_ric50;
        kp.H_bar(1,1) += A_1_1*tmp_ric51;
        kp.H_bar(1,2) += A_0_2*tmp_ric50 + A_1_2*tmp_ric51 + A_2_2*tmp_ric52;
        kp.H_bar(1,3) += A_0_3*tmp_ric50 + A_1_3*tmp_ric51 + A_2_3*tmp_ric52 + A_3_3*(B_0_1*P_0_3 + B_1_1*P_1_3 + B_2_1*P_2_3 + B_4_1*P_3_4 + B_5_1*P_3_5);
        kp.H_bar(1,4) += A_0_4*tmp_ric50 + A_1_4*tmp_ric51 + A_2_4*tmp_ric52 + A_4_4*tmp_ric53;
        kp.H_bar(1,5) += A_0_5*tmp_ric50 + A_1_5*tmp_ric51 + A_2_5*tmp_ric52 + A_4_5*tmp_ric53 + A_5_5*tmp_ric54;
        kp.q_bar(0,0) += A_0_0*p_0;
        kp.q_bar(1,0) += A_1_1*p_1;
        kp.q_bar(2,0) += A_0_2*p_0 + A_1_2*p_1 + A_2_2*p_2;
        kp.q_bar(3,0) += A_0_3*p_0 + A_1_3*p_1 + A_2_3*p_2 + A_3_3*p_3;
        kp.q_bar(4,0) += A_0_4*p_0 + A_1_4*p_1 + A_2_4*p_2 + A_4_4*p_4;
        kp.q_bar(5,0) += A_0_5*p_0 + A_1_5*p_1 + A_2_5*p_2 + A_4_5*p_4 + A_5_5*p_5;
        kp.r_bar(0,0) += B_0_0*p_0 + B_1_0*p_1 + B_2_0*p_2 + B_3_0*p_3;
        kp.r_bar(1,0) += B_0_1*p_0 + B_1_1*p_1 + B_2_1*p_2 + B_4_1*p_4 + B_5_1*p_5;

        // Fill Lower Triangles (Symmetry)
        kp.Q_bar(1,0) = kp.Q_bar(0,1);
        kp.Q_bar(2,0) = kp.Q_bar(0,2);
        kp.Q_bar(3,0) = kp.Q_bar(0,3);
        kp.Q_bar(4,0) = kp.Q_bar(0,4);
        kp.Q_bar(5,0) = kp.Q_bar(0,5);
        kp.Q_bar(2,1) = kp.Q_bar(1,2);
        kp.Q_bar(3,1) = kp.Q_bar(1,3);
        kp.Q_bar(4,1) = kp.Q_bar(1,4);
        kp.Q_bar(5,1) = kp.Q_bar(1,5);
        kp.Q_bar(3,2) = kp.Q_bar(2,3);
        kp.Q_bar(4,2) = kp.Q_bar(2,4);
        kp.Q_bar(5,2) = kp.Q_bar(2,5);
        kp.Q_bar(4,3) = kp.Q_bar(3,4);
        kp.Q_bar(5,3) = kp.Q_bar(3,5);
        kp.Q_bar(5,4) = kp.Q_bar(4,5);
        kp.R_bar(1,0) = kp.R_bar(0,1);

    }
    
};
}
