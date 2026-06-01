#pragma once
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/matrix/matrix_defs.h"
#include "minisolver/integrator/numerical_jacobian.h"
#include <cstdint>
#include <cmath>
#include <string>
#include <array>
#include <stdexcept>

namespace minisolver {

struct DcolTwoCarsModel {
    // --- Constants ---
    static const int NX=8;
    static const int NU=4;
    static const int NC=1;
    static const int NP=42;

    static constexpr std::uint64_t model_fingerprint = 0xf56ffed6fb806caaull;

    static constexpr IntegratorType generated_integrator = IntegratorType::RUNGE_KUTTA_4;

    static constexpr std::array<bool, NC> constraint_has_l1 = {false};
    static constexpr std::array<bool, NC> constraint_has_l2 = {false};
    static constexpr bool any_l1_constraints = false;
    static constexpr bool any_l2_constraints = false;


    // --- Name Arrays (for Map Construction) ---
    static constexpr std::array<const char*, NX> state_names = {
        "x1",
        "y1",
        "theta1",
        "v1",
        "x2",
        "y2",
        "theta2",
        "v2",
    };

    static constexpr std::array<const char*, NU> control_names = {
        "a1",
        "omega1",
        "a2",
        "omega2",
    };

    static constexpr std::array<const char*, NP> param_names = {
        "x1_ref",
        "y1_ref",
        "theta1_ref",
        "v1_ref",
        "x2_ref",
        "y2_ref",
        "theta2_ref",
        "v2_ref",
        "x1_lin",
        "y1_lin",
        "theta1_lin",
        "x2_lin",
        "y2_lin",
        "theta2_lin",
        "dcol_alpha",
        "dcol_gx1",
        "dcol_gy1",
        "dcol_gtheta1",
        "dcol_gx2",
        "dcol_gy2",
        "dcol_gtheta2",
        "dcol_h00",
        "dcol_h01",
        "dcol_h02",
        "dcol_h03",
        "dcol_h04",
        "dcol_h05",
        "dcol_h11",
        "dcol_h12",
        "dcol_h13",
        "dcol_h14",
        "dcol_h15",
        "dcol_h22",
        "dcol_h23",
        "dcol_h24",
        "dcol_h25",
        "dcol_h33",
        "dcol_h34",
        "dcol_h35",
        "dcol_h44",
        "dcol_h45",
        "dcol_h55",
    };


    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in)
    {
        T theta1 = x_in(2);
        T v1 = x_in(3);
        T theta2 = x_in(6);
        T v2 = x_in(7);
        T a1 = u_in(0);
        T omega1 = u_in(1);
        T a2 = u_in(2);
        T omega2 = u_in(3);
        (void)p_in;

        MSVec<T, NX> xdot;
        xdot(0) = v1*cos(theta1);
        xdot(1) = v1*sin(theta1);
        xdot(2) = omega1;
        xdot(3) = a1;
        xdot(4) = v2*cos(theta2);
        xdot(5) = v2*sin(theta2);
        xdot(6) = omega2;
        xdot(7) = a2;
        return xdot;

    }

    // --- Continuous Dynamics Jacobians (for implicit integrators) ---
    template<typename T>
    static ContinuousJacobians<T, NX, NU> jacobian_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in)
    {
        T theta1 = x_in(2);
        T v1 = x_in(3);
        T theta2 = x_in(6);
        T v2 = x_in(7);
        (void)u_in;
        (void)p_in;

        ContinuousJacobians<T, NX, NU> jac;
        T tmp_jc0 = sin(theta1);
        T tmp_jc1 = cos(theta1);
        T tmp_jc2 = sin(theta2);
        T tmp_jc3 = cos(theta2);

        // Clear continuous Jacobian packets; nonzero entries are assigned below.
        jac.Jx.setZero();
        jac.Ju.setZero();

        // Jx = df/dx
        jac.Jx(0,2) = -tmp_jc0*v1;
        jac.Jx(0,3) = tmp_jc1;
        jac.Jx(1,2) = tmp_jc1*v1;
        jac.Jx(1,3) = tmp_jc0;
        jac.Jx(4,6) = -tmp_jc2*v2;
        jac.Jx(4,7) = tmp_jc3;
        jac.Jx(5,6) = tmp_jc3*v2;
        jac.Jx(5,7) = tmp_jc2;

        // Ju = df/du
        jac.Ju(2,1) = 1;
        jac.Ju(3,0) = 1;
        jac.Ju(6,3) = 1;
        jac.Ju(7,2) = 1;

        return jac;

    }

    // --- Integrator Interface ---
    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in,
        double dt,
        IntegratorType type)
    {
        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
                return x_in + dynamics_continuous(x_in, u_in, p_in) * dt;

            case IntegratorType::RUNGE_KUTTA_2:
            {
               auto k1 = dynamics_continuous(x_in, u_in, p_in);
               auto k2 = dynamics_continuous<T>(x_in + k1 * (0.5 * dt), u_in, p_in);
               return x_in + k2 * dt;
            }

            case IntegratorType::EULER_IMPLICIT:
            case IntegratorType::GAUSS_LEGENDRE_2:
            case IntegratorType::GAUSS_LEGENDRE_4:
                throw std::invalid_argument(
                    "Implicit integrators require minisolver::detail::dispatch_integrate");

            case IntegratorType::RUNGE_KUTTA_4:
            {
               auto k1 = dynamics_continuous(x_in, u_in, p_in);
               auto k2 = dynamics_continuous<T>(x_in + k1 * (0.5 * dt), u_in, p_in);
               auto k3 = dynamics_continuous<T>(x_in + k2 * (0.5 * dt), u_in, p_in);
               auto k4 = dynamics_continuous<T>(x_in + k3 * dt, u_in, p_in);
               return x_in + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }

            case IntegratorType::DISCRETE:
                throw std::invalid_argument("DISCRETE integrator requires Next(state) dynamics");
        }
        throw std::invalid_argument("Unsupported integrator type");

    }

    // --- 1. Compute Dynamics (f_resid, A, B) ---
    template<typename T>
    static void compute_dynamics(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        T x1 = kp.x(0);
        T y1 = kp.x(1);
        T theta1 = kp.x(2);
        T v1 = kp.x(3);
        T x2 = kp.x(4);
        T y2 = kp.x(5);
        T theta2 = kp.x(6);
        T v2 = kp.x(7);
        T a1 = kp.u(0);
        T omega1 = kp.u(1);
        T a2 = kp.u(2);
        T omega2 = kp.u(3);

        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
            {
                T tmp_d0 = dt*cos(theta1);
                T tmp_d1 = tmp_d0*v1;
                T tmp_d2 = dt*sin(theta1);
                T tmp_d3 = tmp_d2*v1;
                T tmp_d4 = dt*cos(theta2);
                T tmp_d5 = tmp_d4*v2;
                T tmp_d6 = dt*sin(theta2);
                T tmp_d7 = tmp_d6*v2;
                kp.f_resid(0) = tmp_d1 + x1;
                kp.f_resid(1) = tmp_d3 + y1;
                kp.f_resid(2) = dt*omega1 + theta1;
                kp.f_resid(3) = a1*dt + v1;
                kp.f_resid(4) = tmp_d5 + x2;
                kp.f_resid(5) = tmp_d7 + y2;
                kp.f_resid(6) = dt*omega2 + theta2;
                kp.f_resid(7) = a2*dt + v2;

                // Clear dynamics Jacobian A; nonzero entries are assigned below.
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(0,2) = -tmp_d3;
                kp.A(0,3) = tmp_d0;
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d1;
                kp.A(1,3) = tmp_d2;
                kp.A(2,2) = 1;
                kp.A(3,3) = 1;
                kp.A(4,4) = 1;
                kp.A(4,6) = -tmp_d7;
                kp.A(4,7) = tmp_d4;
                kp.A(5,5) = 1;
                kp.A(5,6) = tmp_d5;
                kp.A(5,7) = tmp_d6;
                kp.A(6,6) = 1;
                kp.A(7,7) = 1;

                // Clear dynamics Jacobian B; nonzero entries are assigned below.
                kp.B.setZero();
                kp.B(2,1) = dt;
                kp.B(3,0) = dt;
                kp.B(6,3) = dt;
                kp.B(7,2) = dt;
                return;
            }
            case IntegratorType::RUNGE_KUTTA_2:
            {
                T tmp_d0 = a1*dt;
                T tmp_d1 = 0.5*tmp_d0 + v1;
                T tmp_d2 = dt*omega1;
                T tmp_d3 = theta1 + 0.5*tmp_d2;
                T tmp_d4 = cos(tmp_d3);
                T tmp_d5 = dt*tmp_d4;
                T tmp_d6 = tmp_d1*tmp_d5;
                T tmp_d7 = sin(tmp_d3);
                T tmp_d8 = dt*tmp_d7;
                T tmp_d9 = tmp_d1*tmp_d8;
                T tmp_d10 = a2*dt;
                T tmp_d11 = 0.5*tmp_d10 + v2;
                T tmp_d12 = dt*omega2;
                T tmp_d13 = theta2 + 0.5*tmp_d12;
                T tmp_d14 = cos(tmp_d13);
                T tmp_d15 = dt*tmp_d14;
                T tmp_d16 = tmp_d11*tmp_d15;
                T tmp_d17 = sin(tmp_d13);
                T tmp_d18 = dt*tmp_d17;
                T tmp_d19 = tmp_d11*tmp_d18;
                T tmp_d20 = 0.5*pow(dt, 2);
                T tmp_d21 = tmp_d20*tmp_d4;
                T tmp_d22 = tmp_d20*tmp_d7;
                T tmp_d23 = tmp_d14*tmp_d20;
                T tmp_d24 = tmp_d17*tmp_d20;
                kp.f_resid(0) = tmp_d6 + x1;
                kp.f_resid(1) = tmp_d9 + y1;
                kp.f_resid(2) = theta1 + tmp_d2;
                kp.f_resid(3) = tmp_d0 + v1;
                kp.f_resid(4) = tmp_d16 + x2;
                kp.f_resid(5) = tmp_d19 + y2;
                kp.f_resid(6) = theta2 + tmp_d12;
                kp.f_resid(7) = tmp_d10 + v2;

                // Clear dynamics Jacobian A; nonzero entries are assigned below.
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(0,2) = -tmp_d9;
                kp.A(0,3) = tmp_d5;
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d6;
                kp.A(1,3) = tmp_d8;
                kp.A(2,2) = 1;
                kp.A(3,3) = 1;
                kp.A(4,4) = 1;
                kp.A(4,6) = -tmp_d19;
                kp.A(4,7) = tmp_d15;
                kp.A(5,5) = 1;
                kp.A(5,6) = tmp_d16;
                kp.A(5,7) = tmp_d18;
                kp.A(6,6) = 1;
                kp.A(7,7) = 1;

                // Clear dynamics Jacobian B; nonzero entries are assigned below.
                kp.B.setZero();
                kp.B(0,0) = tmp_d21;
                kp.B(0,1) = -tmp_d1*tmp_d22;
                kp.B(1,0) = tmp_d22;
                kp.B(1,1) = tmp_d1*tmp_d21;
                kp.B(2,1) = dt;
                kp.B(3,0) = dt;
                kp.B(4,2) = tmp_d23;
                kp.B(4,3) = -tmp_d11*tmp_d24;
                kp.B(5,2) = tmp_d24;
                kp.B(5,3) = tmp_d11*tmp_d23;
                kp.B(6,3) = dt;
                kp.B(7,2) = dt;
                return;
            }
            case IntegratorType::RUNGE_KUTTA_4:
            {
                T tmp_d0 = cos(theta1);
                T tmp_d1 = a1*dt;
                T tmp_d2 = tmp_d1 + v1;
                T tmp_d3 = dt*omega1;
                T tmp_d4 = theta1 + tmp_d3;
                T tmp_d5 = cos(tmp_d4);
                T tmp_d6 = tmp_d2*tmp_d5;
                T tmp_d7 = 0.5*tmp_d1 + v1;
                T tmp_d8 = theta1 + 0.5*tmp_d3;
                T tmp_d9 = cos(tmp_d8);
                T tmp_d10 = 4*tmp_d9;
                T tmp_d11 = 0.16666666666666666*dt;
                T tmp_d12 = tmp_d11*(tmp_d0*v1 + tmp_d10*tmp_d7 + tmp_d6);
                T tmp_d13 = sin(theta1);
                T tmp_d14 = sin(tmp_d4);
                T tmp_d15 = tmp_d14*tmp_d2;
                T tmp_d16 = sin(tmp_d8);
                T tmp_d17 = 4*tmp_d16;
                T tmp_d18 = tmp_d13*v1 + tmp_d15 + tmp_d17*tmp_d7;
                T tmp_d19 = cos(theta2);
                T tmp_d20 = a2*dt;
                T tmp_d21 = tmp_d20 + v2;
                T tmp_d22 = dt*omega2;
                T tmp_d23 = theta2 + tmp_d22;
                T tmp_d24 = cos(tmp_d23);
                T tmp_d25 = tmp_d21*tmp_d24;
                T tmp_d26 = 0.5*tmp_d20 + v2;
                T tmp_d27 = theta2 + 0.5*tmp_d22;
                T tmp_d28 = cos(tmp_d27);
                T tmp_d29 = 4*tmp_d28;
                T tmp_d30 = tmp_d11*(tmp_d19*v2 + tmp_d25 + tmp_d26*tmp_d29);
                T tmp_d31 = sin(theta2);
                T tmp_d32 = sin(tmp_d23);
                T tmp_d33 = tmp_d21*tmp_d32;
                T tmp_d34 = sin(tmp_d27);
                T tmp_d35 = 4*tmp_d34;
                T tmp_d36 = tmp_d26*tmp_d35 + tmp_d31*v2 + tmp_d33;
                T tmp_d37 = 2.0*dt;
                T tmp_d38 = tmp_d37*tmp_d9;
                T tmp_d39 = tmp_d16*tmp_d37;
                T tmp_d40 = 1.0*dt;
                T tmp_d41 = tmp_d28*tmp_d37;
                T tmp_d42 = tmp_d34*tmp_d37;
                kp.f_resid(0) = tmp_d12 + x1;
                kp.f_resid(1) = tmp_d11*tmp_d18 + y1;
                kp.f_resid(2) = theta1 + 1.0*tmp_d3;
                kp.f_resid(3) = 1.0*tmp_d1 + v1;
                kp.f_resid(4) = tmp_d30 + x2;
                kp.f_resid(5) = tmp_d11*tmp_d36 + y2;
                kp.f_resid(6) = theta2 + 1.0*tmp_d22;
                kp.f_resid(7) = 1.0*tmp_d20 + v2;

                // Clear dynamics Jacobian A; nonzero entries are assigned below.
                kp.A.setZero();
                kp.A(0,0) = 1;
                kp.A(0,2) = -tmp_d11*tmp_d18;
                kp.A(0,3) = tmp_d11*(tmp_d0 + tmp_d10 + tmp_d5);
                kp.A(1,1) = 1;
                kp.A(1,2) = tmp_d12;
                kp.A(1,3) = tmp_d11*(tmp_d13 + tmp_d14 + tmp_d17);
                kp.A(2,2) = 1;
                kp.A(3,3) = 1;
                kp.A(4,4) = 1;
                kp.A(4,6) = -tmp_d11*tmp_d36;
                kp.A(4,7) = tmp_d11*(tmp_d19 + tmp_d24 + tmp_d29);
                kp.A(5,5) = 1;
                kp.A(5,6) = tmp_d30;
                kp.A(5,7) = tmp_d11*(tmp_d31 + tmp_d32 + tmp_d35);
                kp.A(6,6) = 1;
                kp.A(7,7) = 1;

                // Clear dynamics Jacobian B; nonzero entries are assigned below.
                kp.B.setZero();
                kp.B(0,0) = tmp_d11*(dt*tmp_d5 + tmp_d38);
                kp.B(0,1) = tmp_d11*(-dt*tmp_d15 - tmp_d39*tmp_d7);
                kp.B(1,0) = tmp_d11*(dt*tmp_d14 + tmp_d39);
                kp.B(1,1) = tmp_d11*(dt*tmp_d6 + tmp_d38*tmp_d7);
                kp.B(2,1) = tmp_d40;
                kp.B(3,0) = tmp_d40;
                kp.B(4,2) = tmp_d11*(dt*tmp_d24 + tmp_d41);
                kp.B(4,3) = tmp_d11*(-dt*tmp_d33 - tmp_d26*tmp_d42);
                kp.B(5,2) = tmp_d11*(dt*tmp_d32 + tmp_d42);
                kp.B(5,3) = tmp_d11*(dt*tmp_d25 + tmp_d26*tmp_d41);
                kp.B(6,3) = tmp_d40;
                kp.B(7,2) = tmp_d40;
                return;
            }
            case IntegratorType::EULER_IMPLICIT:
            case IntegratorType::GAUSS_LEGENDRE_2:
            case IntegratorType::GAUSS_LEGENDRE_4:
                throw std::invalid_argument("Implicit integrators require minisolver::detail::dispatch_compute_dynamics");
            case IntegratorType::DISCRETE:
                throw std::invalid_argument("DISCRETE integrator requires Next(state) dynamics");
        }
        throw std::invalid_argument("Unsupported integrator type");
    }

    // --- 1.5 Update Soft Constraint Weights ---
    template<typename T>
    static void update_soft_constraint_weights(KnotPoint<T,NX,NU,NC,NP>& kp) {
        (void)kp;
    }


    // --- 2. Compute QP/IPM Constraints (g_val, C, D) ---
    template<typename T>
    static void compute_qp_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x1 = kp.x(0);
        T y1 = kp.x(1);
        T theta1 = kp.x(2);
        T x2 = kp.x(4);
        T y2 = kp.x(5);
        T theta2 = kp.x(6);
        T x1_lin = kp.p(8);
        T y1_lin = kp.p(9);
        T theta1_lin = kp.p(10);
        T x2_lin = kp.p(11);
        T y2_lin = kp.p(12);
        T theta2_lin = kp.p(13);
        T dcol_alpha = kp.p(14);
        T dcol_gx1 = kp.p(15);
        T dcol_gy1 = kp.p(16);
        T dcol_gtheta1 = kp.p(17);
        T dcol_gx2 = kp.p(18);
        T dcol_gy2 = kp.p(19);
        T dcol_gtheta2 = kp.p(20);
        T dcol_h00 = kp.p(21);
        T dcol_h01 = kp.p(22);
        T dcol_h02 = kp.p(23);
        T dcol_h03 = kp.p(24);
        T dcol_h04 = kp.p(25);
        T dcol_h05 = kp.p(26);
        T dcol_h11 = kp.p(27);
        T dcol_h12 = kp.p(28);
        T dcol_h13 = kp.p(29);
        T dcol_h14 = kp.p(30);
        T dcol_h15 = kp.p(31);
        T dcol_h22 = kp.p(32);
        T dcol_h23 = kp.p(33);
        T dcol_h24 = kp.p(34);
        T dcol_h25 = kp.p(35);
        T dcol_h33 = kp.p(36);
        T dcol_h34 = kp.p(37);
        T dcol_h35 = kp.p(38);
        T dcol_h44 = kp.p(39);
        T dcol_h45 = kp.p(40);
        T dcol_h55 = kp.p(41);

        // --- Special Constraints Pre-Calculation ---

        T tmp_c0 = theta1 - theta1_lin;
        T tmp_c1 = theta2 - theta2_lin;
        T tmp_c2 = x1 - x1_lin;
        T tmp_c3 = x2 - x2_lin;
        T tmp_c4 = y1 - y1_lin;
        T tmp_c5 = y2 - y2_lin;
        T tmp_c6 = (1.0/2.0)*dcol_h00;
        T tmp_c7 = (1.0/2.0)*dcol_h11;
        T tmp_c8 = (1.0/2.0)*dcol_h22;
        T tmp_c9 = (1.0/2.0)*dcol_h33;
        T tmp_c10 = (1.0/2.0)*dcol_h44;
        T tmp_c11 = (1.0/2.0)*dcol_h55;
        T tmp_c12 = dcol_h01*tmp_c4;
        T tmp_c13 = dcol_h02*tmp_c0;
        T tmp_c14 = dcol_h03*tmp_c3;
        T tmp_c15 = dcol_h04*tmp_c5;
        T tmp_c16 = dcol_h05*tmp_c1;
        T tmp_c17 = dcol_h12*tmp_c0;
        T tmp_c18 = dcol_h13*tmp_c3;
        T tmp_c19 = dcol_h14*tmp_c5;
        T tmp_c20 = dcol_h15*tmp_c1;
        T tmp_c21 = dcol_h23*tmp_c3;
        T tmp_c22 = dcol_h24*tmp_c5;
        T tmp_c23 = dcol_h25*tmp_c1;
        T tmp_c24 = dcol_h34*tmp_c5;
        T tmp_c25 = dcol_h35*tmp_c1;
        T tmp_c26 = dcol_h45*tmp_c1;

        // Clear generated output packets; nonzero entries are assigned below.
        kp.C.setZero();
        kp.D.setZero();

        // g_val
        kp.g_val(0,0) = -dcol_alpha - dcol_gtheta1*tmp_c0 - dcol_gtheta2*tmp_c1 - dcol_gx1*tmp_c2 - dcol_gx2*tmp_c3 - dcol_gy1*tmp_c4 - dcol_gy2*tmp_c5 - pow(tmp_c0, 2)*tmp_c8 - tmp_c0*tmp_c21 - tmp_c0*tmp_c22 - tmp_c0*tmp_c23 - pow(tmp_c1, 2)*tmp_c11 - tmp_c10*pow(tmp_c5, 2) - tmp_c12*tmp_c2 - tmp_c13*tmp_c2 - tmp_c14*tmp_c2 - tmp_c15*tmp_c2 - tmp_c16*tmp_c2 - tmp_c17*tmp_c4 - tmp_c18*tmp_c4 - tmp_c19*tmp_c4 - pow(tmp_c2, 2)*tmp_c6 - tmp_c20*tmp_c4 - tmp_c24*tmp_c3 - tmp_c25*tmp_c3 - tmp_c26*tmp_c5 - pow(tmp_c3, 2)*tmp_c9 - pow(tmp_c4, 2)*tmp_c7 + 1.0;

        // C
        kp.C(0,0) = -dcol_gx1 - tmp_c12 - tmp_c13 - tmp_c14 - tmp_c15 - tmp_c16 - tmp_c6*(2*x1 - 2*x1_lin);
        kp.C(0,1) = -dcol_gy1 - dcol_h01*tmp_c2 - tmp_c17 - tmp_c18 - tmp_c19 - tmp_c20 - tmp_c7*(2*y1 - 2*y1_lin);
        kp.C(0,2) = -dcol_gtheta1 - dcol_h02*tmp_c2 - dcol_h12*tmp_c4 - tmp_c21 - tmp_c22 - tmp_c23 - tmp_c8*(2*theta1 - 2*theta1_lin);
        kp.C(0,4) = -dcol_gx2 - dcol_h03*tmp_c2 - dcol_h13*tmp_c4 - dcol_h23*tmp_c0 - tmp_c24 - tmp_c25 - tmp_c9*(2*x2 - 2*x2_lin);
        kp.C(0,5) = -dcol_gy2 - dcol_h04*tmp_c2 - dcol_h14*tmp_c4 - dcol_h24*tmp_c0 - dcol_h34*tmp_c3 - tmp_c10*(2*y2 - 2*y2_lin) - tmp_c26;
        kp.C(0,6) = -dcol_gtheta2 - dcol_h05*tmp_c2 - dcol_h15*tmp_c4 - dcol_h25*tmp_c0 - dcol_h35*tmp_c3 - dcol_h45*tmp_c5 - tmp_c11*(2*theta2 - 2*theta2_lin);

        // D

    }

    // Legacy alias for hand-written code that still calls compute_constraints().
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_qp_constraints(kp);
    }

    // --- 2.1 Compute True Constraints (g_true) ---
    template<typename T>
    static void compute_true_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x1 = kp.x(0);
        T y1 = kp.x(1);
        T theta1 = kp.x(2);
        T x2 = kp.x(4);
        T y2 = kp.x(5);
        T theta2 = kp.x(6);
        T x1_lin = kp.p(8);
        T y1_lin = kp.p(9);
        T theta1_lin = kp.p(10);
        T x2_lin = kp.p(11);
        T y2_lin = kp.p(12);
        T theta2_lin = kp.p(13);
        T dcol_alpha = kp.p(14);
        T dcol_gx1 = kp.p(15);
        T dcol_gy1 = kp.p(16);
        T dcol_gtheta1 = kp.p(17);
        T dcol_gx2 = kp.p(18);
        T dcol_gy2 = kp.p(19);
        T dcol_gtheta2 = kp.p(20);
        T dcol_h00 = kp.p(21);
        T dcol_h01 = kp.p(22);
        T dcol_h02 = kp.p(23);
        T dcol_h03 = kp.p(24);
        T dcol_h04 = kp.p(25);
        T dcol_h05 = kp.p(26);
        T dcol_h11 = kp.p(27);
        T dcol_h12 = kp.p(28);
        T dcol_h13 = kp.p(29);
        T dcol_h14 = kp.p(30);
        T dcol_h15 = kp.p(31);
        T dcol_h22 = kp.p(32);
        T dcol_h23 = kp.p(33);
        T dcol_h24 = kp.p(34);
        T dcol_h25 = kp.p(35);
        T dcol_h33 = kp.p(36);
        T dcol_h34 = kp.p(37);
        T dcol_h35 = kp.p(38);
        T dcol_h44 = kp.p(39);
        T dcol_h45 = kp.p(40);
        T dcol_h55 = kp.p(41);


        // g_true
        kp.g_true(0,0) = -dcol_alpha - dcol_gtheta1*(theta1 - theta1_lin) - dcol_gtheta2*(theta2 - theta2_lin) - dcol_gx1*(x1 - x1_lin) - dcol_gx2*(x2 - x2_lin) - dcol_gy1*(y1 - y1_lin) - dcol_gy2*(y2 - y2_lin) - 1.0/2.0*dcol_h00*pow(x1 - x1_lin, 2) - dcol_h01*(x1 - x1_lin)*(y1 - y1_lin) - dcol_h02*(theta1 - theta1_lin)*(x1 - x1_lin) - dcol_h03*(x1 - x1_lin)*(x2 - x2_lin) - dcol_h04*(x1 - x1_lin)*(y2 - y2_lin) - dcol_h05*(theta2 - theta2_lin)*(x1 - x1_lin) - 1.0/2.0*dcol_h11*pow(y1 - y1_lin, 2) - dcol_h12*(theta1 - theta1_lin)*(y1 - y1_lin) - dcol_h13*(x2 - x2_lin)*(y1 - y1_lin) - dcol_h14*(y1 - y1_lin)*(y2 - y2_lin) - dcol_h15*(theta2 - theta2_lin)*(y1 - y1_lin) - 1.0/2.0*dcol_h22*pow(theta1 - theta1_lin, 2) - dcol_h23*(theta1 - theta1_lin)*(x2 - x2_lin) - dcol_h24*(theta1 - theta1_lin)*(y2 - y2_lin) - dcol_h25*(theta1 - theta1_lin)*(theta2 - theta2_lin) - 1.0/2.0*dcol_h33*pow(x2 - x2_lin, 2) - dcol_h34*(x2 - x2_lin)*(y2 - y2_lin) - dcol_h35*(theta2 - theta2_lin)*(x2 - x2_lin) - 1.0/2.0*dcol_h44*pow(y2 - y2_lin, 2) - dcol_h45*(theta2 - theta2_lin)*(y2 - y2_lin) - 1.0/2.0*dcol_h55*pow(theta2 - theta2_lin, 2) + 1.0;

    }

    // --- 2.5 Terminal Stage: x-only projection of QP/IPM constraints ---
    template<typename T>
    static void compute_terminal_qp_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x1 = kp.x(0);
        T y1 = kp.x(1);
        T theta1 = kp.x(2);
        T x2 = kp.x(4);
        T y2 = kp.x(5);
        T theta2 = kp.x(6);
        T x1_lin = kp.p(8);
        T y1_lin = kp.p(9);
        T theta1_lin = kp.p(10);
        T x2_lin = kp.p(11);
        T y2_lin = kp.p(12);
        T theta2_lin = kp.p(13);
        T dcol_alpha = kp.p(14);
        T dcol_gx1 = kp.p(15);
        T dcol_gy1 = kp.p(16);
        T dcol_gtheta1 = kp.p(17);
        T dcol_gx2 = kp.p(18);
        T dcol_gy2 = kp.p(19);
        T dcol_gtheta2 = kp.p(20);
        T dcol_h00 = kp.p(21);
        T dcol_h01 = kp.p(22);
        T dcol_h02 = kp.p(23);
        T dcol_h03 = kp.p(24);
        T dcol_h04 = kp.p(25);
        T dcol_h05 = kp.p(26);
        T dcol_h11 = kp.p(27);
        T dcol_h12 = kp.p(28);
        T dcol_h13 = kp.p(29);
        T dcol_h14 = kp.p(30);
        T dcol_h15 = kp.p(31);
        T dcol_h22 = kp.p(32);
        T dcol_h23 = kp.p(33);
        T dcol_h24 = kp.p(34);
        T dcol_h25 = kp.p(35);
        T dcol_h33 = kp.p(36);
        T dcol_h34 = kp.p(37);
        T dcol_h35 = kp.p(38);
        T dcol_h44 = kp.p(39);
        T dcol_h45 = kp.p(40);
        T dcol_h55 = kp.p(41);

        // --- Special Constraints Pre-Calculation ---


        // Clear generated output packets; nonzero entries are assigned below.
        kp.C.setZero();
        kp.D.setZero();

        // g_val
        kp.g_val(0,0) = -dcol_alpha - dcol_gtheta1*(theta1 - theta1_lin) - dcol_gtheta2*(theta2 - theta2_lin) - dcol_gx1*(x1 - x1_lin) - dcol_gx2*(x2 - x2_lin) - dcol_gy1*(y1 - y1_lin) - dcol_gy2*(y2 - y2_lin) - 1.0/2.0*dcol_h00*pow(x1 - x1_lin, 2) - dcol_h01*(x1 - x1_lin)*(y1 - y1_lin) - dcol_h02*(theta1 - theta1_lin)*(x1 - x1_lin) - dcol_h03*(x1 - x1_lin)*(x2 - x2_lin) - dcol_h04*(x1 - x1_lin)*(y2 - y2_lin) - dcol_h05*(theta2 - theta2_lin)*(x1 - x1_lin) - 1.0/2.0*dcol_h11*pow(y1 - y1_lin, 2) - dcol_h12*(theta1 - theta1_lin)*(y1 - y1_lin) - dcol_h13*(x2 - x2_lin)*(y1 - y1_lin) - dcol_h14*(y1 - y1_lin)*(y2 - y2_lin) - dcol_h15*(theta2 - theta2_lin)*(y1 - y1_lin) - 1.0/2.0*dcol_h22*pow(theta1 - theta1_lin, 2) - dcol_h23*(theta1 - theta1_lin)*(x2 - x2_lin) - dcol_h24*(theta1 - theta1_lin)*(y2 - y2_lin) - dcol_h25*(theta1 - theta1_lin)*(theta2 - theta2_lin) - 1.0/2.0*dcol_h33*pow(x2 - x2_lin, 2) - dcol_h34*(x2 - x2_lin)*(y2 - y2_lin) - dcol_h35*(theta2 - theta2_lin)*(x2 - x2_lin) - 1.0/2.0*dcol_h44*pow(y2 - y2_lin, 2) - dcol_h45*(theta2 - theta2_lin)*(y2 - y2_lin) - 1.0/2.0*dcol_h55*pow(theta2 - theta2_lin, 2) + 1.0;

        // C
        kp.C(0,0) = -dcol_gx1 - 1.0/2.0*dcol_h00*(2*x1 - 2*x1_lin) - dcol_h01*(y1 - y1_lin) - dcol_h02*(theta1 - theta1_lin) - dcol_h03*(x2 - x2_lin) - dcol_h04*(y2 - y2_lin) - dcol_h05*(theta2 - theta2_lin);
        kp.C(0,1) = -dcol_gy1 - dcol_h01*(x1 - x1_lin) - 1.0/2.0*dcol_h11*(2*y1 - 2*y1_lin) - dcol_h12*(theta1 - theta1_lin) - dcol_h13*(x2 - x2_lin) - dcol_h14*(y2 - y2_lin) - dcol_h15*(theta2 - theta2_lin);
        kp.C(0,2) = -dcol_gtheta1 - dcol_h02*(x1 - x1_lin) - dcol_h12*(y1 - y1_lin) - 1.0/2.0*dcol_h22*(2*theta1 - 2*theta1_lin) - dcol_h23*(x2 - x2_lin) - dcol_h24*(y2 - y2_lin) - dcol_h25*(theta2 - theta2_lin);
        kp.C(0,4) = -dcol_gx2 - dcol_h03*(x1 - x1_lin) - dcol_h13*(y1 - y1_lin) - dcol_h23*(theta1 - theta1_lin) - 1.0/2.0*dcol_h33*(2*x2 - 2*x2_lin) - dcol_h34*(y2 - y2_lin) - dcol_h35*(theta2 - theta2_lin);
        kp.C(0,5) = -dcol_gy2 - dcol_h04*(x1 - x1_lin) - dcol_h14*(y1 - y1_lin) - dcol_h24*(theta1 - theta1_lin) - dcol_h34*(x2 - x2_lin) - 1.0/2.0*dcol_h44*(2*y2 - 2*y2_lin) - dcol_h45*(theta2 - theta2_lin);
        kp.C(0,6) = -dcol_gtheta2 - dcol_h05*(x1 - x1_lin) - dcol_h15*(y1 - y1_lin) - dcol_h25*(theta1 - theta1_lin) - dcol_h35*(x2 - x2_lin) - dcol_h45*(y2 - y2_lin) - 1.0/2.0*dcol_h55*(2*theta2 - 2*theta2_lin);

        // D

    }

    // Legacy alias for hand-written code that still calls compute_terminal_constraints().
    template<typename T>
    static void compute_terminal_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_terminal_qp_constraints(kp);
    }

    // --- 2.5.1 Terminal Stage: true x-only constraint residuals ---
    template<typename T>
    static void compute_terminal_true_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x1 = kp.x(0);
        T y1 = kp.x(1);
        T theta1 = kp.x(2);
        T x2 = kp.x(4);
        T y2 = kp.x(5);
        T theta2 = kp.x(6);
        T x1_lin = kp.p(8);
        T y1_lin = kp.p(9);
        T theta1_lin = kp.p(10);
        T x2_lin = kp.p(11);
        T y2_lin = kp.p(12);
        T theta2_lin = kp.p(13);
        T dcol_alpha = kp.p(14);
        T dcol_gx1 = kp.p(15);
        T dcol_gy1 = kp.p(16);
        T dcol_gtheta1 = kp.p(17);
        T dcol_gx2 = kp.p(18);
        T dcol_gy2 = kp.p(19);
        T dcol_gtheta2 = kp.p(20);
        T dcol_h00 = kp.p(21);
        T dcol_h01 = kp.p(22);
        T dcol_h02 = kp.p(23);
        T dcol_h03 = kp.p(24);
        T dcol_h04 = kp.p(25);
        T dcol_h05 = kp.p(26);
        T dcol_h11 = kp.p(27);
        T dcol_h12 = kp.p(28);
        T dcol_h13 = kp.p(29);
        T dcol_h14 = kp.p(30);
        T dcol_h15 = kp.p(31);
        T dcol_h22 = kp.p(32);
        T dcol_h23 = kp.p(33);
        T dcol_h24 = kp.p(34);
        T dcol_h25 = kp.p(35);
        T dcol_h33 = kp.p(36);
        T dcol_h34 = kp.p(37);
        T dcol_h35 = kp.p(38);
        T dcol_h44 = kp.p(39);
        T dcol_h45 = kp.p(40);
        T dcol_h55 = kp.p(41);


        // g_true
        kp.g_true(0,0) = -dcol_alpha - dcol_gtheta1*(theta1 - theta1_lin) - dcol_gtheta2*(theta2 - theta2_lin) - dcol_gx1*(x1 - x1_lin) - dcol_gx2*(x2 - x2_lin) - dcol_gy1*(y1 - y1_lin) - dcol_gy2*(y2 - y2_lin) - 1.0/2.0*dcol_h00*pow(x1 - x1_lin, 2) - dcol_h01*(x1 - x1_lin)*(y1 - y1_lin) - dcol_h02*(theta1 - theta1_lin)*(x1 - x1_lin) - dcol_h03*(x1 - x1_lin)*(x2 - x2_lin) - dcol_h04*(x1 - x1_lin)*(y2 - y2_lin) - dcol_h05*(theta2 - theta2_lin)*(x1 - x1_lin) - 1.0/2.0*dcol_h11*pow(y1 - y1_lin, 2) - dcol_h12*(theta1 - theta1_lin)*(y1 - y1_lin) - dcol_h13*(x2 - x2_lin)*(y1 - y1_lin) - dcol_h14*(y1 - y1_lin)*(y2 - y2_lin) - dcol_h15*(theta2 - theta2_lin)*(y1 - y1_lin) - 1.0/2.0*dcol_h22*pow(theta1 - theta1_lin, 2) - dcol_h23*(theta1 - theta1_lin)*(x2 - x2_lin) - dcol_h24*(theta1 - theta1_lin)*(y2 - y2_lin) - dcol_h25*(theta1 - theta1_lin)*(theta2 - theta2_lin) - 1.0/2.0*dcol_h33*pow(x2 - x2_lin, 2) - dcol_h34*(x2 - x2_lin)*(y2 - y2_lin) - dcol_h35*(theta2 - theta2_lin)*(x2 - x2_lin) - 1.0/2.0*dcol_h44*pow(y2 - y2_lin, 2) - dcol_h45*(theta2 - theta2_lin)*(y2 - y2_lin) - 1.0/2.0*dcol_h55*pow(theta2 - theta2_lin, 2) + 1.0;

    }

    // --- 2.6 SOC correction constraints ---
    template<typename T>
    static void compute_soc_constraints(
        const KnotPoint<T,NX,NU,NC,NP>& active_kp,
        KnotPoint<T,NX,NU,NC,NP>& trial_kp) {
        compute_qp_constraints(trial_kp);
        compute_true_constraints(trial_kp);
        (void)active_kp;

    }

    // --- 3. Compute Cost (Implemented via template for Exact/GN) ---
    template<typename T, int Mode>
    static void compute_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x1 = kp.x(0);
        T y1 = kp.x(1);
        T theta1 = kp.x(2);
        T v1 = kp.x(3);
        T x2 = kp.x(4);
        T y2 = kp.x(5);
        T theta2 = kp.x(6);
        T v2 = kp.x(7);
        T a1 = kp.u(0);
        T omega1 = kp.u(1);
        T a2 = kp.u(2);
        T omega2 = kp.u(3);
        T x1_ref = kp.p(0);
        T y1_ref = kp.p(1);
        T theta1_ref = kp.p(2);
        T v1_ref = kp.p(3);
        T x2_ref = kp.p(4);
        T y2_ref = kp.p(5);
        T theta2_ref = kp.p(6);
        T v2_ref = kp.p(7);
        T dcol_h00 = kp.p(21);
        T dcol_h01 = kp.p(22);
        T dcol_h02 = kp.p(23);
        T dcol_h03 = kp.p(24);
        T dcol_h04 = kp.p(25);
        T dcol_h05 = kp.p(26);
        T dcol_h11 = kp.p(27);
        T dcol_h12 = kp.p(28);
        T dcol_h13 = kp.p(29);
        T dcol_h14 = kp.p(30);
        T dcol_h15 = kp.p(31);
        T dcol_h22 = kp.p(32);
        T dcol_h23 = kp.p(33);
        T dcol_h24 = kp.p(34);
        T dcol_h25 = kp.p(35);
        T dcol_h33 = kp.p(36);
        T dcol_h34 = kp.p(37);
        T dcol_h35 = kp.p(38);
        T dcol_h44 = kp.p(39);
        T dcol_h45 = kp.p(40);
        T dcol_h55 = kp.p(41);
        T lam_0 = kp.lam(0);

        T tmp_j0 = -dcol_h01*lam_0;
        T tmp_j1 = -dcol_h02*lam_0;
        T tmp_j2 = -dcol_h03*lam_0;
        T tmp_j3 = -dcol_h04*lam_0;
        T tmp_j4 = -dcol_h05*lam_0;
        T tmp_j5 = -dcol_h12*lam_0;
        T tmp_j6 = -dcol_h13*lam_0;
        T tmp_j7 = -dcol_h14*lam_0;
        T tmp_j8 = -dcol_h15*lam_0;
        T tmp_j9 = -dcol_h23*lam_0;
        T tmp_j10 = -dcol_h24*lam_0;
        T tmp_j11 = -dcol_h25*lam_0;
        T tmp_j12 = -dcol_h34*lam_0;
        T tmp_j13 = -dcol_h35*lam_0;
        T tmp_j14 = -dcol_h45*lam_0;

        // q
        kp.q(0,0) = 0.16*x1 - 0.16*x1_ref;
        kp.q(1,0) = 0.0040000000000000001*y1 - 0.0040000000000000001*y1_ref;
        kp.q(2,0) = 0.90000000000000002*theta1 - 0.90000000000000002*theta1_ref;
        kp.q(3,0) = 0.59999999999999998*v1 - 0.59999999999999998*v1_ref;
        kp.q(4,0) = 0.16*x2 - 0.16*x2_ref;
        kp.q(5,0) = 0.0040000000000000001*y2 - 0.0040000000000000001*y2_ref;
        kp.q(6,0) = 0.90000000000000002*theta2 - 0.90000000000000002*theta2_ref;
        kp.q(7,0) = 0.59999999999999998*v2 - 0.59999999999999998*v2_ref;

        // r
        kp.r(0,0) = 0.080000000000000002*a1;
        kp.r(1,0) = 0.12*omega1;
        kp.r(2,0) = 0.080000000000000002*a2;
        kp.r(3,0) = 0.12*omega2;

        // Clear Hessian packets; nonzero entries are assigned below.
        kp.Q.setZero();
        kp.R.setZero();
        kp.H.setZero();

        // Q (Mode 0=GN, 1=Exact)
        kp.Q(0,0) = 0.16;
        kp.Q(1,1) = 0.0040000000000000001;
        kp.Q(2,2) = 0.90000000000000002;
        kp.Q(3,3) = 0.59999999999999998;
        kp.Q(4,4) = 0.16;
        kp.Q(5,5) = 0.0040000000000000001;
        kp.Q(6,6) = 0.90000000000000002;
        kp.Q(7,7) = 0.59999999999999998;
        if constexpr (Mode == 1) kp.Q(0,0) += -dcol_h00*lam_0;
        if constexpr (Mode == 1) kp.Q(0,1) += tmp_j0;
        if constexpr (Mode == 1) kp.Q(0,2) += tmp_j1;
        if constexpr (Mode == 1) kp.Q(0,4) += tmp_j2;
        if constexpr (Mode == 1) kp.Q(0,5) += tmp_j3;
        if constexpr (Mode == 1) kp.Q(0,6) += tmp_j4;
        if constexpr (Mode == 1) kp.Q(1,0) += tmp_j0;
        if constexpr (Mode == 1) kp.Q(1,1) += -dcol_h11*lam_0;
        if constexpr (Mode == 1) kp.Q(1,2) += tmp_j5;
        if constexpr (Mode == 1) kp.Q(1,4) += tmp_j6;
        if constexpr (Mode == 1) kp.Q(1,5) += tmp_j7;
        if constexpr (Mode == 1) kp.Q(1,6) += tmp_j8;
        if constexpr (Mode == 1) kp.Q(2,0) += tmp_j1;
        if constexpr (Mode == 1) kp.Q(2,1) += tmp_j5;
        if constexpr (Mode == 1) kp.Q(2,2) += -dcol_h22*lam_0;
        if constexpr (Mode == 1) kp.Q(2,4) += tmp_j9;
        if constexpr (Mode == 1) kp.Q(2,5) += tmp_j10;
        if constexpr (Mode == 1) kp.Q(2,6) += tmp_j11;
        if constexpr (Mode == 1) kp.Q(4,0) += tmp_j2;
        if constexpr (Mode == 1) kp.Q(4,1) += tmp_j6;
        if constexpr (Mode == 1) kp.Q(4,2) += tmp_j9;
        if constexpr (Mode == 1) kp.Q(4,4) += -dcol_h33*lam_0;
        if constexpr (Mode == 1) kp.Q(4,5) += tmp_j12;
        if constexpr (Mode == 1) kp.Q(4,6) += tmp_j13;
        if constexpr (Mode == 1) kp.Q(5,0) += tmp_j3;
        if constexpr (Mode == 1) kp.Q(5,1) += tmp_j7;
        if constexpr (Mode == 1) kp.Q(5,2) += tmp_j10;
        if constexpr (Mode == 1) kp.Q(5,4) += tmp_j12;
        if constexpr (Mode == 1) kp.Q(5,5) += -dcol_h44*lam_0;
        if constexpr (Mode == 1) kp.Q(5,6) += tmp_j14;
        if constexpr (Mode == 1) kp.Q(6,0) += tmp_j4;
        if constexpr (Mode == 1) kp.Q(6,1) += tmp_j8;
        if constexpr (Mode == 1) kp.Q(6,2) += tmp_j11;
        if constexpr (Mode == 1) kp.Q(6,4) += tmp_j13;
        if constexpr (Mode == 1) kp.Q(6,5) += tmp_j14;
        if constexpr (Mode == 1) kp.Q(6,6) += -dcol_h55*lam_0;

        // R (Mode 0=GN, 1=Exact)
        kp.R(0,0) = 0.080000000000000002;
        kp.R(1,1) = 0.12;
        kp.R(2,2) = 0.080000000000000002;
        kp.R(3,3) = 0.12;

        // H (Mode 0=GN, 1=Exact)

        kp.cost = 0.040000000000000001*pow(a1, 2) + 0.040000000000000001*pow(a2, 2) + 0.059999999999999998*pow(omega1, 2) + 0.059999999999999998*pow(omega2, 2) + 0.45000000000000001*pow(theta1 - theta1_ref, 2) + 0.45000000000000001*pow(theta2 - theta2_ref, 2) + 0.29999999999999999*pow(v1 - v1_ref, 2) + 0.29999999999999999*pow(v2 - v2_ref, 2) + 0.080000000000000002*pow(x1 - x1_ref, 2) + 0.080000000000000002*pow(x2 - x2_ref, 2) + 0.002*pow(y1 - y1_ref, 2) + 0.002*pow(y2 - y2_ref, 2);
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


    // --- 3.5 Terminal Cost (u projected to zero) ---
    template<typename T, int Mode>
    static void compute_terminal_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T x1 = kp.x(0);
        T y1 = kp.x(1);
        T theta1 = kp.x(2);
        T v1 = kp.x(3);
        T x2 = kp.x(4);
        T y2 = kp.x(5);
        T theta2 = kp.x(6);
        T v2 = kp.x(7);
        T x1_ref = kp.p(0);
        T y1_ref = kp.p(1);
        T theta1_ref = kp.p(2);
        T v1_ref = kp.p(3);
        T x2_ref = kp.p(4);
        T y2_ref = kp.p(5);
        T theta2_ref = kp.p(6);
        T v2_ref = kp.p(7);
        T dcol_h00 = kp.p(21);
        T dcol_h01 = kp.p(22);
        T dcol_h02 = kp.p(23);
        T dcol_h03 = kp.p(24);
        T dcol_h04 = kp.p(25);
        T dcol_h05 = kp.p(26);
        T dcol_h11 = kp.p(27);
        T dcol_h12 = kp.p(28);
        T dcol_h13 = kp.p(29);
        T dcol_h14 = kp.p(30);
        T dcol_h15 = kp.p(31);
        T dcol_h22 = kp.p(32);
        T dcol_h23 = kp.p(33);
        T dcol_h24 = kp.p(34);
        T dcol_h25 = kp.p(35);
        T dcol_h33 = kp.p(36);
        T dcol_h34 = kp.p(37);
        T dcol_h35 = kp.p(38);
        T dcol_h44 = kp.p(39);
        T dcol_h45 = kp.p(40);
        T dcol_h55 = kp.p(41);
        T lam_0 = kp.lam(0);


        // Clear generated output packets; nonzero entries are assigned below.
        kp.r.setZero();

        // q
        kp.q(0,0) = 0.16*x1 - 0.16*x1_ref;
        kp.q(1,0) = 0.0040000000000000001*y1 - 0.0040000000000000001*y1_ref;
        kp.q(2,0) = 0.90000000000000002*theta1 - 0.90000000000000002*theta1_ref;
        kp.q(3,0) = 0.59999999999999998*v1 - 0.59999999999999998*v1_ref;
        kp.q(4,0) = 0.16*x2 - 0.16*x2_ref;
        kp.q(5,0) = 0.0040000000000000001*y2 - 0.0040000000000000001*y2_ref;
        kp.q(6,0) = 0.90000000000000002*theta2 - 0.90000000000000002*theta2_ref;
        kp.q(7,0) = 0.59999999999999998*v2 - 0.59999999999999998*v2_ref;

        // r

        // Clear Hessian packets; nonzero entries are assigned below.
        kp.Q.setZero();
        kp.R.setZero();
        kp.H.setZero();

        // terminal Q (Mode 0=GN, 1=Exact)
        kp.Q(0,0) = 0.16;
        kp.Q(1,1) = 0.0040000000000000001;
        kp.Q(2,2) = 0.90000000000000002;
        kp.Q(3,3) = 0.59999999999999998;
        kp.Q(4,4) = 0.16;
        kp.Q(5,5) = 0.0040000000000000001;
        kp.Q(6,6) = 0.90000000000000002;
        kp.Q(7,7) = 0.59999999999999998;
        if constexpr (Mode == 1) kp.Q(0,0) += -dcol_h00*lam_0;
        if constexpr (Mode == 1) kp.Q(0,1) += -dcol_h01*lam_0;
        if constexpr (Mode == 1) kp.Q(0,2) += -dcol_h02*lam_0;
        if constexpr (Mode == 1) kp.Q(0,4) += -dcol_h03*lam_0;
        if constexpr (Mode == 1) kp.Q(0,5) += -dcol_h04*lam_0;
        if constexpr (Mode == 1) kp.Q(0,6) += -dcol_h05*lam_0;
        if constexpr (Mode == 1) kp.Q(1,0) += -dcol_h01*lam_0;
        if constexpr (Mode == 1) kp.Q(1,1) += -dcol_h11*lam_0;
        if constexpr (Mode == 1) kp.Q(1,2) += -dcol_h12*lam_0;
        if constexpr (Mode == 1) kp.Q(1,4) += -dcol_h13*lam_0;
        if constexpr (Mode == 1) kp.Q(1,5) += -dcol_h14*lam_0;
        if constexpr (Mode == 1) kp.Q(1,6) += -dcol_h15*lam_0;
        if constexpr (Mode == 1) kp.Q(2,0) += -dcol_h02*lam_0;
        if constexpr (Mode == 1) kp.Q(2,1) += -dcol_h12*lam_0;
        if constexpr (Mode == 1) kp.Q(2,2) += -dcol_h22*lam_0;
        if constexpr (Mode == 1) kp.Q(2,4) += -dcol_h23*lam_0;
        if constexpr (Mode == 1) kp.Q(2,5) += -dcol_h24*lam_0;
        if constexpr (Mode == 1) kp.Q(2,6) += -dcol_h25*lam_0;
        if constexpr (Mode == 1) kp.Q(4,0) += -dcol_h03*lam_0;
        if constexpr (Mode == 1) kp.Q(4,1) += -dcol_h13*lam_0;
        if constexpr (Mode == 1) kp.Q(4,2) += -dcol_h23*lam_0;
        if constexpr (Mode == 1) kp.Q(4,4) += -dcol_h33*lam_0;
        if constexpr (Mode == 1) kp.Q(4,5) += -dcol_h34*lam_0;
        if constexpr (Mode == 1) kp.Q(4,6) += -dcol_h35*lam_0;
        if constexpr (Mode == 1) kp.Q(5,0) += -dcol_h04*lam_0;
        if constexpr (Mode == 1) kp.Q(5,1) += -dcol_h14*lam_0;
        if constexpr (Mode == 1) kp.Q(5,2) += -dcol_h24*lam_0;
        if constexpr (Mode == 1) kp.Q(5,4) += -dcol_h34*lam_0;
        if constexpr (Mode == 1) kp.Q(5,5) += -dcol_h44*lam_0;
        if constexpr (Mode == 1) kp.Q(5,6) += -dcol_h45*lam_0;
        if constexpr (Mode == 1) kp.Q(6,0) += -dcol_h05*lam_0;
        if constexpr (Mode == 1) kp.Q(6,1) += -dcol_h15*lam_0;
        if constexpr (Mode == 1) kp.Q(6,2) += -dcol_h25*lam_0;
        if constexpr (Mode == 1) kp.Q(6,4) += -dcol_h35*lam_0;
        if constexpr (Mode == 1) kp.Q(6,5) += -dcol_h45*lam_0;
        if constexpr (Mode == 1) kp.Q(6,6) += -dcol_h55*lam_0;

        // terminal R (Mode 0=GN, 1=Exact)

        // terminal H (Mode 0=GN, 1=Exact)

        kp.cost = 0.45000000000000001*pow(theta1 - theta1_ref, 2) + 0.45000000000000001*pow(theta2 - theta2_ref, 2) + 0.29999999999999999*pow(v1 - v1_ref, 2) + 0.29999999999999999*pow(v2 - v2_ref, 2) + 0.080000000000000002*pow(x1 - x1_ref, 2) + 0.080000000000000002*pow(x2 - x2_ref, 2) + 0.002*pow(y1 - y1_ref, 2) + 0.002*pow(y2 - y2_ref, 2);
    }

    template<typename T>
    static void compute_terminal_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_terminal_cost_impl<T, 0>(kp);
    }

    template<typename T>
    static void compute_terminal_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_terminal_cost_impl<T, 1>(kp);
    }


    // --- 4. Compute All (Convenience) ---
    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        update_soft_constraint_weights(kp);
        compute_dynamics(kp, type, dt);
        compute_qp_constraints(kp);
        compute_true_constraints(kp);
        compute_cost(kp); // Default exact Hessian for backward compatibility.
    }

    template<typename T>
    static void compute_exact(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        update_soft_constraint_weights(kp);
        compute_dynamics(kp, type, dt);
        compute_qp_constraints(kp);
        compute_true_constraints(kp);
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
        T P_0_6 = Vxx(0,6);
        T P_0_7 = Vxx(0,7);
        T P_1_1 = Vxx(1,1);
        T P_1_2 = Vxx(1,2);
        T P_1_3 = Vxx(1,3);
        T P_1_4 = Vxx(1,4);
        T P_1_5 = Vxx(1,5);
        T P_1_6 = Vxx(1,6);
        T P_1_7 = Vxx(1,7);
        T P_2_2 = Vxx(2,2);
        T P_2_3 = Vxx(2,3);
        T P_2_4 = Vxx(2,4);
        T P_2_5 = Vxx(2,5);
        T P_2_6 = Vxx(2,6);
        T P_2_7 = Vxx(2,7);
        T P_3_3 = Vxx(3,3);
        T P_3_4 = Vxx(3,4);
        T P_3_5 = Vxx(3,5);
        T P_3_6 = Vxx(3,6);
        T P_3_7 = Vxx(3,7);
        T P_4_4 = Vxx(4,4);
        T P_4_5 = Vxx(4,5);
        T P_4_6 = Vxx(4,6);
        T P_4_7 = Vxx(4,7);
        T P_5_5 = Vxx(5,5);
        T P_5_6 = Vxx(5,6);
        T P_5_7 = Vxx(5,7);
        T P_6_6 = Vxx(6,6);
        T P_6_7 = Vxx(6,7);
        T P_7_7 = Vxx(7,7);
        T p_0 = Vx(0);
        T p_1 = Vx(1);
        T p_2 = Vx(2);
        T p_3 = Vx(3);
        T p_4 = Vx(4);
        T p_5 = Vx(5);
        T p_6 = Vx(6);
        T p_7 = Vx(7);
        T A_0_0 = kp.A(0,0);
        T A_0_2 = kp.A(0,2);
        T A_0_3 = kp.A(0,3);
        T A_1_1 = kp.A(1,1);
        T A_1_2 = kp.A(1,2);
        T A_1_3 = kp.A(1,3);
        T A_2_2 = kp.A(2,2);
        T A_3_3 = kp.A(3,3);
        T A_4_4 = kp.A(4,4);
        T A_4_6 = kp.A(4,6);
        T A_4_7 = kp.A(4,7);
        T A_5_5 = kp.A(5,5);
        T A_5_6 = kp.A(5,6);
        T A_5_7 = kp.A(5,7);
        T A_6_6 = kp.A(6,6);
        T A_7_7 = kp.A(7,7);
        T B_0_0 = kp.B(0,0);
        T B_0_1 = kp.B(0,1);
        T B_1_0 = kp.B(1,0);
        T B_1_1 = kp.B(1,1);
        T B_2_1 = kp.B(2,1);
        T B_3_0 = kp.B(3,0);
        T B_4_2 = kp.B(4,2);
        T B_4_3 = kp.B(4,3);
        T B_5_2 = kp.B(5,2);
        T B_5_3 = kp.B(5,3);
        T B_6_3 = kp.B(6,3);
        T B_7_2 = kp.B(7,2);

        // CSE Intermediate Variables
        T tmp_ric0 = A_0_2*P_0_0;
        T tmp_ric1 = A_1_2*P_0_1;
        T tmp_ric2 = A_2_2*P_0_2;
        T tmp_ric3 = A_0_3*P_0_0;
        T tmp_ric4 = A_1_3*P_0_1;
        T tmp_ric5 = A_3_3*P_0_3;
        T tmp_ric6 = A_0_0*P_0_4;
        T tmp_ric7 = A_0_0*P_0_5;
        T tmp_ric8 = A_0_2*P_0_1;
        T tmp_ric9 = A_1_2*P_1_1;
        T tmp_ric10 = A_2_2*P_1_2;
        T tmp_ric11 = A_0_3*P_0_1;
        T tmp_ric12 = A_1_3*P_1_1;
        T tmp_ric13 = A_3_3*P_1_3;
        T tmp_ric14 = A_1_1*P_1_4;
        T tmp_ric15 = A_1_1*P_1_5;
        T tmp_ric16 = tmp_ric0 + tmp_ric1 + tmp_ric2;
        T tmp_ric17 = tmp_ric10 + tmp_ric8 + tmp_ric9;
        T tmp_ric18 = A_0_2*P_0_4 + A_1_2*P_1_4 + A_2_2*P_2_4;
        T tmp_ric19 = A_0_2*P_0_5 + A_1_2*P_1_5 + A_2_2*P_2_5;
        T tmp_ric20 = A_0_3*P_0_4 + A_1_3*P_1_4 + A_3_3*P_3_4;
        T tmp_ric21 = A_0_3*P_0_5 + A_1_3*P_1_5 + A_3_3*P_3_5;
        T tmp_ric22 = A_4_6*P_4_4;
        T tmp_ric23 = A_5_6*P_4_5;
        T tmp_ric24 = A_6_6*P_4_6;
        T tmp_ric25 = A_4_7*P_4_4;
        T tmp_ric26 = A_5_7*P_4_5;
        T tmp_ric27 = A_7_7*P_4_7;
        T tmp_ric28 = A_4_6*P_4_5;
        T tmp_ric29 = A_5_6*P_5_5;
        T tmp_ric30 = A_6_6*P_5_6;
        T tmp_ric31 = A_4_7*P_4_5;
        T tmp_ric32 = A_5_7*P_5_5;
        T tmp_ric33 = A_7_7*P_5_7;
        T tmp_ric34 = tmp_ric22 + tmp_ric23 + tmp_ric24;
        T tmp_ric35 = tmp_ric28 + tmp_ric29 + tmp_ric30;
        T tmp_ric36 = B_0_0*P_0_0 + B_1_0*P_0_1 + B_3_0*P_0_3;
        T tmp_ric37 = B_0_0*P_0_1 + B_1_0*P_1_1 + B_3_0*P_1_3;
        T tmp_ric38 = B_0_0*P_0_3 + B_1_0*P_1_3 + B_3_0*P_3_3;
        T tmp_ric39 = B_0_0*P_0_2 + B_1_0*P_1_2 + B_3_0*P_2_3;
        T tmp_ric40 = B_0_0*P_0_4 + B_1_0*P_1_4 + B_3_0*P_3_4;
        T tmp_ric41 = B_0_0*P_0_5 + B_1_0*P_1_5 + B_3_0*P_3_5;
        T tmp_ric42 = B_0_0*P_0_7 + B_1_0*P_1_7 + B_3_0*P_3_7;
        T tmp_ric43 = B_0_0*P_0_6 + B_1_0*P_1_6 + B_3_0*P_3_6;
        T tmp_ric44 = B_0_1*P_0_0 + B_1_1*P_0_1 + B_2_1*P_0_2;
        T tmp_ric45 = B_0_1*P_0_1 + B_1_1*P_1_1 + B_2_1*P_1_2;
        T tmp_ric46 = B_0_1*P_0_2 + B_1_1*P_1_2 + B_2_1*P_2_2;
        T tmp_ric47 = B_0_1*P_0_4 + B_1_1*P_1_4 + B_2_1*P_2_4;
        T tmp_ric48 = B_0_1*P_0_5 + B_1_1*P_1_5 + B_2_1*P_2_5;
        T tmp_ric49 = B_0_1*P_0_7 + B_1_1*P_1_7 + B_2_1*P_2_7;
        T tmp_ric50 = B_0_1*P_0_6 + B_1_1*P_1_6 + B_2_1*P_2_6;
        T tmp_ric51 = B_4_2*P_4_4 + B_5_2*P_4_5 + B_7_2*P_4_7;
        T tmp_ric52 = B_4_2*P_4_5 + B_5_2*P_5_5 + B_7_2*P_5_7;
        T tmp_ric53 = B_4_2*P_4_7 + B_5_2*P_5_7 + B_7_2*P_7_7;
        T tmp_ric54 = B_4_2*P_4_6 + B_5_2*P_5_6 + B_7_2*P_6_7;
        T tmp_ric55 = B_4_3*P_4_4 + B_5_3*P_4_5 + B_6_3*P_4_6;
        T tmp_ric56 = B_4_3*P_4_5 + B_5_3*P_5_5 + B_6_3*P_5_6;
        T tmp_ric57 = B_4_3*P_4_6 + B_5_3*P_5_6 + B_6_3*P_6_6;
        T tmp_ric58 = B_4_2*P_0_4 + B_5_2*P_0_5 + B_7_2*P_0_7;
        T tmp_ric59 = B_4_2*P_1_4 + B_5_2*P_1_5 + B_7_2*P_1_7;
        T tmp_ric60 = B_4_3*P_0_4 + B_5_3*P_0_5 + B_6_3*P_0_6;
        T tmp_ric61 = B_4_3*P_1_4 + B_5_3*P_1_5 + B_6_3*P_1_6;

        // Accumulate Results
        kp.Q_bar(0,0) += pow(A_0_0, 2)*P_0_0;
        kp.Q_bar(0,1) += A_0_0*A_1_1*P_0_1;
        kp.Q_bar(0,2) += A_0_0*tmp_ric0 + A_0_0*tmp_ric1 + A_0_0*tmp_ric2;
        kp.Q_bar(0,3) += A_0_0*tmp_ric3 + A_0_0*tmp_ric4 + A_0_0*tmp_ric5;
        kp.Q_bar(0,4) += A_4_4*tmp_ric6;
        kp.Q_bar(0,5) += A_5_5*tmp_ric7;
        kp.Q_bar(0,6) += A_0_0*A_6_6*P_0_6 + A_4_6*tmp_ric6 + A_5_6*tmp_ric7;
        kp.Q_bar(0,7) += A_0_0*A_7_7*P_0_7 + A_4_7*tmp_ric6 + A_5_7*tmp_ric7;
        kp.Q_bar(1,1) += pow(A_1_1, 2)*P_1_1;
        kp.Q_bar(1,2) += A_1_1*tmp_ric10 + A_1_1*tmp_ric8 + A_1_1*tmp_ric9;
        kp.Q_bar(1,3) += A_1_1*tmp_ric11 + A_1_1*tmp_ric12 + A_1_1*tmp_ric13;
        kp.Q_bar(1,4) += A_4_4*tmp_ric14;
        kp.Q_bar(1,5) += A_5_5*tmp_ric15;
        kp.Q_bar(1,6) += A_1_1*A_6_6*P_1_6 + A_4_6*tmp_ric14 + A_5_6*tmp_ric15;
        kp.Q_bar(1,7) += A_1_1*A_7_7*P_1_7 + A_4_7*tmp_ric14 + A_5_7*tmp_ric15;
        kp.Q_bar(2,2) += A_0_2*tmp_ric16 + A_1_2*tmp_ric17 + A_2_2*(A_0_2*P_0_2 + A_1_2*P_1_2 + A_2_2*P_2_2);
        kp.Q_bar(2,3) += A_0_3*tmp_ric16 + A_1_3*tmp_ric17 + A_3_3*(A_0_2*P_0_3 + A_1_2*P_1_3 + A_2_2*P_2_3);
        kp.Q_bar(2,4) += A_4_4*tmp_ric18;
        kp.Q_bar(2,5) += A_5_5*tmp_ric19;
        kp.Q_bar(2,6) += A_4_6*tmp_ric18 + A_5_6*tmp_ric19 + A_6_6*(A_0_2*P_0_6 + A_1_2*P_1_6 + A_2_2*P_2_6);
        kp.Q_bar(2,7) += A_4_7*tmp_ric18 + A_5_7*tmp_ric19 + A_7_7*(A_0_2*P_0_7 + A_1_2*P_1_7 + A_2_2*P_2_7);
        kp.Q_bar(3,3) += A_0_3*(tmp_ric3 + tmp_ric4 + tmp_ric5) + A_1_3*(tmp_ric11 + tmp_ric12 + tmp_ric13) + A_3_3*(A_0_3*P_0_3 + A_1_3*P_1_3 + A_3_3*P_3_3);
        kp.Q_bar(3,4) += A_4_4*tmp_ric20;
        kp.Q_bar(3,5) += A_5_5*tmp_ric21;
        kp.Q_bar(3,6) += A_4_6*tmp_ric20 + A_5_6*tmp_ric21 + A_6_6*(A_0_3*P_0_6 + A_1_3*P_1_6 + A_3_3*P_3_6);
        kp.Q_bar(3,7) += A_4_7*tmp_ric20 + A_5_7*tmp_ric21 + A_7_7*(A_0_3*P_0_7 + A_1_3*P_1_7 + A_3_3*P_3_7);
        kp.Q_bar(4,4) += pow(A_4_4, 2)*P_4_4;
        kp.Q_bar(4,5) += A_4_4*A_5_5*P_4_5;
        kp.Q_bar(4,6) += A_4_4*tmp_ric22 + A_4_4*tmp_ric23 + A_4_4*tmp_ric24;
        kp.Q_bar(4,7) += A_4_4*tmp_ric25 + A_4_4*tmp_ric26 + A_4_4*tmp_ric27;
        kp.Q_bar(5,5) += pow(A_5_5, 2)*P_5_5;
        kp.Q_bar(5,6) += A_5_5*tmp_ric28 + A_5_5*tmp_ric29 + A_5_5*tmp_ric30;
        kp.Q_bar(5,7) += A_5_5*tmp_ric31 + A_5_5*tmp_ric32 + A_5_5*tmp_ric33;
        kp.Q_bar(6,6) += A_4_6*tmp_ric34 + A_5_6*tmp_ric35 + A_6_6*(A_4_6*P_4_6 + A_5_6*P_5_6 + A_6_6*P_6_6);
        kp.Q_bar(6,7) += A_4_7*tmp_ric34 + A_5_7*tmp_ric35 + A_7_7*(A_4_6*P_4_7 + A_5_6*P_5_7 + A_6_6*P_6_7);
        kp.Q_bar(7,7) += A_4_7*(tmp_ric25 + tmp_ric26 + tmp_ric27) + A_5_7*(tmp_ric31 + tmp_ric32 + tmp_ric33) + A_7_7*(A_4_7*P_4_7 + A_5_7*P_5_7 + A_7_7*P_7_7);
        kp.R_bar(0,0) += B_0_0*tmp_ric36 + B_1_0*tmp_ric37 + B_3_0*tmp_ric38;
        kp.R_bar(0,1) += B_0_1*tmp_ric36 + B_1_1*tmp_ric37 + B_2_1*tmp_ric39;
        kp.R_bar(0,2) += B_4_2*tmp_ric40 + B_5_2*tmp_ric41 + B_7_2*tmp_ric42;
        kp.R_bar(0,3) += B_4_3*tmp_ric40 + B_5_3*tmp_ric41 + B_6_3*tmp_ric43;
        kp.R_bar(1,1) += B_0_1*tmp_ric44 + B_1_1*tmp_ric45 + B_2_1*tmp_ric46;
        kp.R_bar(1,2) += B_4_2*tmp_ric47 + B_5_2*tmp_ric48 + B_7_2*tmp_ric49;
        kp.R_bar(1,3) += B_4_3*tmp_ric47 + B_5_3*tmp_ric48 + B_6_3*tmp_ric50;
        kp.R_bar(2,2) += B_4_2*tmp_ric51 + B_5_2*tmp_ric52 + B_7_2*tmp_ric53;
        kp.R_bar(2,3) += B_4_3*tmp_ric51 + B_5_3*tmp_ric52 + B_6_3*tmp_ric54;
        kp.R_bar(3,3) += B_4_3*tmp_ric55 + B_5_3*tmp_ric56 + B_6_3*tmp_ric57;
        kp.H_bar(0,0) += A_0_0*tmp_ric36;
        kp.H_bar(0,1) += A_1_1*tmp_ric37;
        kp.H_bar(0,2) += A_0_2*tmp_ric36 + A_1_2*tmp_ric37 + A_2_2*tmp_ric39;
        kp.H_bar(0,3) += A_0_3*tmp_ric36 + A_1_3*tmp_ric37 + A_3_3*tmp_ric38;
        kp.H_bar(0,4) += A_4_4*tmp_ric40;
        kp.H_bar(0,5) += A_5_5*tmp_ric41;
        kp.H_bar(0,6) += A_4_6*tmp_ric40 + A_5_6*tmp_ric41 + A_6_6*tmp_ric43;
        kp.H_bar(0,7) += A_4_7*tmp_ric40 + A_5_7*tmp_ric41 + A_7_7*tmp_ric42;
        kp.H_bar(1,0) += A_0_0*tmp_ric44;
        kp.H_bar(1,1) += A_1_1*tmp_ric45;
        kp.H_bar(1,2) += A_0_2*tmp_ric44 + A_1_2*tmp_ric45 + A_2_2*tmp_ric46;
        kp.H_bar(1,3) += A_0_3*tmp_ric44 + A_1_3*tmp_ric45 + A_3_3*(B_0_1*P_0_3 + B_1_1*P_1_3 + B_2_1*P_2_3);
        kp.H_bar(1,4) += A_4_4*tmp_ric47;
        kp.H_bar(1,5) += A_5_5*tmp_ric48;
        kp.H_bar(1,6) += A_4_6*tmp_ric47 + A_5_6*tmp_ric48 + A_6_6*tmp_ric50;
        kp.H_bar(1,7) += A_4_7*tmp_ric47 + A_5_7*tmp_ric48 + A_7_7*tmp_ric49;
        kp.H_bar(2,0) += A_0_0*tmp_ric58;
        kp.H_bar(2,1) += A_1_1*tmp_ric59;
        kp.H_bar(2,2) += A_0_2*tmp_ric58 + A_1_2*tmp_ric59 + A_2_2*(B_4_2*P_2_4 + B_5_2*P_2_5 + B_7_2*P_2_7);
        kp.H_bar(2,3) += A_0_3*tmp_ric58 + A_1_3*tmp_ric59 + A_3_3*(B_4_2*P_3_4 + B_5_2*P_3_5 + B_7_2*P_3_7);
        kp.H_bar(2,4) += A_4_4*tmp_ric51;
        kp.H_bar(2,5) += A_5_5*tmp_ric52;
        kp.H_bar(2,6) += A_4_6*tmp_ric51 + A_5_6*tmp_ric52 + A_6_6*tmp_ric54;
        kp.H_bar(2,7) += A_4_7*tmp_ric51 + A_5_7*tmp_ric52 + A_7_7*tmp_ric53;
        kp.H_bar(3,0) += A_0_0*tmp_ric60;
        kp.H_bar(3,1) += A_1_1*tmp_ric61;
        kp.H_bar(3,2) += A_0_2*tmp_ric60 + A_1_2*tmp_ric61 + A_2_2*(B_4_3*P_2_4 + B_5_3*P_2_5 + B_6_3*P_2_6);
        kp.H_bar(3,3) += A_0_3*tmp_ric60 + A_1_3*tmp_ric61 + A_3_3*(B_4_3*P_3_4 + B_5_3*P_3_5 + B_6_3*P_3_6);
        kp.H_bar(3,4) += A_4_4*tmp_ric55;
        kp.H_bar(3,5) += A_5_5*tmp_ric56;
        kp.H_bar(3,6) += A_4_6*tmp_ric55 + A_5_6*tmp_ric56 + A_6_6*tmp_ric57;
        kp.H_bar(3,7) += A_4_7*tmp_ric55 + A_5_7*tmp_ric56 + A_7_7*(B_4_3*P_4_7 + B_5_3*P_5_7 + B_6_3*P_6_7);
        kp.q_bar(0,0) += A_0_0*p_0;
        kp.q_bar(1,0) += A_1_1*p_1;
        kp.q_bar(2,0) += A_0_2*p_0 + A_1_2*p_1 + A_2_2*p_2;
        kp.q_bar(3,0) += A_0_3*p_0 + A_1_3*p_1 + A_3_3*p_3;
        kp.q_bar(4,0) += A_4_4*p_4;
        kp.q_bar(5,0) += A_5_5*p_5;
        kp.q_bar(6,0) += A_4_6*p_4 + A_5_6*p_5 + A_6_6*p_6;
        kp.q_bar(7,0) += A_4_7*p_4 + A_5_7*p_5 + A_7_7*p_7;
        kp.r_bar(0,0) += B_0_0*p_0 + B_1_0*p_1 + B_3_0*p_3;
        kp.r_bar(1,0) += B_0_1*p_0 + B_1_1*p_1 + B_2_1*p_2;
        kp.r_bar(2,0) += B_4_2*p_4 + B_5_2*p_5 + B_7_2*p_7;
        kp.r_bar(3,0) += B_4_3*p_4 + B_5_3*p_5 + B_6_3*p_6;

        // Fill Lower Triangles (Symmetry)
        kp.Q_bar(1,0) = kp.Q_bar(0,1);
        kp.Q_bar(2,0) = kp.Q_bar(0,2);
        kp.Q_bar(3,0) = kp.Q_bar(0,3);
        kp.Q_bar(4,0) = kp.Q_bar(0,4);
        kp.Q_bar(5,0) = kp.Q_bar(0,5);
        kp.Q_bar(6,0) = kp.Q_bar(0,6);
        kp.Q_bar(7,0) = kp.Q_bar(0,7);
        kp.Q_bar(2,1) = kp.Q_bar(1,2);
        kp.Q_bar(3,1) = kp.Q_bar(1,3);
        kp.Q_bar(4,1) = kp.Q_bar(1,4);
        kp.Q_bar(5,1) = kp.Q_bar(1,5);
        kp.Q_bar(6,1) = kp.Q_bar(1,6);
        kp.Q_bar(7,1) = kp.Q_bar(1,7);
        kp.Q_bar(3,2) = kp.Q_bar(2,3);
        kp.Q_bar(4,2) = kp.Q_bar(2,4);
        kp.Q_bar(5,2) = kp.Q_bar(2,5);
        kp.Q_bar(6,2) = kp.Q_bar(2,6);
        kp.Q_bar(7,2) = kp.Q_bar(2,7);
        kp.Q_bar(4,3) = kp.Q_bar(3,4);
        kp.Q_bar(5,3) = kp.Q_bar(3,5);
        kp.Q_bar(6,3) = kp.Q_bar(3,6);
        kp.Q_bar(7,3) = kp.Q_bar(3,7);
        kp.Q_bar(5,4) = kp.Q_bar(4,5);
        kp.Q_bar(6,4) = kp.Q_bar(4,6);
        kp.Q_bar(7,4) = kp.Q_bar(4,7);
        kp.Q_bar(6,5) = kp.Q_bar(5,6);
        kp.Q_bar(7,5) = kp.Q_bar(5,7);
        kp.Q_bar(7,6) = kp.Q_bar(6,7);
        kp.R_bar(1,0) = kp.R_bar(0,1);
        kp.R_bar(2,0) = kp.R_bar(0,2);
        kp.R_bar(3,0) = kp.R_bar(0,3);
        kp.R_bar(2,1) = kp.R_bar(1,2);
        kp.R_bar(3,1) = kp.R_bar(1,3);
        kp.R_bar(3,2) = kp.R_bar(2,3);

    }

};
}
