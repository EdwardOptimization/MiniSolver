#pragma once
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/matrix_defs.h"
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

    static constexpr std::array<double, NC> constraint_weights = {0.0, 0.0, 0.0, 0.0, 0.0};
    static constexpr std::array<int, NC> constraint_types = {0, 0, 0, 0, 0};


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
        T theta = x_in(2);
        T v = x_in(3);
        T acc = u_in(0);
        T steer = u_in(1);
        T L = p_in(6);

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
    // NEW SPLIT ARCHITECTURE: Reads State, writes ModelData
    template<typename T>
    static void compute_dynamics(
        const StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model,
        IntegratorType type,
        double dt)
    {
        T x = state.x(0);
        T y = state.x(1);
        T theta = state.x(2);
        T v = state.x(3);
        T acc = state.u(0);
        T steer = state.u(1);
        T L = state.p(6);

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
                model.f_resid(0) = tmp_d1 + x;
                model.f_resid(1) = tmp_d3 + y;
                model.f_resid(2) = theta + tmp_d6*v;
                model.f_resid(3) = acc*dt + v;
                model.A.setZero();
                model.A(0,0) = 1;
                model.A(0,2) = -tmp_d3;
                model.A(0,3) = tmp_d0;
                model.A(1,1) = 1;
                model.A(1,2) = tmp_d1;
                model.A(1,3) = tmp_d2;
                model.A(2,2) = 1;
                model.A(2,3) = tmp_d6;
                model.A(3,3) = 1;
                model.B.setZero();
                model.B(2,1) = tmp_d5*v*(pow(tmp_d4, 2) + 1);
                model.B(3,0) = dt;
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
                model.f_resid(0) = tmp_d10 + x;
                model.f_resid(1) = tmp_d13 + y;
                model.f_resid(2) = theta + tmp_d6;
                model.f_resid(3) = tmp_d0 + v;
                model.A.setZero();
                model.A(0,0) = 1;
                model.A(0,2) = -tmp_d13;
                model.A(0,3) = -tmp_d15*tmp_d16 + tmp_d9;
                model.A(1,1) = 1;
                model.A(1,2) = tmp_d10;
                model.A(1,3) = tmp_d12 + tmp_d16*tmp_d17;
                model.A(2,2) = 1;
                model.A(2,3) = tmp_d5;
                model.A(3,3) = 1;
                model.B.setZero();
                model.B(0,0) = -tmp_d11*tmp_d18 + tmp_d17;
                model.B(0,1) = -tmp_d15*tmp_d20;
                model.B(1,0) = tmp_d15 + tmp_d18*tmp_d8;
                model.B(1,1) = tmp_d17*tmp_d20;
                model.B(2,0) = tmp_d14*tmp_d4;
                model.B(2,1) = dt*tmp_d1*tmp_d19;
                model.B(3,0) = dt;
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
                model.f_resid(0) = tmp_d21 + x;
                model.f_resid(1) = tmp_d20*tmp_d29 + y;
                model.f_resid(2) = theta + tmp_d20*(tmp_d2*tmp_d6 + tmp_d30*tmp_d6 + tmp_d6*v);
                model.f_resid(3) = tmp_d16;
                model.A.setZero();
                model.A(0,0) = 1;
                model.A(0,2) = -tmp_d20*tmp_d29;
                model.A(0,3) = tmp_d20*(tmp_d0 + tmp_d15 + tmp_d19 - tmp_d24*tmp_d7 - tmp_d32*tmp_d33 - tmp_d33*tmp_d34 + tmp_d9);
                model.A(1,1) = 1;
                model.A(1,2) = tmp_d21;
                model.A(1,3) = tmp_d20*(tmp_d10*tmp_d7 + tmp_d22 + tmp_d23 + tmp_d26 + tmp_d28 + tmp_d33*tmp_d35 + tmp_d33*tmp_d36);
                model.A(2,2) = 1;
                model.A(2,3) = tmp_d31*tmp_d6;
                model.A(3,3) = 1;
                model.B.setZero();
                model.B(0,0) = tmp_d20*(dt*tmp_d9 - tmp_d24*tmp_d38 - tmp_d25*tmp_d40 - tmp_d27*tmp_d41 + tmp_d35 + tmp_d36);
                model.B(0,1) = tmp_d20*(-tmp_d24*tmp_d44 - tmp_d32*tmp_d43 - tmp_d34*tmp_d45);
                model.B(1,0) = tmp_d20*(dt*tmp_d23 + tmp_d10*tmp_d38 + tmp_d14*tmp_d40 + tmp_d18*tmp_d41 + tmp_d32 + tmp_d34);
                model.B(1,1) = tmp_d20*(tmp_d10*tmp_d44 + tmp_d35*tmp_d43 + tmp_d36*tmp_d45);
                model.B(2,0) = tmp_d39;
                model.B(2,1) = tmp_d20*(tmp_d2*tmp_d42 + tmp_d30*tmp_d42 + tmp_d42*v);
                model.B(3,0) = tmp_d31;
                break;
            }
        }
    }

    // --- 2. Compute Constraints (g_val, C, D) ---
    // NEW SPLIT ARCHITECTURE: Reads/writes State, writes ModelData
    template<typename T>
    static void compute_constraints(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
        T x = state.x(0);
        T y = state.x(1);
        T acc = state.u(0);
        T steer = state.u(1);
        T obs_x = state.p(3);
        T obs_y = state.p(4);
        T obs_rad = state.p(5);
        T car_rad = state.p(7);

        // --- Special Constraints Pre-Calculation ---

        T tmp_c0 = -obs_x + x;
        T tmp_c1 = -obs_y + y;
        T tmp_c2 = sqrt(pow(tmp_c0, 2) + pow(tmp_c1, 2) + 9.9999999999999995e-7);
        T tmp_c3 = 1.0/tmp_c2;

        // g_val
        state.g_val(0,0) = acc - 3.0;
        state.g_val(1,0) = -acc - 3.0;
        state.g_val(2,0) = steer - 0.5;
        state.g_val(3,0) = -steer - 0.5;
        state.g_val(4,0) = -tmp_c2 + sqrt(pow(car_rad + obs_rad, 2));

        // C
        model.C(0,0) = 0;
        model.C(0,1) = 0;
        model.C(0,2) = 0;
        model.C(0,3) = 0;
        model.C(1,0) = 0;
        model.C(1,1) = 0;
        model.C(1,2) = 0;
        model.C(1,3) = 0;
        model.C(2,0) = 0;
        model.C(2,1) = 0;
        model.C(2,2) = 0;
        model.C(2,3) = 0;
        model.C(3,0) = 0;
        model.C(3,1) = 0;
        model.C(3,2) = 0;
        model.C(3,3) = 0;
        model.C(4,0) = -tmp_c0*tmp_c3;
        model.C(4,1) = -tmp_c1*tmp_c3;
        model.C(4,2) = 0;
        model.C(4,3) = 0;

        // D
        model.D(0,0) = 1;
        model.D(0,1) = 0;
        model.D(1,0) = -1;
        model.D(1,1) = 0;
        model.D(2,0) = 0;
        model.D(2,1) = 1;
        model.D(3,0) = 0;
        model.D(3,1) = -1;
        model.D(4,0) = 0;
        model.D(4,1) = 0;

    }

    // --- 3. Compute Cost (Implemented via template for Exact/GN) ---
    template<typename T, int Mode>
    static void compute_cost_impl(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model) {
        T x = state.x(0);
        T y = state.x(1);
        T theta = state.x(2);
        T v = state.x(3);
        T acc = state.u(0);
        T steer = state.u(1);
        T v_ref = state.p(0);
        T x_ref = state.p(1);
        T y_ref = state.p(2);
        T obs_x = state.p(3);
        T obs_y = state.p(4);
        T w_pos = state.p(8);
        T w_vel = state.p(9);
        T w_theta = state.p(10);
        T w_acc = state.p(11);
        T w_steer = state.p(12);
        T lam_4 = state.lam(4);

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
        model.q(0,0) = w_pos*(2*x - 2*x_ref);
        model.q(1,0) = w_pos*(2*y - 2*y_ref);
        model.q(2,0) = theta*tmp_j0;
        model.q(3,0) = w_vel*(2*v - 2*v_ref);

        // r
        model.r(0,0) = acc*tmp_j1;
        model.r(1,0) = steer*tmp_j2;

        // Q (Mode 0=GN, 1=Exact)
        model.Q(0,0) = tmp_j3;
        if constexpr (Mode == 1) model.Q(0,0) += lam_4*(-tmp_j11*tmp_j4 - tmp_j9);
        model.Q(0,1) = 0;
        if constexpr (Mode == 1) model.Q(0,1) += tmp_j12;
        model.Q(0,2) = 0;
        model.Q(0,3) = 0;
        model.Q(1,0) = 0;
        if constexpr (Mode == 1) model.Q(1,0) += tmp_j12;
        model.Q(1,1) = tmp_j3;
        if constexpr (Mode == 1) model.Q(1,1) += lam_4*(-tmp_j10*tmp_j6*tmp_j7 - tmp_j9);
        model.Q(1,2) = 0;
        model.Q(1,3) = 0;
        model.Q(2,0) = 0;
        model.Q(2,1) = 0;
        model.Q(2,2) = tmp_j0;
        model.Q(2,3) = 0;
        model.Q(3,0) = 0;
        model.Q(3,1) = 0;
        model.Q(3,2) = 0;
        model.Q(3,3) = 2*w_vel;

        // R (Mode 0=GN, 1=Exact)
        model.R(0,0) = tmp_j1;
        model.R(0,1) = 0;
        model.R(1,0) = 0;
        model.R(1,1) = tmp_j2;

        // H (Mode 0=GN, 1=Exact)
        model.H(0,0) = 0;
        model.H(0,1) = 0;
        model.H(0,2) = 0;
        model.H(0,3) = 0;
        model.H(1,0) = 0;
        model.H(1,1) = 0;
        model.H(1,2) = 0;
        model.H(1,3) = 0;

        state.cost = pow(acc, 2)*w_acc + pow(steer, 2)*w_steer + pow(theta, 2)*w_theta + w_pos*pow(x - x_ref, 2) + w_pos*pow(y - y_ref, 2) + w_vel*pow(v - v_ref, 2);
    }

template<typename T>
    static void compute_cost_gn(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model) {
        compute_cost_impl<T, 0>(state, model);
    }

    template<typename T>
    static void compute_cost_exact(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model) {
        compute_cost_impl<T, 1>(state, model);
    }

    template<typename T>
    static void compute_cost(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model) {
        compute_cost_impl<T, 1>(state, model);
    }


    // --- 4. Compute All (Convenience wrappers for backward compatibility) ---
    // These are kept for tools that still use the old API
    template<typename T>
    static void compute(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model,
        IntegratorType type,
        double dt)
    {
        compute_dynamics(state, model, type, dt);
        compute_constraints(state, model);
        compute_cost(state, model); // Default GN
    }

    template<typename T>
    static void compute_exact(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model,
        IntegratorType type,
        double dt)
    {
        compute_dynamics(state, model, type, dt);
        compute_constraints(state, model);
        compute_cost_exact(state, model); // Exact Hessian
    }

    // --- 5. Sparse Kernels (Generated) ---
    
    // --- 6. Fused Riccati Kernel (Generated) ---
    // NEW SPLIT ARCHITECTURE: Reads Model, writes Workspace
    // Updates workspace Q_bar, R_bar, H_bar, q_bar, r_bar in one go.
    // Uses Vxx, Vx from next step.
    template<typename T>
    static void compute_fused_riccati_step(
        const MSMat<T, NX, NX>& Vxx, 
        const MSVec<T, NX>& Vx,
        const ModelData<T,NX,NU,NC>& model,
        SolverWorkspace<T,NX,NU,NC>& work) 
    {
        T P_0_0 = Vxx(0,0);
        T P_0_1 = Vxx(0,1);
        T P_0_2 = Vxx(0,2);
        T P_0_3 = Vxx(0,3);
        T P_1_1 = Vxx(1,1);
        T P_1_2 = Vxx(1,2);
        T P_1_3 = Vxx(1,3);
        T P_2_2 = Vxx(2,2);
        T P_2_3 = Vxx(2,3);
        T P_3_3 = Vxx(3,3);
        T p_0 = Vx(0);
        T p_1 = Vx(1);
        T p_2 = Vx(2);
        T p_3 = Vx(3);
        T A_0_0 = model.A(0,0);
        T A_0_2 = model.A(0,2);
        T A_0_3 = model.A(0,3);
        T A_1_1 = model.A(1,1);
        T A_1_2 = model.A(1,2);
        T A_1_3 = model.A(1,3);
        T A_2_2 = model.A(2,2);
        T A_2_3 = model.A(2,3);
        T A_3_3 = model.A(3,3);
        T B_0_0 = model.B(0,0);
        T B_0_1 = model.B(0,1);
        T B_1_0 = model.B(1,0);
        T B_1_1 = model.B(1,1);
        T B_2_0 = model.B(2,0);
        T B_2_1 = model.B(2,1);
        T B_3_0 = model.B(3,0);

        // CSE Intermediate Variables
        T tmp_ric0 = A_0_2*P_0_0;
        T tmp_ric1 = A_1_2*P_0_1;
        T tmp_ric2 = A_2_2*P_0_2;
        T tmp_ric3 = A_0_3*P_0_0;
        T tmp_ric4 = A_1_3*P_0_1;
        T tmp_ric5 = A_2_3*P_0_2;
        T tmp_ric6 = A_3_3*P_0_3;
        T tmp_ric7 = A_0_2*P_0_1;
        T tmp_ric8 = A_1_2*P_1_1;
        T tmp_ric9 = A_2_2*P_1_2;
        T tmp_ric10 = A_0_3*P_0_1;
        T tmp_ric11 = A_1_3*P_1_1;
        T tmp_ric12 = A_2_3*P_1_2;
        T tmp_ric13 = A_3_3*P_1_3;
        T tmp_ric14 = tmp_ric0 + tmp_ric1 + tmp_ric2;
        T tmp_ric15 = tmp_ric7 + tmp_ric8 + tmp_ric9;
        T tmp_ric16 = A_0_2*P_0_2 + A_1_2*P_1_2 + A_2_2*P_2_2;
        T tmp_ric17 = B_0_0*P_0_0 + B_1_0*P_0_1 + B_2_0*P_0_2 + B_3_0*P_0_3;
        T tmp_ric18 = B_0_0*P_0_1 + B_1_0*P_1_1 + B_2_0*P_1_2 + B_3_0*P_1_3;
        T tmp_ric19 = B_0_0*P_0_2 + B_1_0*P_1_2 + B_2_0*P_2_2 + B_3_0*P_2_3;
        T tmp_ric20 = B_0_0*P_0_3 + B_1_0*P_1_3 + B_2_0*P_2_3 + B_3_0*P_3_3;
        T tmp_ric21 = B_0_1*P_0_0 + B_1_1*P_0_1 + B_2_1*P_0_2;
        T tmp_ric22 = B_0_1*P_0_1 + B_1_1*P_1_1 + B_2_1*P_1_2;
        T tmp_ric23 = B_0_1*P_0_2 + B_1_1*P_1_2 + B_2_1*P_2_2;

        // Accumulate Results to Workspace
        work.Q_bar(0,0) += pow(A_0_0, 2)*P_0_0;
        work.Q_bar(0,1) += A_0_0*A_1_1*P_0_1;
        work.Q_bar(0,2) += A_0_0*tmp_ric0 + A_0_0*tmp_ric1 + A_0_0*tmp_ric2;
        work.Q_bar(0,3) += A_0_0*tmp_ric3 + A_0_0*tmp_ric4 + A_0_0*tmp_ric5 + A_0_0*tmp_ric6;
        work.Q_bar(1,1) += pow(A_1_1, 2)*P_1_1;
        work.Q_bar(1,2) += A_1_1*tmp_ric7 + A_1_1*tmp_ric8 + A_1_1*tmp_ric9;
        work.Q_bar(1,3) += A_1_1*tmp_ric10 + A_1_1*tmp_ric11 + A_1_1*tmp_ric12 + A_1_1*tmp_ric13;
        work.Q_bar(2,2) += A_0_2*tmp_ric14 + A_1_2*tmp_ric15 + A_2_2*tmp_ric16;
        work.Q_bar(2,3) += A_0_3*tmp_ric14 + A_1_3*tmp_ric15 + A_2_3*tmp_ric16 + A_3_3*(A_0_2*P_0_3 + A_1_2*P_1_3 + A_2_2*P_2_3);
        work.Q_bar(3,3) += A_0_3*(tmp_ric3 + tmp_ric4 + tmp_ric5 + tmp_ric6) + A_1_3*(tmp_ric10 + tmp_ric11 + tmp_ric12 + tmp_ric13) + A_2_3*(A_0_3*P_0_2 + A_1_3*P_1_2 + A_2_3*P_2_2 + A_3_3*P_2_3) + A_3_3*(A_0_3*P_0_3 + A_1_3*P_1_3 + A_2_3*P_2_3 + A_3_3*P_3_3);
        work.R_bar(0,0) += B_0_0*tmp_ric17 + B_1_0*tmp_ric18 + B_2_0*tmp_ric19 + B_3_0*tmp_ric20;
        work.R_bar(0,1) += B_0_1*tmp_ric17 + B_1_1*tmp_ric18 + B_2_1*tmp_ric19;
        work.R_bar(1,1) += B_0_1*tmp_ric21 + B_1_1*tmp_ric22 + B_2_1*tmp_ric23;
        work.H_bar(0,0) += A_0_0*tmp_ric17;
        work.H_bar(0,1) += A_1_1*tmp_ric18;
        work.H_bar(0,2) += A_0_2*tmp_ric17 + A_1_2*tmp_ric18 + A_2_2*tmp_ric19;
        work.H_bar(0,3) += A_0_3*tmp_ric17 + A_1_3*tmp_ric18 + A_2_3*tmp_ric19 + A_3_3*tmp_ric20;
        work.H_bar(1,0) += A_0_0*tmp_ric21;
        work.H_bar(1,1) += A_1_1*tmp_ric22;
        work.H_bar(1,2) += A_0_2*tmp_ric21 + A_1_2*tmp_ric22 + A_2_2*tmp_ric23;
        work.H_bar(1,3) += A_0_3*tmp_ric21 + A_1_3*tmp_ric22 + A_2_3*tmp_ric23 + A_3_3*(B_0_1*P_0_3 + B_1_1*P_1_3 + B_2_1*P_2_3);
        work.q_bar(0,0) += A_0_0*p_0;
        work.q_bar(1,0) += A_1_1*p_1;
        work.q_bar(2,0) += A_0_2*p_0 + A_1_2*p_1 + A_2_2*p_2;
        work.q_bar(3,0) += A_0_3*p_0 + A_1_3*p_1 + A_2_3*p_2 + A_3_3*p_3;
        work.r_bar(0,0) += B_0_0*p_0 + B_1_0*p_1 + B_2_0*p_2 + B_3_0*p_3;
        work.r_bar(1,0) += B_0_1*p_0 + B_1_1*p_1 + B_2_1*p_2;

        // Fill Lower Triangles (Symmetry)
        work.Q_bar(1,0) = work.Q_bar(0,1);
        work.Q_bar(2,0) = work.Q_bar(0,2);
        work.Q_bar(3,0) = work.Q_bar(0,3);
        work.Q_bar(2,1) = work.Q_bar(1,2);
        work.Q_bar(3,1) = work.Q_bar(1,3);
        work.Q_bar(3,2) = work.Q_bar(2,3);
        work.R_bar(1,0) = work.R_bar(0,1);

    }
    
};
}
