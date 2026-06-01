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

struct InnerDcolNorm2Model {
    // --- Constants ---
    static const int NX=1;
    static const int NU=5;
    static const int NC=10;
    static const int NP=6;

    static constexpr std::uint64_t model_fingerprint = 0x099b16516f1463f4ull;

    static constexpr IntegratorType generated_integrator = IntegratorType::EULER_EXPLICIT;

    static constexpr std::array<bool, NC> constraint_has_l1 = {false, false, false, false, false, false, false, false, false, false};
    static constexpr std::array<bool, NC> constraint_has_l2 = {false, false, false, false, false, false, false, false, false, false};
    static constexpr bool any_l1_constraints = false;
    static constexpr bool any_l2_constraints = false;


    // --- Name Arrays (for Map Construction) ---
    static constexpr std::array<const char*, NX> state_names = {
        "dummy",
    };

    static constexpr std::array<const char*, NU> control_names = {
        "p1x",
        "p1y",
        "p2x",
        "p2y",
        "alpha",
    };

    static constexpr std::array<const char*, NP> param_names = {
        "x1",
        "y1",
        "theta1",
        "x2",
        "y2",
        "theta2",
    };


    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in)
    {
        (void)x_in;
        (void)u_in;
        (void)p_in;

        MSVec<T, NX> xdot;
        xdot(0) = 0;
        return xdot;

    }

    // --- Continuous Dynamics Jacobians (for implicit integrators) ---
    template<typename T>
    static ContinuousJacobians<T, NX, NU> jacobian_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in)
    {
        (void)x_in;
        (void)u_in;
        (void)p_in;

        ContinuousJacobians<T, NX, NU> jac;

        // Clear continuous Jacobian packets; nonzero entries are assigned below.
        jac.Jx.setZero();
        jac.Ju.setZero();

        // Jx = df/dx

        // Ju = df/du

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
        T dummy = kp.x(0);

        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
            {
                kp.f_resid(0) = dummy;

                // Clear dynamics Jacobian A; nonzero entries are assigned below.
                kp.A.setZero();
                kp.A(0,0) = 1;

                // Clear dynamics Jacobian B; nonzero entries are assigned below.
                kp.B.setZero();
                return;
            }
            case IntegratorType::RUNGE_KUTTA_2:
            {
                kp.f_resid(0) = dummy;

                // Clear dynamics Jacobian A; nonzero entries are assigned below.
                kp.A.setZero();
                kp.A(0,0) = 1;

                // Clear dynamics Jacobian B; nonzero entries are assigned below.
                kp.B.setZero();
                return;
            }
            case IntegratorType::RUNGE_KUTTA_4:
            {
                kp.f_resid(0) = dummy;

                // Clear dynamics Jacobian A; nonzero entries are assigned below.
                kp.A.setZero();
                kp.A(0,0) = 1;

                // Clear dynamics Jacobian B; nonzero entries are assigned below.
                kp.B.setZero();
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
        T p1x = kp.u(0);
        T p1y = kp.u(1);
        T p2x = kp.u(2);
        T p2y = kp.u(3);
        T alpha = kp.u(4);
        T x1 = kp.p(0);
        T y1 = kp.p(1);
        T theta1 = kp.p(2);
        T x2 = kp.p(3);
        T y2 = kp.p(4);
        T theta2 = kp.p(5);

        // --- Special Constraints Pre-Calculation ---

        T tmp_c0 = 2.25*alpha;
        T tmp_c1 = -tmp_c0;
        T tmp_c2 = cos(theta1);
        T tmp_c3 = p1x - x1;
        T tmp_c4 = sin(theta1);
        T tmp_c5 = p1y - y1;
        T tmp_c6 = tmp_c2*tmp_c3 + tmp_c4*tmp_c5;
        T tmp_c7 = 0.94999999999999996*alpha;
        T tmp_c8 = tmp_c3*tmp_c4;
        T tmp_c9 = tmp_c2*tmp_c5;
        T tmp_c10 = cos(theta2);
        T tmp_c11 = p2x - x2;
        T tmp_c12 = sin(theta2);
        T tmp_c13 = p2y - y2;
        T tmp_c14 = tmp_c10*tmp_c11 + tmp_c12*tmp_c13;
        T tmp_c15 = tmp_c11*tmp_c12;
        T tmp_c16 = tmp_c10*tmp_c13;
        T tmp_c17 = p1x - p2x;
        T tmp_c18 = p1y - p2y;
        T tmp_c19 = sqrt(pow(tmp_c17, 2) + pow(tmp_c18, 2) + 1.0e-10);
        T tmp_c20 = -tmp_c2;
        T tmp_c21 = -tmp_c4;
        T tmp_c22 = -tmp_c10;
        T tmp_c23 = -tmp_c12;
        T tmp_c24 = 1.0/tmp_c19;

        // Clear generated output packets; nonzero entries are assigned below.
        kp.C.setZero();
        kp.D.setZero();

        // g_val
        kp.g_val(0,0) = tmp_c1 + tmp_c6;
        kp.g_val(1,0) = -tmp_c0 - tmp_c6;
        kp.g_val(2,0) = -tmp_c7 - tmp_c8 + tmp_c9;
        kp.g_val(3,0) = -tmp_c7 + tmp_c8 - tmp_c9;
        kp.g_val(4,0) = tmp_c1 + tmp_c14;
        kp.g_val(5,0) = -tmp_c0 - tmp_c14;
        kp.g_val(6,0) = -tmp_c15 + tmp_c16 - tmp_c7;
        kp.g_val(7,0) = tmp_c15 - tmp_c16 - tmp_c7;
        kp.g_val(8,0) = -1.0*alpha + tmp_c19;
        kp.g_val(9,0) = -alpha;

        // C

        // D
        kp.D(0,0) = tmp_c2;
        kp.D(0,1) = tmp_c4;
        kp.D(0,4) = -2.25;
        kp.D(1,0) = tmp_c20;
        kp.D(1,1) = tmp_c21;
        kp.D(1,4) = -2.25;
        kp.D(2,0) = tmp_c21;
        kp.D(2,1) = tmp_c2;
        kp.D(2,4) = -0.94999999999999996;
        kp.D(3,0) = tmp_c4;
        kp.D(3,1) = tmp_c20;
        kp.D(3,4) = -0.94999999999999996;
        kp.D(4,2) = tmp_c10;
        kp.D(4,3) = tmp_c12;
        kp.D(4,4) = -2.25;
        kp.D(5,2) = tmp_c22;
        kp.D(5,3) = tmp_c23;
        kp.D(5,4) = -2.25;
        kp.D(6,2) = tmp_c23;
        kp.D(6,3) = tmp_c10;
        kp.D(6,4) = -0.94999999999999996;
        kp.D(7,2) = tmp_c12;
        kp.D(7,3) = tmp_c22;
        kp.D(7,4) = -0.94999999999999996;
        kp.D(8,0) = tmp_c17*tmp_c24;
        kp.D(8,1) = tmp_c18*tmp_c24;
        kp.D(8,2) = -tmp_c17*tmp_c24;
        kp.D(8,3) = -tmp_c18*tmp_c24;
        kp.D(8,4) = -1.0;
        kp.D(9,4) = -1;

    }

    // Legacy alias for hand-written code that still calls compute_constraints().
    template<typename T>
    static void compute_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        compute_qp_constraints(kp);
    }

    // --- 2.1 Compute True Constraints (g_true) ---
    template<typename T>
    static void compute_true_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {
        T p1x = kp.u(0);
        T p1y = kp.u(1);
        T p2x = kp.u(2);
        T p2y = kp.u(3);
        T alpha = kp.u(4);
        T x1 = kp.p(0);
        T y1 = kp.p(1);
        T theta1 = kp.p(2);
        T x2 = kp.p(3);
        T y2 = kp.p(4);
        T theta2 = kp.p(5);


        // g_true
        kp.g_true(0,0) = -2.25*alpha + (p1x - x1)*cos(theta1) + (p1y - y1)*sin(theta1);
        kp.g_true(1,0) = -2.25*alpha - (p1x - x1)*cos(theta1) - (p1y - y1)*sin(theta1);
        kp.g_true(2,0) = -0.94999999999999996*alpha - (p1x - x1)*sin(theta1) + (p1y - y1)*cos(theta1);
        kp.g_true(3,0) = -0.94999999999999996*alpha + (p1x - x1)*sin(theta1) - (p1y - y1)*cos(theta1);
        kp.g_true(4,0) = -2.25*alpha + (p2x - x2)*cos(theta2) + (p2y - y2)*sin(theta2);
        kp.g_true(5,0) = -2.25*alpha - (p2x - x2)*cos(theta2) - (p2y - y2)*sin(theta2);
        kp.g_true(6,0) = -0.94999999999999996*alpha - (p2x - x2)*sin(theta2) + (p2y - y2)*cos(theta2);
        kp.g_true(7,0) = -0.94999999999999996*alpha + (p2x - x2)*sin(theta2) - (p2y - y2)*cos(theta2);
        kp.g_true(8,0) = -1.0*alpha + sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2) + 1.0e-10);
        kp.g_true(9,0) = -alpha;

    }

    // --- 2.5 Terminal Stage: x-only projection of QP/IPM constraints ---
    template<typename T>
    static void compute_terminal_qp_constraints(KnotPoint<T,NX,NU,NC,NP>& kp) {

        // --- Special Constraints Pre-Calculation ---


        // Clear generated output packets; nonzero entries are assigned below.
        kp.g_val.setZero();
        kp.C.setZero();
        kp.D.setZero();

        // g_val

        // C

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


        // Clear generated output packets; nonzero entries are assigned below.
        kp.g_true.setZero();

        // g_true

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
        T p1x = kp.u(0);
        T p1y = kp.u(1);
        T p2x = kp.u(2);
        T p2y = kp.u(3);
        T alpha = kp.u(4);
        T lam_8 = kp.lam(8);

        T tmp_j0 = p1y - p2y;
        T tmp_j1 = pow(tmp_j0, 2) + 1.0e-10;
        T tmp_j2 = p1x - p2x;
        T tmp_j3 = pow(tmp_j2, 2);
        T tmp_j4 = lam_8/pow(tmp_j1 + tmp_j3, 3.0/2.0);
        T tmp_j5 = tmp_j1*tmp_j4;
        T tmp_j6 = tmp_j0*tmp_j2*tmp_j4;
        T tmp_j7 = -tmp_j6;
        T tmp_j8 = -tmp_j1*tmp_j4;
        T tmp_j9 = tmp_j3 + 1.0e-10;
        T tmp_j10 = tmp_j4*tmp_j9;
        T tmp_j11 = -tmp_j4*tmp_j9;

        // Clear generated output packets; nonzero entries are assigned below.
        kp.q.setZero();

        // q

        // r
        kp.r(0,0) = 2.0e-8*p1x;
        kp.r(1,0) = 2.0e-8*p1y;
        kp.r(2,0) = 2.0e-8*p2x;
        kp.r(3,0) = 2.0e-8*p2y;
        kp.r(4,0) = 2.0e-8*alpha + 1;

        // Clear Hessian packets; nonzero entries are assigned below.
        kp.Q.setZero();
        kp.R.setZero();
        kp.H.setZero();

        // Q (Mode 0=GN, 1=Exact)

        // R (Mode 0=GN, 1=Exact)
        kp.R(0,0) = 2.0e-8;
        kp.R(1,1) = 2.0e-8;
        kp.R(2,2) = 2.0e-8;
        kp.R(3,3) = 2.0e-8;
        kp.R(4,4) = 2.0e-8;
        if constexpr (Mode == 1) kp.R(0,0) += tmp_j5;
        if constexpr (Mode == 1) kp.R(0,1) += tmp_j7;
        if constexpr (Mode == 1) kp.R(0,2) += tmp_j8;
        if constexpr (Mode == 1) kp.R(0,3) += tmp_j6;
        if constexpr (Mode == 1) kp.R(1,0) += tmp_j7;
        if constexpr (Mode == 1) kp.R(1,1) += tmp_j10;
        if constexpr (Mode == 1) kp.R(1,2) += tmp_j6;
        if constexpr (Mode == 1) kp.R(1,3) += tmp_j11;
        if constexpr (Mode == 1) kp.R(2,0) += tmp_j8;
        if constexpr (Mode == 1) kp.R(2,1) += tmp_j6;
        if constexpr (Mode == 1) kp.R(2,2) += tmp_j5;
        if constexpr (Mode == 1) kp.R(2,3) += tmp_j7;
        if constexpr (Mode == 1) kp.R(3,0) += tmp_j6;
        if constexpr (Mode == 1) kp.R(3,1) += tmp_j11;
        if constexpr (Mode == 1) kp.R(3,2) += tmp_j7;
        if constexpr (Mode == 1) kp.R(3,3) += tmp_j10;

        // H (Mode 0=GN, 1=Exact)

        kp.cost = 1.0e-8*pow(alpha, 2) + alpha + 1.0e-8*pow(p1x, 2) + 1.0e-8*pow(p1y, 2) + 1.0e-8*pow(p2x, 2) + 1.0e-8*pow(p2y, 2);
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


        // Clear generated output packets; nonzero entries are assigned below.
        kp.q.setZero();
        kp.r.setZero();

        // q

        // r

        // Clear Hessian packets; nonzero entries are assigned below.
        kp.Q.setZero();
        kp.R.setZero();
        kp.H.setZero();

        // terminal Q (Mode 0=GN, 1=Exact)

        // terminal R (Mode 0=GN, 1=Exact)

        // terminal H (Mode 0=GN, 1=Exact)

        kp.cost = 0;
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
        T p_0 = Vx(0);
        T A_0_0 = kp.A(0,0);

        // CSE Intermediate Variables

        // Accumulate Results
        kp.Q_bar(0,0) += pow(A_0_0, 2)*P_0_0;
        kp.q_bar(0,0) += A_0_0*p_0;

        // Fill Lower Triangles (Symmetry)
        kp.R_bar(1,0) = kp.R_bar(0,1);
        kp.R_bar(2,0) = kp.R_bar(0,2);
        kp.R_bar(3,0) = kp.R_bar(0,3);
        kp.R_bar(4,0) = kp.R_bar(0,4);
        kp.R_bar(2,1) = kp.R_bar(1,2);
        kp.R_bar(3,1) = kp.R_bar(1,3);
        kp.R_bar(4,1) = kp.R_bar(1,4);
        kp.R_bar(3,2) = kp.R_bar(2,3);
        kp.R_bar(4,2) = kp.R_bar(2,4);
        kp.R_bar(4,3) = kp.R_bar(3,4);

    }

};
}
