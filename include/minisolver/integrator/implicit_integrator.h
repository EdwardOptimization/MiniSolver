#pragma once

#include "minisolver/matrix/matrix_defs.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/integrator/newton_solver.h"
#include "minisolver/integrator/numerical_jacobian.h"

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace minisolver {

// SFINAE: detect if Model provides analytical continuous Jacobians
namespace detail {
    template <typename M, typename T, typename = void>
    struct has_jacobian_continuous : std::false_type {};

    template <typename M, typename T>
    struct has_jacobian_continuous<M, T,
        std::void_t<decltype(M::template jacobian_continuous<T>(
            std::declval<MSVec<T, M::NX>>(),
            std::declval<MSVec<T, M::NU>>(),
            std::declval<MSVec<T, M::NP>>()))>>
        : std::true_type {};

    template <typename M, typename T>
    inline constexpr bool has_jacobian_continuous_v =
        has_jacobian_continuous<M, T>::value;

    // SFINAE: detect if Model provides dynamics_continuous()
    template <typename M, typename T, typename = void>
    struct has_dynamics_continuous : std::false_type {};

    template <typename M, typename T>
    struct has_dynamics_continuous<M, T,
        std::void_t<decltype(M::template dynamics_continuous<T>(
            std::declval<MSVec<T, M::NX>>(),
            std::declval<MSVec<T, M::NU>>(),
            std::declval<MSVec<T, M::NP>>()))>>
        : std::true_type {};

    template <typename M, typename T>
    inline constexpr bool has_dynamics_continuous_v =
        has_dynamics_continuous<M, T>::value;

    inline bool is_implicit_integrator(IntegratorType type)
    {
        return type == IntegratorType::EULER_IMPLICIT
            || type == IntegratorType::RK2_IMPLICIT
            || type == IntegratorType::RK4_IMPLICIT;
    }
} // namespace detail

// Implicit integrator: Newton-based solvers for stiff or implicit ODEs.
// Stateless, all static methods. Model must provide dynamics_continuous().
template <typename Model>
class ImplicitIntegrator {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;

    using Knot = KnotPoint<double, NX, NU, NC, NP>;

    // Primary interface: replaces Model::compute_dynamics for implicit types.
    // Writes f_resid (x_next), A (df/dx_discrete), B (df/du_discrete) to kp.
    static void compute_dynamics(
        Knot& kp, IntegratorType type, double dt,
        const NewtonConfig& config = {})
    {
        switch (type) {
        case IntegratorType::EULER_IMPLICIT:
            backward_euler(kp, dt, config);
            break;
        case IntegratorType::RK2_IMPLICIT:
            implicit_midpoint(kp, dt, config);
            break;
        case IntegratorType::RK4_IMPLICIT:
            compute_gauss_legendre_2(kp, dt, config);
            break;
        default:
            throw std::invalid_argument("ImplicitIntegrator received unsupported integrator type");
        }
    }

    // Standalone integrate (for rollout_dynamics and line search rollout).
    static MSVec<double, NX> integrate(
        const MSVec<double, NX>& x, const MSVec<double, NU>& u,
        const MSVec<double, NP>& p, double dt, IntegratorType type,
        const NewtonConfig& config = {})
    {
        MSVec<double, NX> z = x;

        if (type == IntegratorType::RK2_IMPLICIT) {
            // Implicit midpoint
            auto eval = [&](const MSVec<double, NX>& z_in, MSVec<double, NX>& F,
                            MSMat<double, NX, NX>& J) {
                MSVec<double, NX> m = (x + z_in) * 0.5;
                auto jac = get_continuous_jacobians(m, u, p);
                F = z_in - x - Model::dynamics_continuous(m, u, p) * dt;
                J = MSMat<double, NX, NX>::Identity() - jac.Jx * (dt * 0.5);
            };
            NewtonSolver<double, NX> ns;
            if (!ns.solve(z, eval, config))
                return invalid_state();
        } else if (type == IntegratorType::RK4_IMPLICIT) {
            // Gauss-Legendre 2-stage: coupled 2*NX system
            constexpr int N2 = 2 * NX;
            constexpr double sqrt3 = 1.7320508075688772;
            constexpr double a11 = 0.25;
            constexpr double a12 = 0.25 - sqrt3 / 6.0;
            constexpr double a21 = 0.25 + sqrt3 / 6.0;
            constexpr double a22 = 0.25;

            MSVec<double, NX> f0 = Model::dynamics_continuous(x, u, p);
            MSVec<double, N2> K;
            K.template head<NX>() = f0;
            K.template tail<NX>() = f0;

            auto eval = [&](const MSVec<double, N2>& K_in, MSVec<double, N2>& F,
                            MSMat<double, N2, N2>& J) {
                auto k1 = K_in.template head<NX>();
                auto k2 = K_in.template tail<NX>();
                MSVec<double, NX> s1 = x + (k1 * a11 + k2 * a12) * dt;
                MSVec<double, NX> s2 = x + (k1 * a21 + k2 * a22) * dt;
                auto jac1 = get_continuous_jacobians(s1, u, p);
                auto jac2 = get_continuous_jacobians(s2, u, p);
                F.template head<NX>() = k1 - Model::dynamics_continuous(s1, u, p);
                F.template tail<NX>() = k2 - Model::dynamics_continuous(s2, u, p);
                MSMat<double, NX, NX> I_NX = MSMat<double, NX, NX>::Identity();
                J.template block<NX, NX>(0, 0) = I_NX - jac1.Jx * (dt * a11);
                J.template block<NX, NX>(0, NX) = -jac1.Jx * (dt * a12);
                J.template block<NX, NX>(NX, 0) = -jac2.Jx * (dt * a21);
                J.template block<NX, NX>(NX, NX) = I_NX - jac2.Jx * (dt * a22);
            };

            NewtonSolver<double, N2> ns;
            if (!ns.solve(K, eval, config))
                return invalid_state();
            auto k1 = K.template head<NX>();
            auto k2 = K.template tail<NX>();
            z = x + (k1 + k2) * (dt * 0.5);
        } else if (type == IntegratorType::EULER_IMPLICIT) {
            // Backward Euler (default)
            auto eval = [&](const MSVec<double, NX>& z_in, MSVec<double, NX>& F,
                            MSMat<double, NX, NX>& J) {
                auto jac = get_continuous_jacobians(z_in, u, p);
                F = z_in - x - Model::dynamics_continuous(z_in, u, p) * dt;
                J = MSMat<double, NX, NX>::Identity() - jac.Jx * dt;
            };
            NewtonSolver<double, NX> ns;
            if (!ns.solve(z, eval, config))
                return invalid_state();
        } else {
            throw std::invalid_argument("ImplicitIntegrator received unsupported integrator type");
        }
        return z;
    }

private:
    // Get continuous Jacobians: analytical if available, numerical otherwise.
    static ContinuousJacobians<double, NX, NU> get_continuous_jacobians(
        const MSVec<double, NX>& x,
        const MSVec<double, NU>& u,
        const MSVec<double, NP>& p)
    {
        if constexpr (detail::has_jacobian_continuous_v<Model, double>) {
            return Model::jacobian_continuous(x, u, p);
        } else {
            return compute_numerical_jacobian<Model, double>(x, u, p);
        }
    }

    // Compute M^{-1} via one LU factorization with matrix RHS.
    // M = I - dt*Jx is not symmetric in general, so LLT is invalid.
    static bool invert_matrix(
        const MSMat<double, NX, NX>& M, MSMat<double, NX, NX>& M_inv)
    {
        MSMat<double, NX, NX> I = MSMat<double, NX, NX>::Identity();
        return MatOps::lu_solve_matrix(M, I, M_inv);
    }

    static void mark_jacobians_invalid(Knot& kp)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NX; ++j)
                kp.A(i, j) = nan;
            for (int j = 0; j < NU; ++j)
                kp.B(i, j) = nan;
        }
    }

    static MSVec<double, NX> invalid_state()
    {
        MSVec<double, NX> z;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (int i = 0; i < NX; ++i)
            z(i) = nan;
        return z;
    }

    static void mark_dynamics_invalid(Knot& kp)
    {
        kp.f_resid = invalid_state();
        mark_jacobians_invalid(kp);
    }

    // --- Backward Euler ---
    // Solve: z = x + dt * f(z, u)
    // Jacobians: A = (I - dt*Jf)^{-1}, B = A * dt * Ju
    static void backward_euler(
        Knot& kp, double dt, const NewtonConfig& config)
    {
        const auto& x = kp.x;
        const auto& u = kp.u;
        const auto& p = kp.p;

        MSVec<double, NX> z = x; // initial guess

        auto eval = [&](const MSVec<double, NX>& z_in, MSVec<double, NX>& F,
                        MSMat<double, NX, NX>& J) {
            auto jac = get_continuous_jacobians(z_in, u, p);
            F = z_in - x - Model::dynamics_continuous(z_in, u, p) * dt;
            J = MSMat<double, NX, NX>::Identity() - jac.Jx * dt;
        };

        NewtonSolver<double, NX> ns;
        if (!ns.solve(z, eval, config)) {
            mark_dynamics_invalid(kp);
            return;
        }

        // Compute Jacobians at converged point
        auto jac = get_continuous_jacobians(z, u, p);

        kp.f_resid = z;

        // A = (I - dt * Jf)^{-1}
        MSMat<double, NX, NX> M = MSMat<double, NX, NX>::Identity() - jac.Jx * dt;
        MSMat<double, NX, NX> M_inv;
        if (invert_matrix(M, M_inv)) {
            kp.A = M_inv;
            kp.B.noalias() = M_inv * (jac.Ju * dt);
        } else {
            mark_jacobians_invalid(kp);
        }
    }

    // --- Implicit Midpoint ---
    // Solve: z = x + dt * f((x+z)/2, u)
    // Jacobians at midpoint m = (x+z)/2:
    //   A = (I - dt/2 * Jf(m))^{-1} * (I + dt/2 * Jf(m))
    //   B = (I - dt/2 * Jf(m))^{-1} * dt * Ju(m)
    static void implicit_midpoint(
        Knot& kp, double dt, const NewtonConfig& config)
    {
        const auto& x = kp.x;
        const auto& u = kp.u;
        const auto& p = kp.p;

        MSVec<double, NX> z = x;

        auto eval = [&](const MSVec<double, NX>& z_in, MSVec<double, NX>& F,
                        MSMat<double, NX, NX>& J) {
            MSVec<double, NX> m = (x + z_in) * 0.5;
            auto jac = get_continuous_jacobians(m, u, p);
            F = z_in - x - Model::dynamics_continuous(m, u, p) * dt;
            J = MSMat<double, NX, NX>::Identity() - jac.Jx * (dt * 0.5);
        };

        NewtonSolver<double, NX> ns;
        if (!ns.solve(z, eval, config)) {
            mark_dynamics_invalid(kp);
            return;
        }

        // Jacobians at midpoint
        MSVec<double, NX> m = (x + z) * 0.5;
        auto jac = get_continuous_jacobians(m, u, p);

        kp.f_resid = z;

        MSMat<double, NX, NX> M_minus = MSMat<double, NX, NX>::Identity() - jac.Jx * (dt * 0.5);
        MSMat<double, NX, NX> M_plus  = MSMat<double, NX, NX>::Identity() + jac.Jx * (dt * 0.5);
        MSMat<double, NX, NX> M_minus_inv;
        if (invert_matrix(M_minus, M_minus_inv)) {
            kp.A.noalias() = M_minus_inv * M_plus;
            kp.B.noalias() = M_minus_inv * (jac.Ju * dt);
        } else {
            mark_jacobians_invalid(kp);
        }
    }

    // --- Gauss-Legendre 2-stage (RK4 Implicit, order 4) ---
    // Butcher tableau:
    //   c1=1/2-√3/6, c2=1/2+√3/6
    //   a11=1/4, a12=1/4-√3/6, a21=1/4+√3/6, a22=1/4
    //   b1=1/2, b2=1/2
    //
    // Coupled system of 2*NX unknowns: K = [k1; k2]
    //   k1 = f(x + dt*(a11*k1 + a12*k2), u)
    //   k2 = f(x + dt*(a21*k1 + a22*k2), u)
    // x_next = x + dt*(b1*k1 + b2*k2)
    static void compute_gauss_legendre_2(
        Knot& kp, double dt, const NewtonConfig& config)
    {
        constexpr int N2 = 2 * NX;
        constexpr double sqrt3 = 1.7320508075688772;
        constexpr double a11 = 0.25;
        constexpr double a12 = 0.25 - sqrt3 / 6.0;
        constexpr double a21 = 0.25 + sqrt3 / 6.0;
        constexpr double a22 = 0.25;

        const auto& x = kp.x;
        const auto& u = kp.u;
        const auto& p = kp.p;

        // Initial guess: k1 = k2 = f(x, u)
        MSVec<double, NX> f0 = Model::dynamics_continuous(x, u, p);
        MSVec<double, N2> K;
        K.template head<NX>() = f0;
        K.template tail<NX>() = f0;

        auto eval = [&](const MSVec<double, N2>& K_in, MSVec<double, N2>& F,
                        MSMat<double, N2, N2>& J) {
            auto k1 = K_in.template head<NX>();
            auto k2 = K_in.template tail<NX>();

            MSVec<double, NX> s1 = x + (k1 * a11 + k2 * a12) * dt;
            MSVec<double, NX> s2 = x + (k1 * a21 + k2 * a22) * dt;

            auto jac1 = get_continuous_jacobians(s1, u, p);
            auto jac2 = get_continuous_jacobians(s2, u, p);

            MSVec<double, NX> f1 = Model::dynamics_continuous(s1, u, p);
            MSVec<double, NX> f2 = Model::dynamics_continuous(s2, u, p);

            F.template head<NX>() = k1 - f1;
            F.template tail<NX>() = k2 - f2;

            // J = [I - dt*a11*Jx1,  -dt*a12*Jx1]
            //     [-dt*a21*Jx2,      I - dt*a22*Jx2]
            MSMat<double, NX, NX> I_NX = MSMat<double, NX, NX>::Identity();
            J.template block<NX, NX>(0, 0) = I_NX - jac1.Jx * (dt * a11);
            J.template block<NX, NX>(0, NX) = -jac1.Jx * (dt * a12);
            J.template block<NX, NX>(NX, 0) = -jac2.Jx * (dt * a21);
            J.template block<NX, NX>(NX, NX) = I_NX - jac2.Jx * (dt * a22);
        };

        NewtonSolver<double, N2> ns;
        if (!ns.solve(K, eval, config)) {
            mark_dynamics_invalid(kp);
            return;
        }

        auto k1 = K.template head<NX>();
        auto k2 = K.template tail<NX>();

        kp.f_resid = x + (k1 + k2) * (dt * 0.5);

        // Discrete Jacobians via implicit function theorem.
        // At convergence dR/dK * dK/dx + dR/dx = 0, so dK/dx = -JK^{-1} * Jx_RHS.
        // Then A = I + dt*0.5*(dk1/dx + dk2/dx).
        MSVec<double, NX> s1 = x + (k1 * a11 + k2 * a12) * dt;
        MSVec<double, NX> s2 = x + (k1 * a21 + k2 * a22) * dt;
        auto jac1 = get_continuous_jacobians(s1, u, p);
        auto jac2 = get_continuous_jacobians(s2, u, p);

        // Build JK (same structure as Newton Jacobian)
        MSMat<double, N2, N2> JK;
        MSMat<double, NX, NX> I_NX = MSMat<double, NX, NX>::Identity();
        JK.template block<NX, NX>(0, 0) = I_NX - jac1.Jx * (dt * a11);
        JK.template block<NX, NX>(0, NX) = -jac1.Jx * (dt * a12);
        JK.template block<NX, NX>(NX, 0) = -jac2.Jx * (dt * a21);
        JK.template block<NX, NX>(NX, NX) = I_NX - jac2.Jx * (dt * a22);

        // Solve JK * dK = RHS via one LU factorization (JK is not symmetric).
        MSMat<double, N2, NX> RHS_x;
        RHS_x.template block<NX, NX>(0, 0) = jac1.Jx;
        RHS_x.template block<NX, NX>(NX, 0) = jac2.Jx;

        MSMat<double, N2, NX> dK_dx;
        if (!MatOps::lu_solve_matrix(JK, RHS_x, dK_dx)) {
            mark_jacobians_invalid(kp);
            return;
        }

        // A = I + dt*0.5*(dk1/dx + dk2/dx)
        kp.A = I_NX + (dK_dx.template topRows<NX>() + dK_dx.template bottomRows<NX>()) * (dt * 0.5);

        // Solve for dK/du
        MSMat<double, N2, NU> RHS_u;
        RHS_u.template block<NX, NU>(0, 0) = jac1.Ju;
        RHS_u.template block<NX, NU>(NX, 0) = jac2.Ju;

        MSMat<double, N2, NU> dK_du;
        if (!MatOps::lu_solve_matrix(JK, RHS_u, dK_du)) {
            mark_jacobians_invalid(kp);
            return;
        }

        kp.B.noalias() = (dK_du.template topRows<NX>() + dK_du.template bottomRows<NX>()) * (dt * 0.5);
    }
};

// Dispatch: route to implicit or explicit integrator based on type.
// If the model doesn't provide dynamics_continuous(), implicit types
// fall back to the model's compute_dynamics (which may be explicit or
// a generated approximation).
namespace detail {
    template <typename Model, typename Knot>
    void dispatch_compute_dynamics(
        Knot& kp, IntegratorType type, double dt,
        const NewtonConfig& newton_config = {})
    {
        if constexpr (has_dynamics_continuous_v<Model, double>) {
            if (is_implicit_integrator(type)) {
                ImplicitIntegrator<Model>::compute_dynamics(kp, type, dt, newton_config);
                return;
            }
        }
        Model::compute_dynamics(kp, type, dt);
    }

    template <typename Model>
    MSVec<double, Model::NX> dispatch_integrate(
        const MSVec<double, Model::NX>& x,
        const MSVec<double, Model::NU>& u,
        const MSVec<double, Model::NP>& p,
        double dt, IntegratorType type,
        const NewtonConfig& newton_config = {})
    {
        if constexpr (has_dynamics_continuous_v<Model, double>) {
            if (is_implicit_integrator(type)) {
                return ImplicitIntegrator<Model>::integrate(x, u, p, dt, type, newton_config);
            }
        }
        return Model::integrate(x, u, p, dt, type);
    }
} // namespace detail

} // namespace minisolver
