#include "minisolver/algorithms/model_evaluation.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/integrator/implicit_integrator.h"
#include "minisolver/integrator/numerical_jacobian.h"
#include "minisolver/matrix/matrix_defs.h"
#include <array>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

// Define a simple nonlinear model: dx/dt = -x^2
// Exact solution: x(t) = 1 / (1/x0 + t)
struct NonlinearDecayModel {
    static const int NX = 1;
    static const int NU = 1; // Dummy
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in, const MSVec<T, NU>& /*u_in*/, const MSVec<T, NP>& /*p_in*/)
    {
        T x = x_in(0);
        MSVec<T, NX> xdot;
        xdot(0) = -x * x;
        return xdot;
    }

    // Standard Integrator Implementation (copied from generated code pattern)
    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& p, double dt, IntegratorType type)
    {
        switch (type) {
        case IntegratorType::EULER_EXPLICIT:
            return x + dynamics_continuous(x, u, p) * dt;

        case IntegratorType::RK2_EXPLICIT: {
            auto k1 = dynamics_continuous(x, u, p);
            auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u, p);
            return x + k2 * dt;
        }

        case IntegratorType::EULER_IMPLICIT: {
            // Simple Fixed-Point Iteration
            MSVec<T, NX> x_next = x; // Guess
            for (int i = 0; i < 10; ++i) {
                x_next = x + dynamics_continuous(x_next, u, p) * dt;
            }
            return x_next;
        }

        case IntegratorType::RK2_IMPLICIT: {
            // Implicit Midpoint
            MSVec<T, NX> k = dynamics_continuous(x, u, p); // Guess k0
            for (int i = 0; i < 10; ++i) {
                k = dynamics_continuous<T>(x + k * (0.5 * dt), u, p);
            }
            return x + k * dt;
        }

        case IntegratorType::RK4_EXPLICIT:
        default: {
            auto k1 = dynamics_continuous(x, u, p);
            auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u, p);
            auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u, p);
            auto k4 = dynamics_continuous<T>(x + k3 * dt, u, p);
            return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        }

        case IntegratorType::RK4_IMPLICIT: {
            // Gauss-Legendre RK4 (Implicit) is complex to implement generically without Butcher
            // tableau. The generated code usually maps RK4_IMPLICIT to RK4_EXPLICIT or a specific
            // implicit scheme. For this test, we assume it falls back to explicit or uses a simple
            // implementation if available. In CarModel it was mapped to same block as Explicit.
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

struct ExplicitOnlyDispatchModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> out;
        out(0) = x(0) + u(0) * dt;
        return out;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        kp.f_resid = integrate(kp.x, kp.u, kp.p, dt, type);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }
};

struct TerminalDynamicsCounterModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static inline int dynamics_calls = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        ++dynamics_calls;
        kp.f_resid = integrate(kp.x, kp.u, kp.p, dt, type);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.Q(0, 0) = 2.0;
        kp.r(0) = 0.0;
        kp.R(0, 0) = 0.0;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

TEST(ImplicitIntegratorTest, DispatchRejectsImplicitWithoutContinuousDynamics)
{
    KnotPoint<double, ExplicitOnlyDispatchModel::NX, ExplicitOnlyDispatchModel::NU,
        ExplicitOnlyDispatchModel::NC, ExplicitOnlyDispatchModel::NP>
        kp;
    kp.set_zero();
    kp.u(0) = 1.0;

    EXPECT_THROW(detail::dispatch_compute_dynamics<ExplicitOnlyDispatchModel>(
                     kp, IntegratorType::EULER_IMPLICIT, 0.1),
        std::invalid_argument);

    MSVec<double, 1> x;
    x(0) = 0.0;
    MSVec<double, 1> u;
    u(0) = 1.0;
    MSVec<double, 0> p;
    EXPECT_THROW(detail::dispatch_integrate<ExplicitOnlyDispatchModel>(
                     x, u, p, 0.1, IntegratorType::RK2_IMPLICIT),
        std::invalid_argument);
}

struct LargeScaleLinearJacobianModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/)
    {
        MSVec<T, NX> xdot;
        xdot(0) = static_cast<T>(3.0) * x(0) + static_cast<T>(2.0) * u(0);
        return xdot;
    }
};

TEST(NumericalJacobianTest, UsesScaleAwarePerturbationForLargeStates)
{
    MSVec<double, 1> x;
    x(0) = 1e12;
    MSVec<double, 1> u;
    u(0) = -1e12;
    MSVec<double, 0> p;

    auto jac = compute_numerical_jacobian<LargeScaleLinearJacobianModel, double>(x, u, p);

    EXPECT_NEAR(jac.Jx(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(jac.Ju(0, 0), 2.0, 1e-9);
}

struct SingularImplicitJacobianModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x, const MSVec<T, NU>& /*u*/, const MSVec<T, NP>& /*p*/)
    {
        MSVec<T, NX> xdot;
        xdot(0) = x(0);
        return xdot;
    }

    template <typename T>
    static ContinuousJacobians<T, NX, NU> jacobian_continuous(
        const MSVec<T, NX>& /*x*/, const MSVec<T, NU>& /*u*/, const MSVec<T, NP>& /*p*/)
    {
        ContinuousJacobians<T, NX, NU> jac;
        jac.Jx(0, 0) = static_cast<T>(1.0);
        jac.Ju(0, 0) = static_cast<T>(0.0);
        return jac;
    }
};

TEST(ImplicitIntegratorTest, SingularJacobianMarksDynamicsInvalid)
{
    KnotPoint<double, 1, 1, 0, 0> kp;
    kp.set_zero();
    kp.x(0) = 1.0;

    ImplicitIntegrator<SingularImplicitJacobianModel>::compute_dynamics(
        kp, IntegratorType::EULER_IMPLICIT, 1.0);

    EXPECT_TRUE(MatOps::has_nan(kp.A));
    EXPECT_TRUE(MatOps::has_nan(kp.B));
}

TEST(ImplicitIntegratorTest, RejectsUnsupportedDirectIntegratorType)
{
    KnotPoint<double, 1, 1, 0, 0> kp;
    kp.set_zero();
    MSVec<double, 1> x;
    x(0) = 1.0;
    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 0> p;

    EXPECT_THROW(ImplicitIntegrator<NonlinearDecayModel>::compute_dynamics(
                     kp, IntegratorType::EULER_EXPLICIT, 0.1),
        std::invalid_argument);
    EXPECT_THROW(ImplicitIntegrator<NonlinearDecayModel>::integrate(
                     x, u, p, 0.1, IntegratorType::EULER_EXPLICIT),
        std::invalid_argument);
}

TEST(ImplicitIntegratorTest, FailedNewtonSolveInvalidatesDynamics)
{
    KnotPoint<double, 1, 1, 0, 0> kp;
    kp.set_zero();
    kp.x(0) = 1.0;

    NewtonConfig cfg;
    cfg.max_iters = 1;
    cfg.tol = 1e-14;

    ImplicitIntegrator<NonlinearDecayModel>::compute_dynamics(
        kp, IntegratorType::EULER_IMPLICIT, 1.0, cfg);

    EXPECT_TRUE(MatOps::has_nan(kp.f_resid));
    EXPECT_TRUE(MatOps::has_nan(kp.A));
    EXPECT_TRUE(MatOps::has_nan(kp.B));
}

TEST(ImplicitIntegratorTest, FailedNewtonSolveInvalidatesStandaloneIntegrate)
{
    MSVec<double, 1> x;
    x(0) = 1.0;
    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 0> p;

    NewtonConfig cfg;
    cfg.max_iters = 1;
    cfg.tol = 1e-14;

    auto z = ImplicitIntegrator<NonlinearDecayModel>::integrate(
        x, u, p, 1.0, IntegratorType::EULER_IMPLICIT, cfg);

    EXPECT_TRUE(MatOps::has_nan(z));
}

TEST(IntegratorTest, AccuracyComparison)
{
    double dt = 0.1;
    double x0_val = 1.0;
    double t_end = 1.0;
    int steps = static_cast<int>(t_end / dt);

    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 0> p;

    // 1. Exact Solution
    double x_exact = 1.0 / (1.0 / x0_val + t_end); // 1 / (1 + 1) = 0.5

    // 2. Euler Explicit
    MSVec<double, 1> x_ee;
    x_ee(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_ee = NonlinearDecayModel::integrate(x_ee, u, p, dt, IntegratorType::EULER_EXPLICIT);
    }
    double err_ee = std::abs(x_ee(0) - x_exact);

    // 3. RK4 Explicit
    MSVec<double, 1> x_rk4;
    x_rk4(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_rk4 = NonlinearDecayModel::integrate(x_rk4, u, p, dt, IntegratorType::RK4_EXPLICIT);
    }
    double err_rk4 = std::abs(x_rk4(0) - x_exact);

    // 4. Euler Implicit
    MSVec<double, 1> x_ei;
    x_ei(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_ei = NonlinearDecayModel::integrate(x_ei, u, p, dt, IntegratorType::EULER_IMPLICIT);
    }
    double err_ei = std::abs(x_ei(0) - x_exact);

    // Verify Accuracy Hierarchy: RK4 > Euler
    EXPECT_LT(err_rk4, err_ee);
    EXPECT_LT(err_rk4, 1e-5); // RK4 should be very accurate

    // Implicit vs Explicit Euler on this stable decaying system
    // Euler explicit: x_{k+1} = x_k - dt*x_k^2
    // Euler implicit: x_{k+1} = x_k - dt*x_{k+1}^2  => dt*x^2 + x - x_k = 0
    // Implicit usually has different error characteristics.
    // For x' = -x^2, Implicit is actually slightly less accurate than explicit in early steps but
    // more stable? Let's just check they are reasonable.
    EXPECT_LT(err_ee, 0.1);
    EXPECT_LT(err_ei, 0.1);

    // RK2 Explicit
    MSVec<double, 1> x_rk2;
    x_rk2(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_rk2 = NonlinearDecayModel::integrate(x_rk2, u, p, dt, IntegratorType::RK2_EXPLICIT);
    }
    double err_rk2 = std::abs(x_rk2(0) - x_exact);

    EXPECT_LT(err_rk2, err_ee);
    EXPECT_LT(err_rk4, err_rk2);
}

// =============================================================================
// Stiff ODE model: dx/dt = -lambda*x (large lambda → stiff)
// Exact: x(t) = x0 * exp(-lambda*t)
// Explicit Euler unstable for dt > 2/lambda. Implicit Euler unconditionally stable.
// =============================================================================
struct StiffDecayModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 1; // p(0) = lambda

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "lambda" };
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in, const MSVec<T, NU>& /*u*/, const MSVec<T, NP>& p)
    {
        MSVec<T, NX> xdot;
        xdot(0) = -p(0) * x_in(0); // dx/dt = -lambda * x
        return xdot;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        auto xn = detail::dispatch_integrate<StiffDecayModel>(kp.x, kp.u, kp.p, dt, type);
        kp.f_resid = xn;
        kp.A(0, 0) = 1.0; // placeholder
        kp.B(0, 0) = 0.0;
    }

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& p, double dt, IntegratorType type)
    {
        return detail::dispatch_integrate<StiffDecayModel>(x, u, p, dt, type);
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }
    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// --- Newton solver convergence ---
TEST(ImplicitIntegratorTest, NewtonSolverConvergence)
{
    // Solve f(x) = x^2 - 2 = 0, root = sqrt(2)
    MSVec<double, 1> x;
    x(0) = 1.0; // initial guess

    auto eval = [](const MSVec<double, 1>& x_in, MSVec<double, 1>& F, MSMat<double, 1, 1>& J) {
        F(0) = x_in(0) * x_in(0) - 2.0;
        J(0, 0) = 2.0 * x_in(0);
    };

    NewtonConfig cfg;
    cfg.tol = 1e-12;
    NewtonSolver<double, 1> ns;
    bool converged = ns.solve(x, eval, cfg);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(x(0), std::sqrt(2.0), 1e-10);
}

TEST(ImplicitIntegratorTest, NewtonDetectsConvergenceAfterFinalStep)
{
    MSVec<double, 1> x;
    x(0) = 0.0;

    NewtonConfig cfg;
    cfg.max_iters = 1;
    cfg.tol = 1e-12;
    cfg.regularization = 0.0;

    NewtonSolver<double, 1> ns;
    bool ok = ns.solve(
        x,
        [](const MSVec<double, 1>& z, MSVec<double, 1>& F, MSMat<double, 1, 1>& J) {
            F(0) = z(0) - 1.0;
            J(0, 0) = 1.0;
        },
        cfg, false);

    EXPECT_TRUE(ok);
    EXPECT_NEAR(x(0), 1.0, 1e-12);
}

TEST(ImplicitIntegratorTest, NewtonDoesNotDampSolvableIllConditionedJacobian)
{
    MSVec<double, 2> x;
    x(0) = 0.0;
    x(1) = 0.0;

    const double eps = 1e-12;
    MSVec<double, 2> root;
    root(0) = 1.0;
    root(1) = -1.0;

    NewtonConfig cfg;
    cfg.max_iters = 6;
    cfg.tol = 1e-18;
    cfg.regularization = 1e-12;

    auto eval = [&](const MSVec<double, 2>& z, MSVec<double, 2>& F, MSMat<double, 2, 2>& J) {
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
        J(1, 0) = 1.0;
        J(1, 1) = 1.0 + eps;

        const MSVec<double, 2> e = z - root;
        F = J * e;
    };

    NewtonSolver<double, 2> ns;
    const bool ok = ns.solve(x, eval, cfg, false);

    EXPECT_TRUE(ok);
    EXPECT_NEAR(x(0), root(0), 1e-9);
    EXPECT_NEAR(x(1), root(1), 1e-9);
}

// --- Backward Euler accuracy: O(dt^2) ---
TEST(ImplicitIntegratorTest, BackwardEulerAccuracy)
{
    double lambda = 2.0;
    double dt = 0.01;
    double x0_val = 1.0;
    double t_end = 1.0;
    int steps = static_cast<int>(t_end / dt);

    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 1> p;
    p(0) = lambda;

    // Backward Euler via ImplicitIntegrator
    MSVec<double, 1> x_be;
    x_be(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_be = ImplicitIntegrator<StiffDecayModel>::integrate(
            x_be, u, p, dt, IntegratorType::EULER_IMPLICIT);
    }

    double x_exact = x0_val * std::exp(-lambda * t_end);
    double err = std::abs(x_be(0) - x_exact);

    std::cerr << "[ImplicitIntegrator] Backward Euler: x=" << x_be(0) << " exact=" << x_exact
              << " err=" << err << "\n";

    // Backward Euler is O(dt) for nonlinear, O(dt) for linear stiff.
    // For this simple linear case, error should be small.
    EXPECT_LT(err, 0.01);
}

// --- Implicit Midpoint accuracy: O(dt^3) for linear ---
TEST(ImplicitIntegratorTest, ImplicitMidpointAccuracy)
{
    double lambda = 2.0;
    double dt = 0.01;
    double x0_val = 1.0;
    double t_end = 1.0;
    int steps = static_cast<int>(t_end / dt);

    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 1> p;
    p(0) = lambda;

    MSVec<double, 1> x_im;
    x_im(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_im = ImplicitIntegrator<StiffDecayModel>::integrate(
            x_im, u, p, dt, IntegratorType::RK2_IMPLICIT);
    }

    double x_exact = x0_val * std::exp(-lambda * t_end);
    double err = std::abs(x_im(0) - x_exact);

    std::cerr << "[ImplicitIntegrator] Implicit Midpoint: x=" << x_im(0) << " exact=" << x_exact
              << " err=" << err << "\n";

    // Midpoint is O(dt^2) — should be more accurate than backward Euler
    EXPECT_LT(err, 0.001);
}

// --- Stiff stability: explicit fails, implicit succeeds ---
TEST(ImplicitIntegratorTest, StiffStabilityExplicitFailsImplicitSucceeds)
{
    double lambda = 1000.0; // very stiff
    double dt = 0.005; // dt > 2/lambda = 0.002 → explicit unstable
    double x0_val = 1.0;
    int steps = 20;

    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 1> p;
    p(0) = lambda;

    // Explicit Euler: should blow up (unstable).
    // Use simple forward Euler formula directly (not ImplicitIntegrator).
    MSVec<double, 1> x_exp;
    x_exp(0) = x0_val;
    bool exp_finite = true;
    for (int k = 0; k < steps; ++k) {
        // Forward Euler: x_{k+1} = x_k + dt * f(x_k)
        auto f = StiffDecayModel::dynamics_continuous(x_exp, u, p);
        x_exp = x_exp + f * dt;
        if (!std::isfinite(x_exp(0)) || std::abs(x_exp(0)) > 1e10) {
            exp_finite = false;
            break;
        }
    }

    // Backward Euler: should stay stable
    MSVec<double, 1> x_imp;
    x_imp(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_imp = ImplicitIntegrator<StiffDecayModel>::integrate(
            x_imp, u, p, dt, IntegratorType::EULER_IMPLICIT);
    }

    double x_exact = x0_val * std::exp(-lambda * steps * dt);

    std::cerr << "[ImplicitIntegrator] Stiff stability:\n"
              << "  explicit: " << x_exp(0) << (exp_finite ? " (finite)" : " (DIVERGED)") << "\n"
              << "  implicit: " << x_imp(0) << "\n"
              << "  exact:    " << x_exact << "\n";

    // Explicit should diverge or be wildly inaccurate
    EXPECT_FALSE(exp_finite || std::abs(x_exp(0) - x_exact) < 1.0)
        << "Explicit Euler should be unstable for stiff ODE";

    // Implicit should be close to exact
    EXPECT_NEAR(x_imp(0), x_exact, 0.1) << "Implicit Euler should handle stiff ODE";
}

// --- A/B Jacobian verification ---
TEST(ImplicitIntegratorTest, JacobianAccuracy)
{
    // Verify A from ImplicitIntegrator against finite-difference of discrete map.
    // NonlinearDecayModel has NC=0, NP=0.
    using Knot = KnotPoint<double, 1, 1, 0, 0>;

    Knot kp;
    kp.set_zero();
    kp.x(0) = 0.5;
    kp.u(0) = 0.0;

    double dt = 0.1;
    NewtonConfig cfg;
    cfg.tol = 1e-12;

    // Compute discrete Jacobian via finite difference of the integrate map
    auto x_next_fn = [&](double x_val) -> double {
        MSVec<double, 1> x_in;
        x_in(0) = x_val;
        MSVec<double, 1> u_in;
        u_in(0) = 0.0;
        MSVec<double, 0> p_in;
        auto z = ImplicitIntegrator<NonlinearDecayModel>::integrate(
            x_in, u_in, p_in, dt, IntegratorType::EULER_IMPLICIT, cfg);
        return z(0);
    };

    double eps = 1e-6;
    double x0 = kp.x(0);
    double A_numerical = (x_next_fn(x0 + eps) - x_next_fn(x0 - eps)) / (2.0 * eps);

    // Compute A via ImplicitIntegrator::compute_dynamics
    ImplicitIntegrator<NonlinearDecayModel>::compute_dynamics(
        kp, IntegratorType::EULER_IMPLICIT, dt, cfg);

    std::cerr << "[ImplicitIntegrator] Jacobian check:\n"
              << "  A_numerical = " << A_numerical << "\n"
              << "  A_analytical = " << kp.A(0, 0) << "\n";

    EXPECT_NEAR(kp.A(0, 0), A_numerical, 1e-4)
        << "Analytical A should match numerical discrete Jacobian";
}

struct ControlledImplicitJacobianModel {
    static const int NX = 2;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x0", "x1" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/)
    {
        MSVec<T, NX> xdot;
        xdot(0) = -static_cast<T>(0.2) * x(0) * x(0) + std::sin(x(1)) + static_cast<T>(0.7) * u(0);
        xdot(1) = x(0) * x(1) + static_cast<T>(0.1) * u(0) * u(0) - static_cast<T>(0.3) * x(1);
        return xdot;
    }

    template <typename T>
    static ContinuousJacobians<T, NX, NU> jacobian_continuous(
        const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/)
    {
        ContinuousJacobians<T, NX, NU> jac;
        jac.Jx(0, 0) = -static_cast<T>(0.4) * x(0);
        jac.Jx(0, 1) = std::cos(x(1));
        jac.Jx(1, 0) = x(1);
        jac.Jx(1, 1) = x(0) - static_cast<T>(0.3);
        jac.Ju(0, 0) = static_cast<T>(0.7);
        jac.Ju(1, 0) = static_cast<T>(0.2) * u(0);
        return jac;
    }

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& p, double dt, IntegratorType /*type*/)
    {
        return x + dynamics_continuous(x, u, p) * dt;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        kp.f_resid = integrate(kp.x, kp.u, kp.p, dt, type);
        auto jac = jacobian_continuous(kp.x, kp.u, kp.p);
        kp.A = MSMat<T, NX, NX>::Identity() + jac.Jx * dt;
        kp.B = jac.Ju * dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0) + kp.x(1) * kp.x(1) + kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(2.0) * kp.x(0);
        kp.q(1) = static_cast<T>(2.0) * kp.x(1);
        kp.r(0) = static_cast<T>(2.0) * kp.u(0);
        kp.Q.setZero();
        kp.Q(0, 0) = 2.0;
        kp.Q(1, 1) = 2.0;
        kp.R(0, 0) = 2.0;
        kp.H.setZero();
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

TEST(ImplicitIntegratorTest, JacobiansMatchFiniteDifferenceForAllImplicitSchemes)
{
    using Model = ControlledImplicitJacobianModel;
    using Knot = KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>;

    MSVec<double, Model::NX> x;
    x(0) = 0.4;
    x(1) = -0.2;
    MSVec<double, Model::NU> u;
    u(0) = 0.3;
    MSVec<double, Model::NP> p;

    const double dt = 0.03;
    const double eps = 1e-6;
    NewtonConfig cfg;
    cfg.max_iters = 30;
    cfg.tol = 1e-13;

    const std::array<IntegratorType, 3> implicit_types = {
        IntegratorType::EULER_IMPLICIT,
        IntegratorType::RK2_IMPLICIT,
        IntegratorType::RK4_IMPLICIT,
    };

    for (IntegratorType type : implicit_types) {
        Knot kp;
        kp.set_zero();
        kp.x = x;
        kp.u = u;
        ImplicitIntegrator<Model>::compute_dynamics(kp, type, dt, cfg);

        ASSERT_TRUE(kp.f_resid.allFinite()) << static_cast<int>(type);
        ASSERT_TRUE(kp.A.allFinite()) << static_cast<int>(type);
        ASSERT_TRUE(kp.B.allFinite()) << static_cast<int>(type);

        for (int j = 0; j < Model::NX; ++j) {
            auto x_plus = x;
            auto x_minus = x;
            x_plus(j) += eps;
            x_minus(j) -= eps;
            const auto f_plus = ImplicitIntegrator<Model>::integrate(x_plus, u, p, dt, type, cfg);
            const auto f_minus = ImplicitIntegrator<Model>::integrate(x_minus, u, p, dt, type, cfg);

            for (int i = 0; i < Model::NX; ++i) {
                const double fd = (f_plus(i) - f_minus(i)) / (2.0 * eps);
                EXPECT_NEAR(kp.A(i, j), fd, 2e-5)
                    << "A mismatch for integrator " << static_cast<int>(type) << " at (" << i << ","
                    << j << ")";
            }
        }

        for (int j = 0; j < Model::NU; ++j) {
            auto u_plus = u;
            auto u_minus = u;
            u_plus(j) += eps;
            u_minus(j) -= eps;
            const auto f_plus = ImplicitIntegrator<Model>::integrate(x, u_plus, p, dt, type, cfg);
            const auto f_minus = ImplicitIntegrator<Model>::integrate(x, u_minus, p, dt, type, cfg);

            for (int i = 0; i < Model::NX; ++i) {
                const double fd = (f_plus(i) - f_minus(i)) / (2.0 * eps);
                EXPECT_NEAR(kp.B(i, j), fd, 2e-5)
                    << "B mismatch for integrator " << static_cast<int>(type) << " at (" << i << ","
                    << j << ")";
            }
        }
    }
}

TEST(ImplicitIntegratorTest, TerminalImplicitEvaluationAtZeroDtIsFinite)
{
    using Model = ControlledImplicitJacobianModel;
    using Knot = KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>;

    SolverConfig config;
    config.integrator = IntegratorType::RK4_IMPLICIT;
    config.newton_config.tol = 1e-13;

    Knot kp;
    kp.set_zero();
    kp.x(0) = 0.4;
    kp.x(1) = -0.2;
    kp.u(0) = 0.3;

    detail::evaluate_model_stage<Model>(kp, config, 0.0, true);

    EXPECT_TRUE(kp.f_resid.allFinite());
    EXPECT_TRUE(kp.A.allFinite());
    EXPECT_TRUE(kp.B.allFinite());
    EXPECT_NEAR(kp.f_resid(0), 0.0, 1e-12);
    EXPECT_NEAR(kp.f_resid(1), 0.0, 1e-12);
    EXPECT_NEAR(kp.A(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(kp.A(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(kp.A(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(kp.A(1, 1), 0.0, 1e-12);
    EXPECT_NEAR(kp.B(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(kp.B(1, 0), 0.0, 1e-12);
}

TEST(ImplicitIntegratorTest, TerminalEvaluateModelStageSkipsDynamics)
{
    using Model = TerminalDynamicsCounterModel;
    using Knot = KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>;

    SolverConfig config;
    config.integrator = IntegratorType::RK4_EXPLICIT;

    Knot kp;
    kp.set_zero();
    kp.x(0) = 2.0;
    kp.u(0) = 3.0;

    Model::dynamics_calls = 0;
    detail::evaluate_model_stage<Model>(kp, config, 0.0, true);

    EXPECT_EQ(Model::dynamics_calls, 0)
        << "terminal evaluation must not run dynamics/integrator work";
    EXPECT_DOUBLE_EQ(MatOps::norm_inf(kp.f_resid), 0.0);
    EXPECT_DOUBLE_EQ(MatOps::norm_inf(kp.A), 0.0);
    EXPECT_DOUBLE_EQ(MatOps::norm_inf(kp.B), 0.0);
}

// --- Gauss-Legendre (RK4 Implicit) accuracy: O(dt^4) for linear ---
TEST(ImplicitIntegratorTest, GaussLegendreAccuracy)
{
    double lambda = 2.0;
    double dt = 0.1; // larger dt to show accuracy advantage
    double x0_val = 1.0;
    double t_end = 1.0;
    int steps = static_cast<int>(t_end / dt);

    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 1> p;
    p(0) = lambda;

    // Backward Euler
    MSVec<double, 1> x_be;
    x_be(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_be = ImplicitIntegrator<StiffDecayModel>::integrate(
            x_be, u, p, dt, IntegratorType::EULER_IMPLICIT);
    }
    double err_be = std::abs(x_be(0) - x0_val * std::exp(-lambda * t_end));

    // Implicit Midpoint
    MSVec<double, 1> x_im;
    x_im(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_im = ImplicitIntegrator<StiffDecayModel>::integrate(
            x_im, u, p, dt, IntegratorType::RK2_IMPLICIT);
    }
    double err_im = std::abs(x_im(0) - x0_val * std::exp(-lambda * t_end));

    // Gauss-Legendre (RK4 Implicit)
    MSVec<double, 1> x_gl;
    x_gl(0) = x0_val;
    for (int k = 0; k < steps; ++k) {
        x_gl = ImplicitIntegrator<StiffDecayModel>::integrate(
            x_gl, u, p, dt, IntegratorType::RK4_IMPLICIT);
    }
    double err_gl = std::abs(x_gl(0) - x0_val * std::exp(-lambda * t_end));

    std::cerr << "[ImplicitIntegrator] Accuracy comparison (dt=" << dt << "):\n"
              << "  Backward Euler:  err=" << err_be << "\n"
              << "  Implicit Midpoint: err=" << err_im << "\n"
              << "  Gauss-Legendre:  err=" << err_gl << "\n"
              << "  ratio BE/GL:     " << err_be / std::max(err_gl, 1e-30) << "\n"
              << "  ratio IM/GL:     " << err_im / std::max(err_gl, 1e-30) << "\n";

    // Gauss-Legendre should be more accurate than midpoint, which is more
    // accurate than backward Euler.
    EXPECT_LT(err_gl, err_im) << "Gauss-Legendre should beat midpoint";
    EXPECT_LT(err_im, err_be) << "Midpoint should beat backward Euler";
    // For this linear problem, Gauss-Legendre is order 4 — error ~ dt^4.
    // At dt=0.1, expect err ~ 1e-4 or less.
    EXPECT_LT(err_gl, 1e-3) << "Gauss-Legendre error too large for dt=0.1";
}

// --- Benchmark: warm start vs cold start ---
TEST(ImplicitIntegratorTest, WarmStartVsColdStart)
{
    // Run many Newton solves on the same problem, measure iterations.
    // Warm start should converge in fewer iterations because the previous
    // solution is a good initial guess.
    double lambda = 2.0;

    MSVec<double, 1> u;
    u.setZero();
    MSVec<double, 1> p;
    p(0) = lambda;
    NewtonConfig cfg;
    cfg.max_iters = 50;
    cfg.tol = 1e-12;

    // Use NonlinearDecayModel (dx/dt = -x^2) with small dt so x barely
    // changes between steps. Warm start's z_prev should be very close to
    // the next solution, reducing Newton iterations.
    double small_dt = 0.001; // x changes by ~0.001 per step

    // Cold start: new solver each time, always start from x
    int cold_total_iters = 0;
    {
        MSVec<double, 1> x;
        x(0) = 1.0;
        for (int step = 0; step < 1000; ++step) {
            NewtonSolver<double, 1> ns;
            MSVec<double, 1> z = x;
            int iters = 0;
            auto eval
                = [&](const MSVec<double, 1>& z_in, MSVec<double, 1>& F, MSMat<double, 1, 1>& J) {
                      iters++;
                      double zz = z_in(0);
                      F(0) = zz - x(0) + small_dt * zz * zz;
                      J(0, 0) = 1.0 + 2.0 * small_dt * zz;
                  };
            ns.solve(z, eval, cfg, /*warm_start=*/false);
            cold_total_iters += iters;
            x = z;
        }
    }

    // Warm start: reuse solver, seed from previous converged solution
    int warm_total_iters = 0;
    {
        MSVec<double, 1> x;
        x(0) = 1.0;
        NewtonSolver<double, 1> ns;
        for (int step = 0; step < 1000; ++step) {
            MSVec<double, 1> z = x;
            int iters = 0;
            auto eval
                = [&](const MSVec<double, 1>& z_in, MSVec<double, 1>& F, MSMat<double, 1, 1>& J) {
                      iters++;
                      double zz = z_in(0);
                      F(0) = zz - x(0) + small_dt * zz * zz;
                      J(0, 0) = 1.0 + 2.0 * small_dt * zz;
                  };
            ns.solve(z, eval, cfg, /*warm_start=*/true);
            warm_total_iters += iters;
            x = z;
        }
    }

    std::cerr << "[Benchmark] Newton warm start vs cold start (1000 steps, dt=0.001):\n"
              << "  cold start total iters: " << cold_total_iters << "\n"
              << "  warm start total iters: " << warm_total_iters << "\n";

    // Now measure wall-clock time for 100k solves (Gauss-Legendre sized)
    // to quantify the workspace reuse benefit.
    const int N_ITER = 100000;
    MSVec<double, 1> x0;
    x0(0) = 0.5;

    // Cold: create workspace each call
    auto t_cold_start = std::chrono::steady_clock::now();
    for (int i = 0; i < N_ITER; ++i) {
        NewtonSolver<double, 1> ns;
        MSVec<double, 1> z = x0;
        ns.solve(
            z,
            [&](const MSVec<double, 1>& z_in, MSVec<double, 1>& F, MSMat<double, 1, 1>& J) {
                F(0) = z_in(0) - 0.5 + 0.01 * z_in(0) * z_in(0);
                J(0, 0) = 1.0 + 0.02 * z_in(0);
            },
            cfg, false);
    }
    auto t_cold_end = std::chrono::steady_clock::now();
    double cold_us = std::chrono::duration<double, std::micro>(t_cold_end - t_cold_start).count();

    // Warm: reuse workspace
    NewtonSolver<double, 1> ns_warm;
    auto t_warm_start = std::chrono::steady_clock::now();
    for (int i = 0; i < N_ITER; ++i) {
        MSVec<double, 1> z = x0;
        ns_warm.solve(
            z,
            [&](const MSVec<double, 1>& z_in, MSVec<double, 1>& F, MSMat<double, 1, 1>& J) {
                F(0) = z_in(0) - 0.5 + 0.01 * z_in(0) * z_in(0);
                J(0, 0) = 1.0 + 0.02 * z_in(0);
            },
            cfg, true);
    }
    auto t_warm_end = std::chrono::steady_clock::now();
    double warm_us = std::chrono::duration<double, std::micro>(t_warm_end - t_warm_start).count();

    std::cerr << "[Benchmark] 100k Newton solves (NX=1):\n"
              << "  cold (new workspace): " << cold_us / 1000.0 << " ms\n"
              << "  warm (reuse workspace): " << warm_us / 1000.0 << " ms\n"
              << "  speedup: " << cold_us / warm_us << "x\n";
}
