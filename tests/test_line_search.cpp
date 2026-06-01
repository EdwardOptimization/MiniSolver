/**
 * @file test_line_search.cpp
 * @brief Tests for both Filter and Merit line search strategies.
 */
#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/algorithms/line_search.h"
#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/line_search_utils.h"
#include "minisolver/solver/solver.h"
#include "solver_internal_access.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace minisolver;

// =============================================================================
// Mock Linear Solver (shared by Filter and Merit tests)
// =============================================================================
class MockLinearSolver
    : public LinearSolver<Trajectory<KnotPoint<double, 4, 2, 5, 13>, 10>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 4, 2, 5, 13>, 10>::TrajArray;

    LinearSolveResult solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        for (int k = 0; k <= N; ++k) {
            traj[k].dx = -0.1 * traj[k].x;
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
        }
        return true;
    }
};

struct MixedBoundaryModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<bool, NC> constraint_has_l1 = { true };
    static constexpr std::array<bool, NC> constraint_has_l2 = { true };
    static constexpr bool any_l1_constraints = true;
    static constexpr bool any_l2_constraints = true;
};

struct BasicFilterAcceptanceModel {
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
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r.setZero();
        kp.Q(0, 0) = 2.0;
        kp.R.setZero();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }
};

class BasicFilterLinearSolver
    : public LinearSolver<Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2>::TrajArray;

    LinearSolveResult solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        for (int k = 0; k <= N; ++k) {
            traj[k].dx = -0.1 * traj[k].x;
            traj[k].du.setZero();
        }
        return true;
    }
};

// =============================================================================
// Filter Line Search
// =============================================================================
TEST(LineSearchTest, FilterAcceptance)
{
    SolverConfig config;
    config.line_search_type = LineSearchType::FILTER;

    constexpr int N = 0;
    constexpr int MAX_N = 2;
    using Model = BasicFilterAcceptanceModel;
    using Strategy = FilterLineSearch<Model, MAX_N>;

    Strategy ls;
    BasicFilterLinearSolver linear_solver;

    Trajectory<KnotPoint<double, 1, 1, 0, 0>, MAX_N> trajectory(N);
    std::array<double, MAX_N> dts;
    dts.fill(0.1);

    for (int k = 0; k <= N; ++k) {
        trajectory.active()[k].set_zero();
        trajectory.active()[k].x.fill(10.0);
        Model::compute_dynamics(trajectory.active()[k], config.integrator, 0.0);
        Model::compute_cost_gn(trajectory.active()[k]);
    }

    linear_solver.solve(trajectory.active(), N, 0.1, 1e-6, InertiaStrategy::REGULARIZATION, config);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    const double alpha = result.alpha;

    EXPECT_GT(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    EXPECT_LT(trajectory.active()[0].x(0), 10.0);
    EXPECT_NEAR(trajectory.active()[0].x(0), 10.0 - alpha, 1e-12);
}

TEST(LineSearchTest, FractionToBoundaryProtectsMixedSoftInterior)
{
    using Knot = KnotPoint<double, MixedBoundaryModel::NX, MixedBoundaryModel::NU,
        MixedBoundaryModel::NC, MixedBoundaryModel::NP>;
    std::array<Knot, 1> traj;
    SolverConfig config;
    config.min_barrier_slack = 1e-12;
    constexpr double tau = 0.995;

    traj[0].set_zero();
    traj[0].s(0) = 1.0;
    traj[0].lam(0) = 0.5;
    traj[0].soft_s(0) = 0.4;
    traj[0].l1_weight(0) = 2.0;
    traj[0].l2_weight(0) = 3.0;
    traj[0].dsoft_s(0) = -10.0;

    double alpha
        = fraction_to_boundary_rule<decltype(traj), MixedBoundaryModel>(traj, 0, tau, config);
    EXPECT_NEAR(alpha, tau * 0.4 / 10.0, 1e-15);

    traj[0].dsoft_s(0) = 0.0;
    traj[0].dlam(0) = 10.0;
    const double soft_dual = detail::l1_soft_dual_gap<MixedBoundaryModel>(traj[0], 0)
        - detail::l1_soft_dual_floor(traj[0].l1_weight(0), config);

    alpha = fraction_to_boundary_rule<decltype(traj), MixedBoundaryModel>(traj, 0, tau, config);
    EXPECT_NEAR(alpha, tau * soft_dual / 10.0, 1e-15);
}

// =============================================================================
// Merit Model (highly nonlinear constraint for backtracking test)
// =============================================================================
struct MeritModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        return x + u * dt;
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
        kp.cost = 0.0;
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.q.setZero();
        kp.r.setZero();
        kp.g_val(0) = pow(kp.x(0), 4) - 1.0;
        kp.C(0, 0) = 4 * pow(kp.x(0), 3);
        kp.D.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute(kp, type, dt);
    }
    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
};

// =============================================================================
// Rollout Line Search Model (linear dynamics, no constraints)
// Used to verify enable_line_search_rollout produces a rollout-consistent state
// trajectory (x propagated from x0 using the candidate controls).
// =============================================================================
struct RolloutLineSearchModel {
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
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
        kp.cost = kp.x(0) * kp.x(0) + kp.u(0) * kp.u(0);
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 2.0 * kp.u(0);
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute(kp, type, dt);
    }
    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }
};

struct MeritAnalyticDphiModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static inline int cost_evaluations = 0;

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
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        ++cost_evaluations;
        kp.cost = kp.x(0) * kp.x(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r.setZero();
        kp.Q(0, 0) = 2.0;
        kp.R.setZero();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }
};

struct MeritOverflowDphiModel {
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
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = T(1);
        kp.B(0, 0) = T(dt);
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = T(0);
        kp.q(0) = std::numeric_limits<T>::max();
        kp.r.setZero();
        kp.Q.setZero();
        kp.R.setZero();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }
};

struct MeritOverflowPhiModel {
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
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = T(1);
        kp.B(0, 0) = T(dt);
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = (kp.x(0) > T(0.5)) ? std::numeric_limits<T>::infinity() : kp.x(0) * kp.x(0);
        kp.q(0) = T(0);
        kp.r.setZero();
        kp.Q.setZero();
        kp.R.setZero();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }
};

struct FilterThetaMaxModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x, const MSVec<T, NU>&, const MSVec<T, NP>&, double, IntegratorType)
    {
        return x;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double)
    {
        kp.f_resid = kp.x;
        kp.A.setIdentity();
        kp.B.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = (kp.x(0) < T(0)) ? T(100) : T(0);
        kp.q.setZero();
        kp.r.setZero();
        kp.Q.setZero();
        kp.R.setZero();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0);
        kp.C(0, 0) = 1.0;
        kp.D.setZero();
    }
};

struct FilterSwitchingModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x, const MSVec<T, NU>&, const MSVec<T, NP>&, double, IntegratorType)
    {
        return x;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double)
    {
        kp.f_resid = kp.x;
        kp.A.setIdentity();
        kp.B.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = static_cast<T>(100.0) - static_cast<T>(100.0) * kp.x(0)
            + static_cast<T>(99.999) * kp.x(0) * kp.x(0);
        kp.q(0) = static_cast<T>(-100.0) + static_cast<T>(199.998) * kp.x(0);
        kp.r.setZero();
        kp.Q(0, 0) = static_cast<T>(199.998);
        kp.R.setZero();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = 0.0;
        kp.C.setZero();
        kp.D.setZero();
    }
};

struct L2ResidualFilterModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<bool, NC> constraint_has_l1 = { false };
    static constexpr std::array<bool, NC> constraint_has_l2 = { true };
    static constexpr bool any_l1_constraints = false;
    static constexpr bool any_l2_constraints = true;

    template <typename T>
    static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.l1_weight(0) = T(0);
        kp.l2_weight(0) = T(1);
    }

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> out;
        out(0) = x(0) + u(0) * dt;
        return out;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0);
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = 0.0;
        kp.q.setZero();
        kp.r.setZero();
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.H.setZero();
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct SocHardResidualModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& /*u*/,
        const MSVec<T, NP>& /*p*/, double /*dt*/, IntegratorType /*type*/)
    {
        return x;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double /*dt*/)
    {
        kp.f_resid(0) = kp.x(0);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = 0.0;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = 0.0;
        kp.C(0, 0) = 0.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = 0.0;
        kp.q.setZero();
        kp.r.setZero();
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.H.setZero();
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct SocOverrideResidualModel : public SocHardResidualModel {
    template <typename T>
    static void compute_soc_constraints(
        const KnotPoint<T, NX, NU, NC, NP>& /*active_kp*/, KnotPoint<T, NX, NU, NC, NP>& trial_kp)
    {
        trial_kp.g_val(0) = 7.0;
    }
};

struct FilterSocNonFiniteMetricsModel : public SocHardResidualModel {
    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = (kp.s(0) < T(1.0)) ? std::numeric_limits<T>::infinity() : T(0);
        kp.q.setZero();
        kp.r.setZero();
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct TrueResidualLineSearchModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& /*u*/,
        const MSVec<T, NP>& /*p*/, double /*dt*/, IntegratorType /*type*/)
    {
        return x;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double /*dt*/)
    {
        kp.f_resid(0) = kp.x(0);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = 0.0;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = 100.0; // QP residual intentionally disagrees with true residual.
        kp.C(0, 0) = 0.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_true_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_true(0) = kp.x(0);
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = 0.0;
        kp.q.setZero();
        kp.r.setZero();
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.H.setZero();
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

class RolloutStubLinearSolver
    : public LinearSolver<Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2>::TrajArray;
    LinearSolveResult solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        return true;
    }
};

template <typename TrajArray>
class OverflowDphiRiccatiSolver : public RiccatiSolver<TrajArray, MeritOverflowDphiModel> {
public:
    LinearSolveResult solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        for (int k = 0; k <= N; ++k) {
            traj[k].dx(0) = 2.0;
            if (k < N) {
                traj[k].du.setZero();
            }
            traj[k].r_bar.setOnes();
        }
        return true;
    }
};

class L2ResidualStubLinearSolver
    : public LinearSolver<Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1>::TrajArray;
    LinearSolveResult solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        return true;
    }
};

class SocCaptureLinearSolver
    : public LinearSolver<Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1>::TrajArray> {
public:
    using TrajArray = Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1>::TrajArray;

    bool called = false;
    double correction_ds = 0.0;
    double observed_soc_base_s = 0.0;
    double observed_trial_s = 0.0;
    double observed_soc_rhs_g = 0.0;

    LinearSolveResult solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        return true;
    }

    LinearSolveResult solve_soc(TrajArray& traj, const TrajArray& soc_rhs_traj, int N,
        double /*mu*/, double /*reg*/, InertiaStrategy /*strategy*/,
        const SolverConfig& /*config*/) override
    {
        called = true;
        observed_soc_base_s = traj[0].s(0);
        observed_trial_s = soc_rhs_traj[0].s(0);
        observed_soc_rhs_g = soc_rhs_traj[0].g_val(0);
        for (int k = 0; k <= N; ++k) {
            traj[k].dx.setZero();
            traj[k].du.setZero();
            traj[k].ds(0) = correction_ds;
            traj[k].dlam.setZero();
            traj[k].dsoft_s.setZero();
        }
        return true;
    }
};

// =============================================================================
// Merit Line Search with Backtracking
// =============================================================================
TEST(LineSearchTest, MeritFunctionBacktracking)
{
    int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.line_search_type = LineSearchType::MERIT;
    config.max_iters = 20;
    config.enable_feasibility_restoration = true;
    config.enable_residual_stagnation_detection = false;
    config.merit_nu_init = 1000.0;

    MiniSolver<MeritModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 2.0);

    SolverStatus status = solver.solve();

    if (status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE) {
        EXPECT_NEAR(solver.get_state(0, 0), 1.0, 0.2);
    } else {
        EXPECT_TRUE(status == SolverStatus::MAX_ITER || status == SolverStatus::RESTORATION_FAILED
            || status == SolverStatus::STEP_TOO_SMALL || status == SolverStatus::NUMERICAL_ERROR)
            << "Unexpected merit line-search solve status: " << status_to_string(status);
    }
}

TEST(LineSearchTest, FilterRejectsPureL2KktResidualIncrease)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 3;
    config.enable_soc = false;

    using Model = L2ResidualFilterModel;
    FilterLineSearch<Model, 1> ls;
    L2ResidualStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 0.0;
    active[0].u(0) = 0.0;
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0; // L2 residual g+s-lam/w is initially zero.
    active[0].ds(0) = 0.1; // Increases g+s and therefore L2 KKT residual.
    active[0].dlam(0) = 0.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::update_soft_constraint_weights(active[0]);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    const double alpha = result.alpha;

    EXPECT_DOUBLE_EQ(alpha, 0.0)
        << "Filter theta must include the L2 residual g+s-lam/w, otherwise a pure "
           "L2 KKT residual increase is accepted.";
}

TEST(LineSearchTest, FilterSocUsesCandidateSlackAsCorrectionBase)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = true;
    config.soc_trigger_alpha = 0.1;
    config.line_search_tau = 0.9;

    using Model = SocHardResidualModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    linear_solver.correction_ds = 0.0;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].ds(0) = 0.2; // trial slack becomes 1.2, making the filter reject.
    active[0].dlam(0) = 0.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1e-6, config);

    EXPECT_TRUE(result.soc_attempted);
    ASSERT_TRUE(linear_solver.called);
    EXPECT_DOUBLE_EQ(linear_solver.observed_trial_s, 1.2);
    EXPECT_DOUBLE_EQ(linear_solver.observed_soc_base_s, linear_solver.observed_trial_s)
        << "SOC correction is applied to candidate, so the correction baseline must use "
           "candidate slack/dual variables, not active ones.";
}

TEST(LineSearchTest, FilterSocDampsCorrectionToStayInterior)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = true;
    config.soc_trigger_alpha = 0.1;
    config.line_search_tau = 0.5;

    using Model = SocHardResidualModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    linear_solver.correction_ds = -2.4; // Full SOC would move trial slack 1.2 to -1.2.
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].ds(0) = 0.2; // trial slack = 1.2, filter rejects before SOC.
    active[0].dlam(0) = 0.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1e-6, config);
    const double alpha = result.alpha;

    EXPECT_TRUE(result.soc_attempted);
    EXPECT_TRUE(result.soc_accepted);
    ASSERT_TRUE(linear_solver.called);
    EXPECT_GT(alpha, 0.0)
        << "SOC should be damped to an interior candidate instead of applying an invalid full "
           "correction.";
    EXPECT_GT(trajectory.active()[0].s(0), 0.0);
    EXPECT_NEAR(trajectory.active()[0].s(0), 0.6, 1e-12);
}

TEST(LineSearchTest, FilterSocUsesModelSocConstraintOverride)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = true;
    config.soc_trigger_alpha = 0.1;
    config.line_search_tau = 0.9;

    using Model = SocOverrideResidualModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].ds(0) = 0.2;
    active[0].dlam(0) = 0.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1e-6, config);

    EXPECT_TRUE(result.soc_attempted);
    ASSERT_TRUE(linear_solver.called);
    EXPECT_DOUBLE_EQ(linear_solver.observed_soc_rhs_g, 7.0)
        << "SOC should let the model override correction residuals without changing normal true "
           "constraint evaluation.";
}

TEST(LineSearchTest, FilterAcceptanceUsesTrueResidualNotQpResidual)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = false;
    config.line_search_tau = 0.9;

    using Model = TrueResidualLineSearchModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 1.0;
    active[0].dx(0) = -1.0;
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_true_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1e-6, config);
    const double alpha = result.alpha;

    EXPECT_GT(alpha, 0.0)
        << "Filter globalization should evaluate true nonlinear residuals, not the QP/IPM "
           "linearization residual packet.";
    EXPECT_NEAR(trajectory.active()[0].g_true(0), trajectory.active()[0].x(0), 1e-12);
}

TEST(LineSearchTest, FilterRejectsTrialAboveThetaMax)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.line_search_tau = 0.9;
    config.enable_soc = false;

    using Model = FilterThetaMaxModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = -1.0;
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].dx(0) = 20000.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);
    const double alpha = result.alpha;

    EXPECT_EQ(alpha, 0.0)
        << "Filter must reject trials whose violation exceeds theta_max even if phi decreases.";
    EXPECT_NEAR(trajectory.active()[0].x(0), -1.0, 1e-12)
        << "Rejected filter candidates must not swap into the active trajectory.";
}

TEST(LineSearchTest, FilterFTypeUsesArmijoAndDoesNotAugmentFilter)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 3;
    config.line_search_tau = 0.999;
    config.line_search_backtrack_factor = 0.5;
    config.enable_soc = false;
    config.eta_suff_descent = 1e-4;

    using Model = FilterSwitchingModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 0.0;
    active[0].dx(0) = 1.0;
    active[0].s(0) = 1e-8;
    active[0].lam(0) = 1.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_GT(result.alpha, 0.0);
    EXPECT_LT(result.alpha, 1.0)
        << "Near-feasible f-type steps should use Armijo sufficient decrease, not the weaker "
           "filter OR condition.";
    EXPECT_EQ(ls.filter_size(), 0u) << "Accepted f-type steps must not augment the filter.";
}

TEST(LineSearchTest, FilterHTypeAcceptanceStillAugmentsFilter)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.line_search_tau = 0.999;
    config.enable_soc = false;

    using Model = FilterSwitchingModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 0.0;
    active[0].dx(0) = 1.0;
    active[0].s(0) = 1.0; // not near-feasible, so this is h-type.
    active[0].lam(0) = 1.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_GT(result.alpha, 0.0);
    EXPECT_EQ(ls.filter_size(), 1u) << "Accepted h-type steps should still augment the filter.";
}

TEST(LineSearchTest, FilterNonFiniteInitialMetricsReturnsNumericalError)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = false;

    using Model = MeritOverflowPhiModel;
    FilterLineSearch<Model, 2> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2> trajectory(N);
    std::array<double, 2> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 1.0;
    active[0].dx(0) = -1.0;
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_EQ(result.status, SolverStatus::NUMERICAL_ERROR)
        << "A non-finite initial filter theta/phi pair is not a valid acceptance baseline.";
}

TEST(LineSearchTest, FilterNonFiniteDphiReturnsNumericalError)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = false;

    using Model = MeritOverflowDphiModel;
    FilterLineSearch<Model, 2> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2> trajectory(N);
    std::array<double, 2> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].dx(0) = 2.0;
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_EQ(result.status, SolverStatus::NUMERICAL_ERROR)
        << "A non-finite filter directional derivative must surface as numerical failure.";
}

TEST(LineSearchTest, FilterNonFiniteTrialMetricsReturnsNumericalError)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = false;

    using Model = MeritOverflowPhiModel;
    FilterLineSearch<Model, 2> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2> trajectory(N);
    std::array<double, 2> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 0.0;
    active[0].dx(0) = 1.0;
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_EQ(result.status, SolverStatus::NUMERICAL_ERROR)
        << "A non-finite trial filter theta/phi pair must surface as numerical failure, not "
           "ordinary backtracking or h-type acceptance.";
}

TEST(LineSearchTest, FilterSocNonFiniteMetricsReturnsNumericalError)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = true;
    config.soc_trigger_alpha = 0.1;
    config.line_search_tau = 0.5;

    using Model = FilterSocNonFiniteMetricsModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    linear_solver.correction_ds = -2.4; // SOC damps candidate slack from 1.2 to 0.6.
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].ds(0) = 0.2; // trial slack = 1.2, filter rejects before SOC.
    active[0].dlam(0) = 0.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_TRUE(result.soc_attempted);
    EXPECT_EQ(result.status, SolverStatus::NUMERICAL_ERROR)
        << "SOC candidate metrics are filter acceptance scalars too; non-finite SOC metrics "
           "must not be accepted as h-type progress.";
}

TEST(LineSearchTest, MeritAcceptanceUsesTrueResidualNotQpResidual)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::MERIT;
    config.line_search_max_iters = 1;
    config.armijo_c1 = 0.0;
    config.line_search_tau = 0.9;

    using Model = TrueResidualLineSearchModel;
    MeritLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 1.0;
    active[0].dx(0) = -1.0;
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_true_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1e-6, config);
    const double alpha = result.alpha;

    EXPECT_GT(alpha, 0.0)
        << "Merit globalization should evaluate true nonlinear residuals, not the QP/IPM "
           "linearization residual packet.";
    EXPECT_NEAR(trajectory.active()[0].g_true(0), trajectory.active()[0].x(0), 1e-12);
}

TEST(LineSearchTest, MeritArmijoDoesNotBuildFiniteDifferenceProbe)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::MERIT;
    config.line_search_max_iters = 1;
    config.armijo_c1 = 1.0e-4;
    config.enable_line_search_rollout = false;

    using Model = MeritAnalyticDphiModel;
    MeritLineSearch<Model, 2> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2> trajectory(N);
    std::array<double, 2> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 1.0;
    active[0].dx(0) = -0.5;
    active[0].s.setZero();
    active[0].lam.setZero();
    Model::compute_cost_gn(active[0]);
    Model::cost_evaluations = 0;

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);
    const double alpha = result.alpha;

    EXPECT_GT(alpha, 0.0);
    EXPECT_NEAR(trajectory.active()[0].x(0), 1.0 - 0.5 * alpha, 1e-12);
    EXPECT_EQ(Model::cost_evaluations, 1)
        << "Merit Armijo should use analytic dphi instead of building an extra finite-difference "
           "trial point before the first backtracking evaluation.";
}

TEST(LineSearchTest, MeritNonFiniteDphiReturnsNumericalError)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::MERIT;
    config.line_search_max_iters = 1;
    config.armijo_c1 = 1.0e-4;
    config.enable_line_search_rollout = false;

    using Model = MeritOverflowDphiModel;
    MeritLineSearch<Model, 2> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2> trajectory(N);
    std::array<double, 2> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].dx(0) = 2.0;
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_EQ(result.status, SolverStatus::NUMERICAL_ERROR)
        << "Finite but huge model gradients can overflow the merit directional derivative; "
           "line search should surface that numerical failure instead of coercing dphi to 0.";
}

TEST(LineSearchTest, MeritNonFiniteInitialPhiReturnsNumericalError)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::MERIT;
    config.line_search_max_iters = 1;
    config.armijo_c1 = 0.0;

    using Model = MeritOverflowPhiModel;
    MeritLineSearch<Model, 2> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2> trajectory(N);
    std::array<double, 2> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 1.0;
    active[0].dx(0) = -1.0;
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_EQ(result.status, SolverStatus::NUMERICAL_ERROR)
        << "A non-finite initial merit value is not a valid Armijo baseline.";
}

TEST(LineSearchTest, MeritNonFiniteTrialPhiReturnsNumericalError)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::MERIT;
    config.line_search_max_iters = 1;
    config.armijo_c1 = 0.0;

    using Model = MeritOverflowPhiModel;
    MeritLineSearch<Model, 2> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 2> trajectory(N);
    std::array<double, 2> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 0.0;
    active[0].dx(0) = 1.0;
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1.0e-6, config);

    EXPECT_EQ(result.status, SolverStatus::NUMERICAL_ERROR)
        << "A non-finite trial merit value must surface as numerical failure, not backtracking "
           "noise.";
}

TEST(LineSearchTest, MeritNonFiniteDphiPropagatesToSolverStatus)
{
    constexpr int N = 0;
    using Model = MeritOverflowDphiModel;
    using Solver = MiniSolver<Model, 2>;
    using Access = minisolver::test::SolverInternalAccess<Model, 2>;
    using FakeSolver = OverflowDphiRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 1;
    config.line_search_type = LineSearchType::MERIT;
    config.armijo_c1 = 1.0e-4;
    config.enable_line_search_rollout = false;

    Solver solver(N, Backend::CPU_SERIAL, config);
    Access::set_linear_solver(solver, std::make_unique<FakeSolver>());

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::NUMERICAL_ERROR);
    EXPECT_EQ(solver.get_info().loop_status, SolverStatus::NUMERICAL_ERROR);
    EXPECT_EQ(solver.get_info().termination_reason, TerminationReason::NUMERICAL_ERROR);
}

TEST(LineSearchTest, FilterSocSkippedInRolloutMode)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = true;
    config.enable_line_search_rollout = true;
    config.soc_trigger_alpha = 0.1;
    config.line_search_tau = 0.9;

    using Model = SocHardResidualModel;
    FilterLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].ds(0) = 0.2;
    active[0].dlam(0) = 0.0;
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.0, 1e-6, config);

    EXPECT_FALSE(result.soc_attempted);
    EXPECT_FALSE(linear_solver.called)
        << "Current SOC implementation is multiple-shooting only; rollout mode needs a separate "
           "control-space SOC definition.";
}

TEST(LineSearchTest, MeritRolloutProducesConsistentStates)
{
    constexpr int N = 2;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::MERIT;
    config.enable_line_search_rollout = true;
    config.line_search_max_iters = 5;
    config.armijo_c1 = 0.0; // This test targets rollout state construction, not Armijo acceptance.

    using Model = RolloutLineSearchModel;
    MeritLineSearch<Model, N> ls;
    RolloutStubLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, N> trajectory(N);

    std::array<double, N> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    for (int k = 0; k <= N; ++k) {
        active[k].set_zero();
        active[k].x(0) = (k == 0) ? 0.0 : 1000.0;
        active[k].u(0) = 0.0;
        active[k].cost = 1e9; // make phi_0 huge so acceptance is easy
        active[k].f_resid(0) = 0.0; // make theta_0 huge (defect term)
        active[k].dx(0) = 0.0;
        active[k].du(0) = 0.0;
    }
    // Make the linear-step update differ from the rollout propagation at stage 1.
    // Buggy rollout overwrites propagated x1 with x1 + dx1 (=100).
    active[1].dx(0) = -900.0;

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    const double alpha = result.alpha;
    EXPECT_GT(alpha, 0.0);

    const auto& after = trajectory.active();
    double x1_expected = after[0].x(0) + after[0].u(0) * dts[0];
    EXPECT_NEAR(after[1].x(0), x1_expected, 1e-12);
}

// =============================================================================
// No Line Search: must refresh accepted-point evaluations (cost/f_resid/...)
// =============================================================================
struct NoLineSearchEvalModel {
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
        return x + u * dt;
    }

    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0) + kp.u(0) * kp.u(0);
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 2.0 * kp.u(0);
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost(kp);
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost(kp);
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }
};

template <int MAX_N>
class NoLineSearchStubLinearSolver
    : public LinearSolver<typename Trajectory<KnotPoint<double, 1, 1, 0, 0>, MAX_N>::TrajArray> {
public:
    using TrajArray = typename Trajectory<KnotPoint<double, 1, 1, 0, 0>, MAX_N>::TrajArray;
    LinearSolveResult solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        return true;
    }
};

TEST(LineSearchTest, FilterHistoryWrapsAtFixedCapacity)
{
    constexpr int N = 1;
    constexpr int Repetitions = 1100;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = false;

    using Model = NoLineSearchEvalModel;
    FilterLineSearch<Model, N> ls;
    NoLineSearchStubLinearSolver<N> linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, N> trajectory(N);

    std::array<double, N> dts;
    dts.fill(0.1);

    for (int iter = 0; iter < Repetitions; ++iter) {
        auto& active = trajectory.active();
        for (int k = 0; k <= N; ++k) {
            active[k].set_zero();
        }

        // Dynamic defect theta starts large and decreases monotonically.
        // The trial point reduces the defect enough to be acceptable against
        // every previous filter entry, while still pushing beyond FILTER_CAPACITY.
        active[0].x(0) = 0.0;
        active[0].u(0) = 0.0;
        active[1].x(0) = 2000.0 - static_cast<double>(iter);
        active[1].dx(0) = -0.5;

        for (int k = 0; k <= N; ++k) {
            const double current_dt = (k < N) ? dts[static_cast<size_t>(k)] : 0.0;
            detail::evaluate_model_stage<Model>(active[k], config, current_dt, k == N);
        }

        const LineSearchResult result
            = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
        const double alpha = result.alpha;
        ASSERT_GT(alpha, 0.0) << "filter rejected at iteration " << iter;
    }

    EXPECT_EQ(ls.filter_size(), 1024u);
}

TEST(LineSearchTest, FilterBarrierUpdateClearsHistory)
{
    constexpr int N = 1;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = false;

    using Model = NoLineSearchEvalModel;
    FilterLineSearch<Model, N> ls;
    NoLineSearchStubLinearSolver<N> linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, N> trajectory(N);

    std::array<double, N> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    for (int k = 0; k <= N; ++k) {
        active[k].set_zero();
    }

    active[0].x(0) = 0.0;
    active[0].u(0) = 0.0;
    active[1].x(0) = 10.0;
    active[1].dx(0) = -1.0;

    for (int k = 0; k <= N; ++k) {
        const double current_dt = (k < N) ? dts[static_cast<size_t>(k)] : 0.0;
        detail::evaluate_model_stage<Model>(active[k], config, current_dt, k == N);
    }

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    ASSERT_GT(result.alpha, 0.0);
    ASSERT_EQ(ls.filter_size(), 1u);

    ls.on_barrier_update();

    EXPECT_EQ(ls.filter_size(), 0u)
        << "Filter entries recorded under the old barrier parameter are not comparable.";
}

TEST(LineSearchTest, NoLineSearchRefreshesAcceptedPointEvaluations)
{
    constexpr int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::NONE;

    using Model = NoLineSearchEvalModel;
    NoLineSearch<Model, N> ls;
    NoLineSearchStubLinearSolver<N> linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, N> trajectory(N);

    std::array<double, N> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    for (int k = 0; k <= N; ++k) {
        active[k].set_zero();
        active[k].x(0) = 10.0;
        active[k].u(0) = 1.0;

        // Stale evaluation fields that should be refreshed for the accepted point.
        active[k].cost = 12345.0;
        active[k].f_resid(0) = -999.0;

        // Full step.
        active[k].dx(0) = -1.0;
        active[k].du(0) = 0.0;
    }

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    const double alpha = result.alpha;
    EXPECT_GT(alpha, 0.0);

    const auto& after = trajectory.active();
    EXPECT_NEAR(after[0].x(0), 9.0, 1e-12);
    EXPECT_NEAR(after[0].u(0), 1.0, 1e-12);
    EXPECT_NEAR(after[0].cost, 82.0, 1e-12); // 9^2 + 1^2
    EXPECT_NEAR(after[0].f_resid(0), 9.1, 1e-12); // x + u*dt
}

TEST(LineSearchTest, NoLineSearchAppliesFractionToBoundary)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::NONE;
    config.line_search_tau = 0.5;

    using Model = SocHardResidualModel;
    NoLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].ds(0) = -2.0;
    active[0].dlam(0) = 0.0;

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);

    EXPECT_NEAR(result.alpha, 0.25, 1e-15);
    EXPECT_NEAR(trajectory.active()[0].s(0), 0.5, 1e-15);
}

TEST(LineSearchTest, NoLineSearchTinyAlphaReturnsZeroStepResult)
{
    constexpr int N = 0;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::NONE;
    config.line_search_tau = 0.995;

    using Model = SocHardResidualModel;
    NoLineSearch<Model, 1> ls;
    SocCaptureLinearSolver linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 1, 0>, 1> trajectory(N);
    std::array<double, 1> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].s(0) = 1.0;
    active[0].lam(0) = 1.0;
    active[0].ds(0) = -1.0e12;

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);

    EXPECT_DOUBLE_EQ(result.alpha, 0.0);
    EXPECT_EQ(result.status, SolverStatus::UNSOLVED);
}

TEST(LineSearchTest, NoLineSearchDoesNotUpdateTerminalControl)
{
    constexpr int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::NONE;

    using Model = NoLineSearchEvalModel;
    NoLineSearch<Model, N> ls;
    NoLineSearchStubLinearSolver<N> linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, N> trajectory(N);

    std::array<double, N> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    for (int k = 0; k <= N; ++k) {
        active[k].set_zero();
        active[k].u(0) = 0.0;
        active[k].du(0) = 7.0;
    }

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    const double alpha = result.alpha;
    EXPECT_GT(alpha, 0.0);
    EXPECT_DOUBLE_EQ(trajectory.active()[N].u(0), 0.0)
        << "u_N is not a decision variable and must not be advanced by line search.";
}

TEST(LineSearchTest, NoLineSearchRolloutProducesConsistentStates)
{
    constexpr int N = 2;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::NONE;
    config.enable_line_search_rollout = true;

    using Model = NoLineSearchEvalModel;
    NoLineSearch<Model, N> ls;
    NoLineSearchStubLinearSolver<N> linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, N> trajectory(N);

    std::array<double, N> dts;
    dts.fill(0.1);

    auto& active = trajectory.active();
    for (int k = 0; k <= N; ++k) {
        active[k].set_zero();
        active[k].x(0) = (k == 0) ? 0.0 : 1000.0;
        active[k].u(0) = 0.0;
        active[k].dx(0) = 0.0;
        active[k].du(0) = 0.0;
    }
    // Make the linear-step update differ from rollout at stage 1.
    active[1].dx(0) = -900.0;
    active[0].du(0) = 1.0;

    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    const double alpha = result.alpha;
    EXPECT_GT(alpha, 0.0);

    const auto& after = trajectory.active();
    double x1_expected = after[0].x(0) + after[0].u(0) * dts[0];
    EXPECT_NEAR(after[1].x(0), x1_expected, 1e-12);
}
