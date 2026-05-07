// FULL_KKT_ITERATIVE_REFINEMENT contract tests.
//
// These tests pin three behaviours of the new direction-refinement mode:
//   1. The default config remains DirectionRefinementMode::NONE so the
//      legacy code path is untouched.
//   2. Enabling FULL_KKT_ITERATIVE_REFINEMENT must not regress an
//      unconstrained nonlinear tracking problem, must produce no more
//      passes than `direction_refinement_max_passes`, and must record the
//      pass count in SolverInfo.
//   3. On a problem with active inequality constraints the mode must
//      auto-degrade to a single pass so the OD-005 dual-consistency hazard
//      cannot be amplified by repeated primal-only refinements.

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

namespace {

// Smooth nonlinear pendulum-like model with no inequality constraints.
struct PendulumModel {
    static constexpr int NX = 2;
    static constexpr int NU = 1;
    static constexpr int NC = 0;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "theta", "omega" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>&,
        double dt, IntegratorType)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + x(1) * dt;
        xn(1) = x(1) + (-std::sin(static_cast<double>(x(0))) + u(0)) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double dt)
    {
        const double s = std::sin(static_cast<double>(kp.x(0)));
        const double c = std::cos(static_cast<double>(kp.x(0)));
        kp.f_resid(0) = kp.x(0) + kp.x(1) * dt;
        kp.f_resid(1) = kp.x(1) + (-s + kp.u(0)) * dt;
        kp.A(0, 0) = 1.0;
        kp.A(0, 1) = dt;
        kp.A(1, 0) = -c * dt;
        kp.A(1, 1) = 1.0;
        kp.B(0, 0) = 0.0;
        kp.B(1, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = static_cast<T>(2.0) * kp.x(0) * kp.x(0) + static_cast<T>(0.1) * kp.x(1) * kp.x(1)
            + static_cast<T>(0.05) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(4.0) * kp.x(0);
        kp.q(1) = static_cast<T>(0.2) * kp.x(1);
        kp.r(0) = static_cast<T>(0.1) * kp.u(0);
        kp.Q(0, 0) = 4.0;
        kp.Q(0, 1) = 0.0;
        kp.Q(1, 0) = 0.0;
        kp.Q(1, 1) = 0.2;
        kp.R(0, 0) = 0.1;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

// Box-constrained variant of a double-integrator that drives the control to
// the upper bound, producing materially non-zero inequality duals.
struct BoxConstrainedModel {
    static constexpr int NX = 2;
    static constexpr int NU = 1;
    static constexpr int NC = 2;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x", "v" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0, 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0, 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>&,
        double dt, IntegratorType)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + x(1) * dt;
        xn(1) = x(1) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.x(1) * dt;
        kp.f_resid(1) = kp.x(1) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.A(0, 1) = dt;
        kp.A(1, 0) = 0.0;
        kp.A(1, 1) = 1.0;
        kp.B(0, 0) = 0.0;
        kp.B(1, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.u(0) - 0.3;
        kp.g_val(1) = -kp.u(0) - 0.3;
        kp.C(0, 0) = 0.0;
        kp.C(0, 1) = 0.0;
        kp.C(1, 0) = 0.0;
        kp.C(1, 1) = 0.0;
        kp.D(0, 0) = 1.0;
        kp.D(1, 0) = -1.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = static_cast<T>(20.0) * (kp.x(0) - static_cast<T>(5.0))
                * (kp.x(0) - static_cast<T>(5.0))
            + static_cast<T>(0.05) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(40.0) * (kp.x(0) - static_cast<T>(5.0));
        kp.q(1) = static_cast<T>(0.0);
        kp.r(0) = static_cast<T>(0.1) * kp.u(0);
        kp.Q(0, 0) = 40.0;
        kp.Q(0, 1) = 0.0;
        kp.Q(1, 0) = 0.0;
        kp.Q(1, 1) = 0.0;
        kp.R(0, 0) = 0.1;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

bool acceptable(SolverStatus s)
{
    return s == SolverStatus::OPTIMAL || s == SolverStatus::FEASIBLE;
}

} // namespace

TEST(DirectionRefinementTest, DefaultConfigKeepsRefinementOff)
{
    SolverConfig config;
    EXPECT_EQ(config.direction_refinement, DirectionRefinementMode::NONE);
    EXPECT_GE(config.direction_refinement_max_passes, 1);
    EXPECT_GT(config.direction_refinement_tol, 0.0);
}

TEST(DirectionRefinementTest, ConfigValidationRejectsInvalidRefinementParameters)
{
    SolverConfig config;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::OK);

    config.direction_refinement_max_passes = 0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);

    config.direction_refinement_max_passes = 4;
    config.direction_refinement_tol = 0.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);

    config.direction_refinement_tol = -1.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
}

TEST(DirectionRefinementTest, IterativeRefinementSolvesUnconstrainedNonlinearProblem)
{
    constexpr int N = 12;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.direction_refinement = DirectionRefinementMode::FULL_KKT_ITERATIVE_REFINEMENT;
    config.direction_refinement_max_passes = 4;
    config.direction_refinement_tol = 1e-12;
    config.barrier_strategy = BarrierStrategy::MONOTONE;

    MiniSolver<PendulumModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    EXPECT_EQ(solver.set_initial_state("theta", 0.6), ApiStatus::OK);
    EXPECT_EQ(solver.set_initial_state("omega", 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    const SolverStatus status = solver.solve();
    EXPECT_TRUE(acceptable(status)) << status_to_string(status);
    const SolverInfo& info = solver.get_info();
    EXPECT_GE(info.direction_refinement_passes, info.iterations)
        << "Each accepted iteration must consume at least one refinement pass when the iterative "
           "mode is opted in";
    const int max_total = info.iterations * config.direction_refinement_max_passes;
    EXPECT_LE(info.direction_refinement_passes, max_total)
        << "Total refinement passes must be bounded by iterations * max_passes";
    EXPECT_GE(info.direction_refinement_last_defect, 0.0);
    EXPECT_TRUE(std::isfinite(info.direction_refinement_last_defect));
}

TEST(DirectionRefinementTest, IterativeRefinementMatchesSinglePassOnConstrainedProblem)
{
    constexpr int N = 10;
    SolverConfig single_pass;
    single_pass.print_level = PrintLevel::NONE;
    single_pass.direction_refinement = DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT;
    single_pass.barrier_strategy = BarrierStrategy::ADAPTIVE;

    SolverConfig iterative = single_pass;
    iterative.direction_refinement = DirectionRefinementMode::FULL_KKT_ITERATIVE_REFINEMENT;
    iterative.direction_refinement_max_passes = 8;
    iterative.direction_refinement_tol = 1e-14;

    auto run = [N](const SolverConfig& cfg) {
        MiniSolver<BoxConstrainedModel, 16> solver(N, Backend::CPU_SERIAL, cfg);
        solver.set_dt(0.1);
        EXPECT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
        EXPECT_EQ(solver.set_initial_state("v", 0.0), ApiStatus::OK);
        solver.rollout_dynamics();
        const SolverStatus status = solver.solve();
        EXPECT_TRUE(acceptable(status)) << status_to_string(status);
        return solver.get_info();
    };

    const SolverInfo single_info = run(single_pass);
    const SolverInfo iter_info = run(iterative);

    // Auto-degrade contract: when active inequality duals are present,
    // FULL_KKT_ITERATIVE_REFINEMENT consumes the same single primal pass per
    // accepted iteration as DYNAMICS_DEFECT_ROLLOUT. This keeps the total
    // primal-only refinement work bounded by iterations and prevents OD-005
    // amplification.
    EXPECT_EQ(iter_info.direction_refinement_passes, iter_info.iterations)
        << "FULL_KKT_ITERATIVE_REFINEMENT must auto-degrade to one pass per iteration when active "
           "inequality duals are present";
    EXPECT_EQ(single_info.direction_refinement_passes, single_info.iterations)
        << "Single-pass DYNAMICS_DEFECT_ROLLOUT must consume exactly one refinement pass per "
           "iteration";
}

TEST(DirectionRefinementTest, NoneModeKeepsRefinementCountersAtZero)
{
    constexpr int N = 8;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.direction_refinement = DirectionRefinementMode::NONE;

    MiniSolver<PendulumModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    EXPECT_EQ(solver.set_initial_state("theta", 0.4), ApiStatus::OK);
    EXPECT_EQ(solver.set_initial_state("omega", 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    const SolverStatus status = solver.solve();
    EXPECT_TRUE(acceptable(status)) << status_to_string(status);
    EXPECT_EQ(solver.get_info().direction_refinement_passes, 0);
    EXPECT_DOUBLE_EQ(solver.get_info().direction_refinement_last_defect, 0.0);
}
