#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <gtest/gtest.h>

using namespace minisolver;

namespace {

struct RtiLiteTrackingModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 0;
    static constexpr int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "x_ref" };
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>&,
        double dt, IntegratorType)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff = kp.x(0) - kp.p(0);
        kp.cost = static_cast<T>(5.0) * diff * diff + static_cast<T>(0.25) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(10.0) * diff;
        kp.r(0) = static_cast<T>(0.5) * kp.u(0);
        kp.Q(0, 0) = 10.0;
        kp.R(0, 0) = 0.5;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

void configure_tracking(MiniSolver<RtiLiteTrackingModel, 16>& solver, int N, double x_ref)
{
    solver.set_dt(0.1);
    for (int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "x_ref", x_ref);
    }
    solver.rollout_dynamics();
}

bool acceptable(SolverStatus s)
{
    return s == SolverStatus::OPTIMAL || s == SolverStatus::FEASIBLE;
}

} // namespace

TEST(RtiLiteTest, DisabledRtiLiteLeavesDiagnosticsAtBaseline)
{
    constexpr int N = 8;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_rti_lite = false;

    MiniSolver<RtiLiteTrackingModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0);
    configure_tracking(solver, N, 0.0);

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();
    ASSERT_TRUE(acceptable(status));
    EXPECT_FALSE(info.rti_lite_reused_linearization);
    EXPECT_EQ(info.rti_lite_linearization_age, 0);

    // Second solve must also stay at age 0 -- RTI-lite is opt-in.
    solver.set_initial_state("x", 0.95);
    const SolverStatus second = solver.solve();
    ASSERT_TRUE(acceptable(second));
    EXPECT_FALSE(solver.get_info().rti_lite_reused_linearization);
    EXPECT_EQ(solver.get_info().rti_lite_linearization_age, 0);
}

TEST(RtiLiteTest, EnabledRtiLiteReusesLinearizationOnSmallStateDelta)
{
    constexpr int N = 8;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_rti_lite = true;
    config.rti_lite_max_linearization_age = 3;
    config.rti_lite_max_state_delta = 0.5;

    MiniSolver<RtiLiteTrackingModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0);
    configure_tracking(solver, N, 0.0);

    // First solve cannot reuse anything (no previous solve recorded).
    const SolverStatus first = solver.solve();
    ASSERT_TRUE(acceptable(first));
    EXPECT_FALSE(solver.get_info().rti_lite_reused_linearization)
        << "First solve has no previous iterate; reuse must be false";
    EXPECT_EQ(solver.get_info().rti_lite_linearization_age, 0);

    // Tiny delta: the next solve must reuse and bump the age.
    solver.set_initial_state("x", 0.98);
    const SolverStatus second = solver.solve();
    ASSERT_TRUE(acceptable(second));
    EXPECT_TRUE(solver.get_info().rti_lite_reused_linearization)
        << "Small state delta must trigger RTI-lite reuse";
    EXPECT_EQ(solver.get_info().rti_lite_linearization_age, 1);

    solver.set_initial_state("x", 0.96);
    ASSERT_TRUE(acceptable(solver.solve()));
    EXPECT_TRUE(solver.get_info().rti_lite_reused_linearization);
    EXPECT_EQ(solver.get_info().rti_lite_linearization_age, 2);
}

TEST(RtiLiteTest, LargeStateDeltaResetsLinearizationAge)
{
    constexpr int N = 8;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_rti_lite = true;
    config.rti_lite_max_linearization_age = 5;
    config.rti_lite_max_state_delta = 0.1;

    MiniSolver<RtiLiteTrackingModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0);
    configure_tracking(solver, N, 0.0);
    ASSERT_TRUE(acceptable(solver.solve()));

    // Bump the state by way more than rti_lite_max_state_delta.
    solver.set_initial_state("x", 5.0);
    ASSERT_TRUE(acceptable(solver.solve()));
    EXPECT_FALSE(solver.get_info().rti_lite_reused_linearization)
        << "State delta exceeded the gate; RTI-lite must fall back";
    EXPECT_EQ(solver.get_info().rti_lite_linearization_age, 0);
}

TEST(RtiLiteTest, AgeBudgetForcesFullSolveAfterReuseLimit)
{
    constexpr int N = 6;
    constexpr int max_age = 2;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_rti_lite = true;
    config.rti_lite_max_linearization_age = max_age;
    config.rti_lite_max_state_delta = 5.0;

    MiniSolver<RtiLiteTrackingModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0);
    configure_tracking(solver, N, 0.0);
    ASSERT_TRUE(acceptable(solver.solve()));

    for (int i = 1; i <= max_age; ++i) {
        solver.set_initial_state("x", 1.0 + 0.01 * static_cast<double>(i));
        ASSERT_TRUE(acceptable(solver.solve()));
        EXPECT_TRUE(solver.get_info().rti_lite_reused_linearization);
        EXPECT_EQ(solver.get_info().rti_lite_linearization_age, i);
    }

    // The next solve hits the age budget and must fall back to a full solve.
    solver.set_initial_state("x", 1.05);
    ASSERT_TRUE(acceptable(solver.solve()));
    EXPECT_FALSE(solver.get_info().rti_lite_reused_linearization)
        << "Linearization age budget exhausted; must force a full solve";
    EXPECT_EQ(solver.get_info().rti_lite_linearization_age, 0);
}

TEST(RtiLiteTest, SetConfigClearsRtiLiteHistory)
{
    constexpr int N = 6;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_rti_lite = true;
    config.rti_lite_max_state_delta = 5.0;
    config.rti_lite_max_linearization_age = 5;

    MiniSolver<RtiLiteTrackingModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0);
    configure_tracking(solver, N, 0.0);
    ASSERT_TRUE(acceptable(solver.solve()));

    solver.set_initial_state("x", 1.05);
    ASSERT_TRUE(acceptable(solver.solve()));
    EXPECT_TRUE(solver.get_info().rti_lite_reused_linearization);

    SolverConfig new_config = solver.get_config();
    new_config.tol_con *= 0.5; // any change triggers history reset
    ASSERT_EQ(solver.set_config(new_config), ApiStatus::OK);

    solver.set_initial_state("x", 1.10);
    ASSERT_TRUE(acceptable(solver.solve()));
    EXPECT_FALSE(solver.get_info().rti_lite_reused_linearization)
        << "set_config must invalidate the RTI-lite history";
    EXPECT_EQ(solver.get_info().rti_lite_linearization_age, 0);
}
