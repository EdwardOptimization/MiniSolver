#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

// Model: x in R^1
// Cost: x^2
// Con: x >= 2 AND x <= 1 (Impossible)
struct InfeasibleModel {
    static const int NX = 1;
    static const int NU = 1; // Dummy
    static const int NC = 2;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0, 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0, 0 }; // Hard

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& /*u*/,
        const MSVec<T, NP>& /*p*/, double /*dt*/, IntegratorType /*type*/)
    {
        return x; // Static
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double /*dt*/)
    {
        // Dynamics: x' = x (Static)
        kp.f_resid(0) = kp.x(0);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = 0.0;

        // Cost: x^2
        kp.cost = kp.x(0) * kp.x(0);
        kp.Q(0, 0) = 2.0;
        kp.q(0) = 2.0 * kp.x(0);

        // Constraints
        // 1. x >= 2  => 2 - x <= 0  => g0 = 2 - x
        // 2. x <= 1  => x - 1 <= 0  => g1 = x - 1
        kp.g_val(0) = 2.0 - kp.x(0);
        kp.g_val(1) = kp.x(0) - 1.0;

        // C = dg/dx
        kp.C(0, 0) = -1.0;
        kp.C(1, 0) = 1.0;

        kp.D.setZero();
    }

    // Explicit Exact/GN mapping
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

ApiStatus no_op_infeasible_model_callback(MiniSolver<InfeasibleModel, 10>& /*solver*/, void* user)
{
    if (user != nullptr) {
        ++(*static_cast<int*>(user));
    }
    return ApiStatus::OK;
}

// Feasible problem, but an intentionally inconsistent initial trajectory.
// With max_iters = 0, MiniSolver has only exhausted the iteration budget; it
// has not proven that the OCP is mathematically infeasible.
struct FeasibleBudgetModel {
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
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;

        kp.cost = kp.x(0) * kp.x(0) + 0.01 * kp.u(0) * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 0.02;
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 0.02 * kp.u(0);

        kp.g_val(0) = kp.u(0) - 1.0;
        kp.C(0, 0) = 0.0;
        kp.D(0, 0) = 1.0;
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

struct DegradedRiccatiModel {
    static const int NX = 1;
    static const int NU = 2;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u0", "u1" };
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
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
        kp.B(0, 1) = 0.0;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0);
        kp.Q(0, 0) = 2.0;
        kp.q(0) = 2.0 * kp.x(0);
        kp.R(0, 0) = 1.0;
        kp.R(1, 1) = -1000.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

TEST(StatusTest, StatusToStringUsesEnumNameForOptimal)
{
    EXPECT_STREQ(status_to_string(SolverStatus::OPTIMAL), "OPTIMAL");
}

TEST(StatusTest, ApiStatusHelpersExposeStableNames)
{
    EXPECT_TRUE(api_status_ok(ApiStatus::OK));
    EXPECT_FALSE(api_status_ok(ApiStatus::UnknownName));
    EXPECT_STREQ(api_status_to_string(ApiStatus::TerminalControl), "TerminalControl");
    EXPECT_STREQ(api_status_to_string(ApiStatus::NonFiniteValue), "NonFiniteValue");
}

TEST(StatusTest, IterationBudgetExhaustionIsNotInfeasibility)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0;

    MiniSolver<FeasibleBudgetModel, 10> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 10.0);

    SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_EQ(status, SolverStatus::MAX_ITER)
        << "A feasible model with zero iteration budget should report budget exhaustion, "
           "not mathematical infeasibility.";
    EXPECT_EQ(info.status, status);
    EXPECT_EQ(info.loop_status, SolverStatus::MAX_ITER);
    EXPECT_EQ(info.termination_reason, TerminationReason::MAX_ITERATIONS);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_TRUE(std::isfinite(info.primal_inf));
    EXPECT_TRUE(std::isfinite(info.dual_inf));
}

TEST(StatusTest, ConflictingConstraintsShouldEarlyExitBeforeMaxIter)
{
    constexpr int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 200;
    config.enable_feasibility_restoration = true; // Give it a chance to try restoration

    MiniSolver<InfeasibleModel, 10> solver(N, Backend::CPU_SERIAL, config);

    // x=1.5 violates both x >= 2 and x <= 1.
    solver.set_initial_state("x", 1.5);

    SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_NE(status, SolverStatus::MAX_ITER)
        << "Conflicting hard constraints should trigger an early failure path instead "
           "of spending the full iteration budget.";
    EXPECT_LT(info.iterations, config.max_iters);
    EXPECT_NE(info.loop_status, SolverStatus::MAX_ITER);
    EXPECT_NE(info.termination_reason, TerminationReason::MAX_ITERATIONS);
    EXPECT_EQ(status, SolverStatus::INSUFFICIENT_PROGRESS)
        << "Conflicting hard constraints should exit as residual stagnation, not as a "
           "certified infeasibility proof.";
    EXPECT_EQ(info.loop_status, SolverStatus::INSUFFICIENT_PROGRESS);
    EXPECT_EQ(info.termination_reason, TerminationReason::RESIDUAL_STAGNATION);
}

TEST(StatusTest, ResidualStagnationMonitorSkipsModelUpdateCallbacks)
{
    constexpr int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 20;
    config.enable_feasibility_restoration = false;

    MiniSolver<InfeasibleModel, 10> solver(N, Backend::CPU_SERIAL, config);
    int callback_calls = 0;
    ASSERT_EQ(solver.set_model_update_callback(no_op_infeasible_model_callback, &callback_calls),
        ApiStatus::OK);
    solver.set_initial_state("x", 1.5);

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_NE(info.termination_reason, TerminationReason::RESIDUAL_STAGNATION)
        << "Residual progress is not comparable when a model-update callback may change "
           "problem data between iterations.";
    EXPECT_GT(callback_calls, 0);
    EXPECT_EQ(callback_calls, info.iterations + 1)
        << "The callback should run once before presolve and once per solve iteration; "
           "this ensures callback-mode residual-stagnation suppression is exercised.";
    EXPECT_EQ(status, info.status);
}

TEST(StatusTest, RtiFixedIterationProfileStopsAfterOneIteration)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 10;
    config.termination_profile = TerminationProfile::RTI_FIXED_ITERATION;

    MiniSolver<FeasibleBudgetModel, 10> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);

    (void)solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_EQ(solver.get_iteration_count(), 1)
        << "RTI_FIXED_ITERATION should be selected through SolverConfig.";
    EXPECT_EQ(info.loop_status, SolverStatus::UNSOLVED);
    EXPECT_EQ(info.termination_reason, TerminationReason::FIXED_ITERATION);
    EXPECT_EQ(info.iterations, 1);
    EXPECT_TRUE(info.primal_inf >= 0.0);
    EXPECT_TRUE(info.dual_inf >= 0.0);
    EXPECT_DOUBLE_EQ(
        info.alpha, solver.get_alpha_log().empty() ? 1.0 : solver.get_alpha_log().back());
}

TEST(StatusTest, SolverInfoReportsDegradedRiccatiStep)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 1;
    config.line_search_type = LineSearchType::NONE;
    config.inertia_strategy = InertiaStrategy::REGULARIZATION;
    config.reg_init = 1e-9;
    config.reg_min = 1e-9;
    config.singular_threshold = 1e-4;

    MiniSolver<DegradedRiccatiModel, 10> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 1.0);
    (void)solver.solve();

    EXPECT_TRUE(solver.get_info().degraded_step)
        << "Small-NU Riccati freeze fallback should be visible in SolverInfo";
    EXPECT_GT(solver.get_info().degraded_riccati_freeze_count, 0)
        << "SolverInfo should expose how many Riccati stages used the freeze fallback";
}
