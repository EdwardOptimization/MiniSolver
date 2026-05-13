#include "bugfix_common.h"
#include <type_traits>
#include <utility>

namespace {
template <typename T, typename = void> struct HasTolGradMember : std::false_type { };

template <typename T>
struct HasTolGradMember<T, decltype((void)std::declval<T&>().tol_grad, void())> : std::true_type {
};
} // namespace

TEST(ConfigRegressionTest, SolverConfigDoesNotExposeDeadTolGrad)
{
    EXPECT_FALSE(HasTolGradMember<SolverConfig>::value)
        << "Stationarity is controlled by tol_dual; a separate tol_grad field was dead API.";
}

TEST(ConfigRegressionTest, NegativeHorizonRejected)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    EXPECT_THROW(
        (MiniSolver<BugTestModel, 10>(-3, Backend::CPU_SERIAL, config)), std::invalid_argument);
    EXPECT_THROW(
        (MiniSolver<BugTestModel, 10>(11, Backend::CPU_SERIAL, config)), std::invalid_argument);

    MiniSolver<BugTestModel, 10> solver(0, Backend::CPU_SERIAL, config);
    solver.resize_horizon(-1);
    EXPECT_EQ(solver.get_horizon(), 0);
}

TEST(ConfigRegressionTest, NegativeConstraintQueryReturnsZero)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    MiniSolver<BugTestModel, 10> solver(2, Backend::CPU_SERIAL, config);
    EXPECT_DOUBLE_EQ(solver.get_constraint_val(-1, 0), 0.0);
    EXPECT_DOUBLE_EQ(solver.get_constraint_val(0, -1), 0.0);
}

struct ApiStatusTestModel {
    static const int NX = 2;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x", "y" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "p" };
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& p, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + dt * u(0);
        xn(1) = x(1) + dt * p(0);
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + dt * kp.u(0);
        kp.f_resid(1) = kp.x(1) + dt * kp.p(0);
        kp.A.setIdentity();
        kp.B.setZero();
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.u(0) - 1.0;
        kp.C.setZero();
        kp.D.setZero();
        kp.D(0, 0) = 1.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0) + kp.x(1) * kp.x(1) + kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.q(1) = 2.0 * kp.x(1);
        kp.r(0) = 2.0 * kp.u(0);
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
};

TEST(ConfigRegressionTest, ApiSettersReturnExplicitStatusAndDoNotMutate)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    MiniSolver<ApiStatusTestModel, 3> solver(2, Backend::CPU_SERIAL, config);

    EXPECT_EQ(solver.resize_horizon(4), ApiStatus::InvalidHorizon);
    EXPECT_EQ(solver.get_horizon(), 2);

    EXPECT_EQ(solver.set_initial_state(std::vector<double> { 1.0 }), ApiStatus::SizeMismatch);
    EXPECT_DOUBLE_EQ(solver.get_state(0, 0), 0.0);

    EXPECT_EQ(solver.set_initial_state("missing", 1.0), ApiStatus::UnknownName);
    EXPECT_EQ(solver.set_initial_state("x", std::numeric_limits<double>::infinity()),
        ApiStatus::NonFiniteValue);
    EXPECT_DOUBLE_EQ(solver.get_state(0, 0), 0.0);

    EXPECT_EQ(solver.set_parameter(3, 0, 1.0), ApiStatus::InvalidStage);
    EXPECT_EQ(solver.set_parameter(0, 2, 1.0), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.set_parameter(0, "missing", 1.0), ApiStatus::UnknownName);
    EXPECT_DOUBLE_EQ(solver.get_parameter(0, 0), 0.0);

    EXPECT_EQ(solver.set_state_guess(-1, 0, 2.0), ApiStatus::InvalidStage);
    EXPECT_EQ(solver.set_state_guess(0, 2, 2.0), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.set_state_guess(0, "missing", 2.0), ApiStatus::UnknownName);
    EXPECT_DOUBLE_EQ(solver.get_state(0, 0), 0.0);

    EXPECT_EQ(solver.set_control_guess(2, 0, 3.0), ApiStatus::TerminalControl);
    EXPECT_EQ(solver.set_control_guess(0, 1, 3.0), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.set_control_guess(0, "missing", 3.0), ApiStatus::UnknownName);
    EXPECT_DOUBLE_EQ(solver.get_control(0, 0), 0.0);

    EXPECT_EQ(solver.set_slack_guess(3, 0, 4.0), ApiStatus::InvalidStage);
    EXPECT_EQ(solver.set_dual_guess(0, 1, 4.0), ApiStatus::InvalidIndex);
    EXPECT_DOUBLE_EQ(solver.get_slack(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_dual(0, 0), 1.0);

    EXPECT_EQ(solver.set_dt(std::numeric_limits<double>::quiet_NaN()), ApiStatus::NonFiniteValue);
}

TEST(ConfigRegressionTest, CheckedScalarGettersReportInvalidAccess)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    MiniSolver<ApiStatusTestModel, 3> solver(2, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_initial_state("x", 1.25), ApiStatus::OK);
    ASSERT_EQ(solver.set_parameter(0, 0, 2.5), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, 0, -0.5), ApiStatus::OK);
    ASSERT_EQ(solver.set_slack_guess(0, 0, 3.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_dual_guess(0, 0, 4.0), ApiStatus::OK);

    double value = -99.0;
    EXPECT_EQ(solver.get_state(0, 0, value), ApiStatus::OK);
    EXPECT_DOUBLE_EQ(value, 1.25);

    value = -99.0;
    EXPECT_EQ(solver.get_parameter(0, 0, value), ApiStatus::OK);
    EXPECT_DOUBLE_EQ(value, 2.5);

    value = -99.0;
    EXPECT_EQ(solver.get_control(0, 0, value), ApiStatus::OK);
    EXPECT_DOUBLE_EQ(value, -0.5);

    value = -99.0;
    EXPECT_EQ(solver.get_slack(0, 0, value), ApiStatus::OK);
    EXPECT_DOUBLE_EQ(value, 3.0);

    value = -99.0;
    EXPECT_EQ(solver.get_dual(0, 0, value), ApiStatus::OK);
    EXPECT_DOUBLE_EQ(value, 4.0);

    value = -99.0;
    EXPECT_EQ(solver.get_state(3, 0, value), ApiStatus::InvalidStage);
    EXPECT_DOUBLE_EQ(value, -99.0);

    EXPECT_EQ(solver.get_state(0, 2, value), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.get_control(2, 0, value), ApiStatus::TerminalControl);
    EXPECT_EQ(solver.get_parameter(0, 2, value), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.get_slack(-1, 0, value), ApiStatus::InvalidStage);
    EXPECT_EQ(solver.get_dual(0, 2, value), ApiStatus::InvalidIndex);
}

TEST(ConfigRegressionTest, SetConfigPreservesBackendInvariant)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;

    MiniSolver<BugTestModel, 10> solver(3, Backend::GPU_MPX, conf);
    ASSERT_EQ(solver.get_config().backend, Backend::GPU_MPX);

    SolverConfig new_conf;
    new_conf.print_level = PrintLevel::NONE;
    new_conf.backend = Backend::GPU_PCR;
    EXPECT_EQ(solver.set_config(new_conf), ApiStatus::OK);

    EXPECT_EQ(solver.get_config().backend, Backend::GPU_MPX)
        << "set_config must preserve the constructor-set backend invariant";
}

TEST(ConfigRegressionTest, SetConfigRejectsInvalidConfigWithoutMutation)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    conf.max_iters = 5;
    conf.default_dt = 0.1;

    MiniSolver<BugTestModel, 10> solver(3, Backend::CPU_SERIAL, conf);

    SolverConfig invalid = solver.get_config();
    invalid.max_iters = -1;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_EQ(solver.get_config().max_iters, 5);

    invalid = solver.get_config();
    invalid.default_dt = std::numeric_limits<double>::infinity();
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::NonFiniteValue);
    EXPECT_DOUBLE_EQ(solver.get_config().default_dt, 0.1);

    invalid = solver.get_config();
    invalid.constraint_scaling = ConstraintScalingMethod::ROW_INF_NORM;
    invalid.constraint_row_scale_min = 0.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_EQ(solver.get_config().constraint_scaling, ConstraintScalingMethod::NONE);

    invalid = solver.get_config();
    invalid.restoration_sufficient_decrease_factor = 1.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_LT(solver.get_config().restoration_sufficient_decrease_factor, 1.0);

    invalid = solver.get_config();
    invalid.warm_start_slack_init = 0.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_GT(solver.get_config().warm_start_slack_init, 0.0);

    invalid = solver.get_config();
    invalid.line_search_backtrack_factor = 1.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_LT(solver.get_config().line_search_backtrack_factor, 1.0);

    invalid = solver.get_config();
    invalid.reg_scale_up = 1.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_GT(solver.get_config().reg_scale_up, 1.0);

    invalid = solver.get_config();
    invalid.reg_scale_down = 1.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_GT(solver.get_config().reg_scale_down, 1.0);
}

TEST(ConfigRegressionTest, SetConfigRejectsInvalidGlobalizationParameters)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    MiniSolver<BugTestModel, 10> solver(3, Backend::CPU_SERIAL, conf);

    auto expect_invalid = [&](const SolverConfig& invalid) {
        const SolverConfig before = solver.get_config();
        EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
        EXPECT_DOUBLE_EQ(solver.get_config().armijo_c1, before.armijo_c1);
        EXPECT_DOUBLE_EQ(solver.get_config().filter_gamma_theta, before.filter_gamma_theta);
        EXPECT_DOUBLE_EQ(solver.get_config().soc_trigger_alpha, before.soc_trigger_alpha);
    };

    SolverConfig invalid = solver.get_config();
    invalid.armijo_c1 = -1e-4;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.armijo_c1 = 1.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.eta_suff_descent = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.eta_suff_descent = 1.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.filter_gamma_theta = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.filter_gamma_phi = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.filter_theta_max_factor = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.soc_trigger_alpha = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.barrier_inf_cost = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.restoration_mu = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.restoration_reg = -1e-2;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.tol_cost = -1e-8;
    expect_invalid(invalid);
}

TEST(ConfigRegressionTest, SetConfigRejectsInvalidNumericalControlParameters)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    MiniSolver<BugTestModel, 10> solver(3, Backend::CPU_SERIAL, conf);

    auto expect_invalid = [&](const SolverConfig& invalid) {
        const SolverConfig before = solver.get_config();
        EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
        EXPECT_EQ(solver.get_config().linear_solve_max_attempts, before.linear_solve_max_attempts);
        EXPECT_DOUBLE_EQ(solver.get_config().tol_con, before.tol_con);
        EXPECT_DOUBLE_EQ(solver.get_config().huge_penalty, before.huge_penalty);
    };

    SolverConfig invalid = solver.get_config();
    invalid.linear_solve_max_attempts = 0;
    expect_invalid(invalid);

    SolverConfig single_attempt = solver.get_config();
    single_attempt.linear_solve_max_attempts = 1;
    EXPECT_EQ(solver.set_config(single_attempt), ApiStatus::OK);
    EXPECT_EQ(solver.get_config().linear_solve_max_attempts, 1);

    invalid = solver.get_config();
    invalid.tol_con = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.tol_dual = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.tol_mu = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.feasible_tol_scale = 0.5;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.slack_reset_trigger = -1e-3;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.merit_nu_init = 0.0;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.singular_threshold = -1e-4;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.regularization_step = -1e-6;
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.huge_penalty = 0.0;
    expect_invalid(invalid);
}

TEST(ConfigRegressionTest, SetConfigRejectsInvalidEnumValues)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    MiniSolver<BugTestModel, 10> solver(3, Backend::CPU_SERIAL, conf);

    auto expect_invalid = [&](const SolverConfig& invalid) {
        const SolverConfig before = solver.get_config();
        EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
        EXPECT_EQ(solver.get_config().initialization, before.initialization);
        EXPECT_EQ(solver.get_config().line_search_type, before.line_search_type);
        EXPECT_EQ(solver.get_config().integrator, before.integrator);
    };

    SolverConfig invalid = solver.get_config();
    invalid.initialization = static_cast<InitializationMode>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.warm_start_barrier = static_cast<WarmStartBarrierMode>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.warm_start_regularization = static_cast<WarmStartRegularizationMode>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.termination_profile = static_cast<TerminationProfile>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.constraint_scaling = static_cast<ConstraintScalingMethod>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.objective_scaling = static_cast<ObjectiveScalingMethod>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.problem_scaling = static_cast<ProblemScalingMethod>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.integrator = static_cast<IntegratorType>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.barrier_strategy = static_cast<BarrierStrategy>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.inertia_strategy = static_cast<InertiaStrategy>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.line_search_type = static_cast<LineSearchType>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.print_level = static_cast<PrintLevel>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.hessian_approximation = static_cast<HessianApproximation>(99);
    expect_invalid(invalid);

    invalid = solver.get_config();
    invalid.direction_refinement = static_cast<DirectionRefinementMode>(99);
    expect_invalid(invalid);

    EXPECT_THROW(
        (MiniSolver<BugTestModel, 10>(3, static_cast<Backend>(99), conf)), std::invalid_argument);
}

TEST(ConfigRegressionTest, SetConfigRejectsInvalidLineSearchBoundaryParameters)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    MiniSolver<BugTestModel, 10> solver(3, Backend::CPU_SERIAL, conf);

    SolverConfig invalid = solver.get_config();
    invalid.line_search_tau = 1.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_LT(solver.get_config().line_search_tau, 1.0);

    invalid = solver.get_config();
    invalid.line_search_tau = 0.0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_GT(solver.get_config().line_search_tau, 0.0);

    invalid = solver.get_config();
    invalid.line_search_type = LineSearchType::MERIT;
    invalid.line_search_max_iters = 0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_GT(solver.get_config().line_search_max_iters, 0);

    invalid = solver.get_config();
    invalid.line_search_type = LineSearchType::FILTER;
    invalid.line_search_max_iters = 0;
    EXPECT_EQ(solver.set_config(invalid), ApiStatus::InvalidArgument);
    EXPECT_GT(solver.get_config().line_search_max_iters, 0);

    SolverConfig no_line_search = solver.get_config();
    no_line_search.line_search_type = LineSearchType::NONE;
    no_line_search.line_search_max_iters = 0;
    EXPECT_EQ(solver.set_config(no_line_search), ApiStatus::OK);
    EXPECT_EQ(solver.get_config().line_search_max_iters, 0);
}

TEST(ConfigRegressionTest, ConstructorRejectsInvalidConfig)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = -1;
    EXPECT_THROW(
        (MiniSolver<BugTestModel, 10>(3, Backend::CPU_SERIAL, config)), std::invalid_argument);
}

TEST(ConfigRegressionTest, SetConfigDefersPlanRebuildUntilSolve)
{
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;
    conf.max_iters = 0;
    conf.line_search_type = LineSearchType::FILTER;

    MiniSolver<BugTestModel, 10> solver(3, Backend::CPU_SERIAL, conf);
    ASSERT_FALSE(Access::build_dirty(solver));
    ASSERT_EQ(Access::plan_line_search_type(solver), LineSearchType::FILTER);

    SolverConfig new_conf = solver.get_config();
    new_conf.line_search_type = LineSearchType::MERIT;
    solver.set_config(new_conf);

    EXPECT_TRUE(Access::build_dirty(solver));
    EXPECT_EQ(Access::plan_line_search_type(solver), LineSearchType::FILTER)
        << "set_config should only mark the build state dirty, not rebuild immediately";

    (void)solver.solve();

    EXPECT_FALSE(Access::build_dirty(solver));
    EXPECT_EQ(Access::plan_line_search_type(solver), LineSearchType::MERIT);
    EXPECT_EQ(Access::plan_backend(solver), Backend::CPU_SERIAL);
}

TEST(ConfigRegressionTest, DefaultWarmStartResetsBarrierAndRegularization)
{
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.max_iters = 0;
    config.mu_init = 1e-1;
    config.reg_init = 1e-4;

    MiniSolver<BugTestModel, 10> solver(2, Backend::CPU_SERIAL, config);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        solver.set_slack_guess(k, 0, 1e-3);
        solver.set_dual_guess(k, 0, 2e-3);
    }
    Access::mu(solver) = 2e-6;
    Access::reg(solver) = 3e-2;

    (void)solver.solve();

    EXPECT_DOUBLE_EQ(Access::mu(solver), config.mu_init);
    EXPECT_DOUBLE_EQ(Access::reg(solver), config.reg_init);
}

TEST(ConfigRegressionTest, WarmStartCanReusePreviousBarrier)
{
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.warm_start_barrier = WarmStartBarrierMode::REUSE_PREVIOUS_MU;
    config.max_iters = 0;
    config.mu_init = 1e-1;
    config.mu_final = 1e-8;

    MiniSolver<BugTestModel, 10> solver(2, Backend::CPU_SERIAL, config);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        solver.set_slack_guess(k, 0, 1e-3);
        solver.set_dual_guess(k, 0, 2e-3);
    }
    Access::mu(solver) = 2e-5;

    (void)solver.solve();

    EXPECT_DOUBLE_EQ(Access::mu(solver), 2e-5);
}

TEST(ConfigRegressionTest, WarmStartCanUseComplementarityGapBarrier)
{
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.warm_start_barrier = WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP;
    config.max_iters = 0;
    config.mu_init = 1e-1;
    config.mu_final = 1e-8;

    MiniSolver<BugTestModel, 10> solver(2, Backend::CPU_SERIAL, config);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        solver.set_slack_guess(k, 0, 1e-2);
        solver.set_dual_guess(k, 0, 3e-3);
    }
    Access::mu(solver) = 1e-6;

    (void)solver.solve();

    EXPECT_NEAR(Access::mu(solver), 3e-5, 1e-14);
}

TEST(ConfigRegressionTest, WarmStartComplementarityGapIncludesL1SoftPair)
{
    using Access = minisolver::test::SolverInternalAccess<L1TestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.warm_start_barrier = WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP;
    config.max_iters = 0;
    config.mu_init = 1e-1;
    config.mu_final = 1e-8;

    MiniSolver<L1TestModel, 10> solver(1, Backend::CPU_SERIAL, config);
    auto& traj = Access::get_trajectory(solver);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        traj[k].s(0) = 1e-2;
        traj[k].lam(0) = 3e-3;
        traj[k].soft_s(0) = 4.000120003600108e-7; // soft_s * (100 - 0.003) = 4e-5
    }

    (void)solver.solve();

    EXPECT_NEAR(Access::mu(solver), 3.5e-5, 1e-13);
}

TEST(ConfigRegressionTest, WarmStartInvalidPrimalDualFallsBackToMuInit)
{
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.warm_start_barrier = WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP;
    config.max_iters = 0;
    config.mu_init = 1e-1;

    MiniSolver<BugTestModel, 10> solver(2, Backend::CPU_SERIAL, config);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        solver.set_slack_guess(k, 0, -1.0);
        solver.set_dual_guess(k, 0, 2e-3);
    }
    Access::mu(solver) = 2e-5;

    (void)solver.solve();

    EXPECT_DOUBLE_EQ(Access::mu(solver), config.mu_init);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        EXPECT_GT(solver.get_slack(k)[0], 0.0);
        EXPECT_GT(solver.get_dual(k)[0], 0.0);
    }
}

TEST(ConfigRegressionTest, WarmStartRegularizationModesAreExplicit)
{
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.warm_start_regularization = WarmStartRegularizationMode::DECAY_PREVIOUS_REG;
    config.max_iters = 0;
    config.reg_init = 1e-4;
    config.reg_min = 1e-8;
    config.reg_scale_down = 10.0;

    MiniSolver<BugTestModel, 10> solver(2, Backend::CPU_SERIAL, config);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        solver.set_slack_guess(k, 0, 1e-3);
        solver.set_dual_guess(k, 0, 2e-3);
    }
    Access::reg(solver) = 1e-2;

    (void)solver.solve();

    EXPECT_DOUBLE_EQ(Access::reg(solver), 1e-3);
}

struct CallbackUpdateModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "reference" };
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    inline static int cost_call_count = 0;
    inline static double first_cost_reference = -1.0;

    static void reset_observations()
    {
        cost_call_count = 0;
        first_cost_reference = -1.0;
    }

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + dt * u(0);
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + dt * kp.u(0);
        kp.A.setIdentity();
        kp.B.setZero();
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val.setZero();
        kp.C.setZero();
        kp.D.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        if (cost_call_count == 0) {
            first_cost_reference = static_cast<double>(kp.p(0));
        }
        ++cost_call_count;

        const T error = kp.x(0) - kp.p(0);
        kp.cost = error * error + T(0.1) * kp.u(0) * kp.u(0);
        kp.q(0) = T(2.0) * error;
        kp.r(0) = T(0.2) * kp.u(0);
        kp.Q.setZero();
        kp.Q(0, 0) = T(2.0);
        kp.R.setZero();
        kp.R(0, 0) = T(0.2);
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct CallbackUpdateState {
    int calls = 0;
    double reference = 0.0;
};

using CallbackSolver = MiniSolver<CallbackUpdateModel, 4>;

ApiStatus update_reference_before_evaluation(CallbackSolver& solver, void* user)
{
    auto* state = static_cast<CallbackUpdateState*>(user);
    ++state->calls;
    return solver.set_global_parameter("reference", state->reference);
}

ApiStatus fail_model_update_callback(CallbackSolver& /*solver*/, void* user)
{
    auto* state = static_cast<CallbackUpdateState*>(user);
    ++state->calls;
    return ApiStatus::InvalidArgument;
}

TEST(ConfigRegressionTest, ModelUpdateCallbackRunsBeforeFirstEvaluation)
{
    CallbackUpdateModel::reset_observations();

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::NONE;
    config.max_iters = 1;

    CallbackSolver solver(2, Backend::CPU_SERIAL, config);
    CallbackUpdateState state;
    state.reference = 3.5;

    ASSERT_EQ(solver.set_model_update_callback(update_reference_before_evaluation, &state),
        ApiStatus::OK);

    (void)solver.solve();

    EXPECT_EQ(state.calls, 2);
    EXPECT_DOUBLE_EQ(solver.get_parameter(0, "reference"), 3.5);
    EXPECT_DOUBLE_EQ(CallbackUpdateModel::first_cost_reference, 3.5);
}

TEST(ConfigRegressionTest, ModelUpdateCallbackFailureStopsSolveAsInvalidInput)
{
    CallbackUpdateModel::reset_observations();

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 3;

    CallbackSolver solver(2, Backend::CPU_SERIAL, config);
    CallbackUpdateState state;

    ASSERT_EQ(solver.set_model_update_callback(fail_model_update_callback, &state), ApiStatus::OK);

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(state.calls, 1);
    EXPECT_EQ(solver.get_info().loop_status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(solver.get_info().termination_reason, TerminationReason::INVALID_INPUT);
    EXPECT_EQ(CallbackUpdateModel::cost_call_count, 0);
}

ApiStatus fail_model_update_after_presolve(CallbackSolver& solver, void* user)
{
    auto* state = static_cast<CallbackUpdateState*>(user);
    ++state->calls;
    if (solver.get_iteration_count() == 0) {
        return ApiStatus::OK;
    }
    return ApiStatus::InvalidArgument;
}

ApiStatus dirty_plan_during_iteration_callback(CallbackSolver& solver, void* user)
{
    auto* state = static_cast<CallbackUpdateState*>(user);
    ++state->calls;
    if (solver.get_iteration_count() == 0) {
        return ApiStatus::OK;
    }

    SolverConfig config = solver.get_config();
    config.max_iters = 1;
    return solver.set_config(config);
}

TEST(ConfigRegressionTest, ModelUpdateCallbackFailureDuringIterationStopsBeforeEvaluation)
{
    CallbackUpdateModel::reset_observations();

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 2;

    CallbackSolver solver(2, Backend::CPU_SERIAL, config);
    CallbackUpdateState state;

    ASSERT_EQ(
        solver.set_model_update_callback(fail_model_update_after_presolve, &state), ApiStatus::OK);

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(state.calls, 2);
    EXPECT_EQ(solver.get_info().loop_status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(solver.get_info().termination_reason, TerminationReason::INVALID_INPUT);
}

TEST(ConfigRegressionTest, ModelUpdateCallbackDirtyPlanDuringIterationIsRejected)
{
    CallbackUpdateModel::reset_observations();

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 2;

    CallbackSolver solver(2, Backend::CPU_SERIAL, config);
    CallbackUpdateState state;

    ASSERT_EQ(solver.set_model_update_callback(dirty_plan_during_iteration_callback, &state),
        ApiStatus::OK);

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(state.calls, 2);
    EXPECT_EQ(solver.get_info().loop_status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(solver.get_info().termination_reason, TerminationReason::INVALID_INPUT);
}

struct PresolveCallbackConstraintModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "g_shift" };
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + dt * u(0);
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + dt * kp.u(0);
        kp.A.setIdentity();
        kp.B.setZero();
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.p(0);
        kp.C.setZero();
        kp.D.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.u(0) * kp.u(0);
        kp.q.setZero();
        kp.r(0) = T(2.0) * kp.u(0);
        kp.Q.setZero();
        kp.R.setZero();
        kp.R(0, 0) = T(2.0);
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

using PresolveCallbackSolver = MiniSolver<PresolveCallbackConstraintModel, 4>;

ApiStatus update_presolve_constraint_parameter(PresolveCallbackSolver& solver, void* user)
{
    auto* state = static_cast<CallbackUpdateState*>(user);
    ++state->calls;
    return solver.set_global_parameter("g_shift", state->reference);
}

TEST(ConfigRegressionTest, ModelUpdateCallbackRunsBeforePresolveSlackInitialization)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0;
    config.mu_init = 1e-1;

    PresolveCallbackSolver solver(2, Backend::CPU_SERIAL, config);
    CallbackUpdateState state;
    state.reference = 4.0;

    ASSERT_EQ(solver.set_model_update_callback(update_presolve_constraint_parameter, &state),
        ApiStatus::OK);

    (void)solver.solve();

    EXPECT_EQ(state.calls, 1);
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        EXPECT_DOUBLE_EQ(solver.get_parameter(k, "g_shift"), 4.0);
        EXPECT_DOUBLE_EQ(solver.get_slack(k, 0), 4.0);
        EXPECT_DOUBLE_EQ(solver.get_dual(k, 0), config.mu_init / 4.0);
    }
}
