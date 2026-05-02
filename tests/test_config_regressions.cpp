#include "bugfix_common.h"

TEST(ConfigRegressionTest, NegativeHorizonRejected)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    MiniSolver<BugTestModel, 10> solver(-3, Backend::CPU_SERIAL, config);
    EXPECT_EQ(solver.get_horizon(), 0);

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

TEST(ConfigRegressionTest, SetConfigPreservesBackendInvariant)
{
    SolverConfig conf;
    conf.print_level = PrintLevel::NONE;

    MiniSolver<BugTestModel, 10> solver(3, Backend::GPU_MPX, conf);
    ASSERT_EQ(solver.get_config().backend, Backend::GPU_MPX);

    SolverConfig new_conf;
    new_conf.print_level = PrintLevel::NONE;
    new_conf.backend = Backend::GPU_PCR;
    solver.set_config(new_conf);

    EXPECT_EQ(solver.get_config().backend, Backend::GPU_MPX)
        << "set_config must preserve the constructor-set backend invariant";
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
