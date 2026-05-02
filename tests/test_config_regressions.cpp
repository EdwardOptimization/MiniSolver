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
