#include "bugfix_common.h"
#include "minisolver/algorithms/barrier_update.h"

namespace {
template <typename TrajArray>
class FixedDualResidualRiccatiSolver : public RiccatiSolver<TrajArray, BugTestModel> {
public:
    explicit FixedDualResidualRiccatiSolver(double dual_residual)
        : dual_residual_(dual_residual)
    {
    }

    bool solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        return true;
    }

    bool evaluate_dual_residual(TrajArray& /*scratch_traj*/, int /*N*/, double /*mu*/,
        double /*reg*/, InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        double& max_dual_inf) override
    {
        max_dual_inf = dual_residual_;
        return true;
    }

private:
    double dual_residual_ = 0.0;
};
}

TEST(BarrierResidualContractTest, StepResidualSummaryRecordsBarrierMuSnapshot)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    Solver solver(N, Backend::CPU_SERIAL, config);
    auto& traj = Access::get_trajectory(solver);
    for (int k = 0; k <= N; ++k) {
        traj[k].set_zero();
        traj[k].s(0) = 0.25;
        traj[k].lam(0) = 0.5;
    }

    Access::mu(solver) = 1e-2;
    const StepResidualSummary residuals = Access::evaluate_step_model(solver, traj);

    EXPECT_DOUBLE_EQ(residuals.barrier_mu, 1e-2);
    EXPECT_NEAR(residuals.max_barrier_complementarity_residual, std::abs(0.25 * 0.5 - 1e-2), 1e-14);
}

TEST(BarrierResidualContractTest, ConvergenceUsesResidualSnapshotMuNotCurrentSolverMu)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.mu_final = 1e-6;
    config.tol_con = 1e-8;
    config.tol_dual = 1e-8;
    config.tol_mu = 1e-8;

    Solver solver(N, Backend::CPU_SERIAL, config);

    StepResidualSummary residuals;
    residuals.barrier_mu = 1e-2;
    residuals.max_primal_inf = 0.0;
    residuals.max_barrier_complementarity_residual = 5e-6;

    // Simulate a barrier update after residual evaluation. The old wrapper
    // used context_.solve.mu and would incorrectly accept this stale snapshot.
    Access::mu(solver) = config.mu_final;

    EXPECT_FALSE(Access::check_convergence(solver, residuals, 0.0))
        << "termination must interpret residuals with residuals.barrier_mu";
}

TEST(BarrierResidualContractTest, PostsolveResidualsRecordBarrierMuSnapshot)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    Solver solver(N, Backend::CPU_SERIAL, config);
    auto& traj = Access::get_trajectory(solver);
    for (int k = 0; k <= N; ++k) {
        traj[k].set_zero();
        traj[k].s(0) = 0.25;
        traj[k].lam(0) = 0.5;
    }

    Access::mu(solver) = 2e-2;
    const PostsolveResiduals residuals = Access::refresh_postsolve_residuals(solver, traj);

    EXPECT_DOUBLE_EQ(residuals.barrier_mu, 2e-2);
    EXPECT_NEAR(residuals.max_barrier_complementarity_residual, std::abs(0.25 * 0.5 - 2e-2), 1e-14);
}

TEST(BarrierResidualContractTest, MehrotraTargetMuHandlesZeroCurrentMu)
{
    SolverConfig config;
    config.mu_final = 1e-6;

    const double target_from_zero_affine
        = detail::BarrierUpdateKernel::mehrotra_target_mu(config, 0.0, 0.0, 1.0);
    const double target_from_positive_affine
        = detail::BarrierUpdateKernel::mehrotra_target_mu(config, 0.0, 0.1, 1.0);

    EXPECT_TRUE(std::isfinite(target_from_zero_affine));
    EXPECT_TRUE(std::isfinite(target_from_positive_affine));
    EXPECT_DOUBLE_EQ(target_from_zero_affine, config.mu_final);
    EXPECT_DOUBLE_EQ(target_from_positive_affine, config.mu_final);
}

TEST(BarrierResidualContractTest, PostsolveRechecksDualResidualAfterLoopOptimal)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = FixedDualResidualRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.tol_dual = 1e-8;

    Solver solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    for (int k = 0; k <= N; ++k) {
        solver.set_state_guess(k, 0, 0.0);
        solver.set_slack_guess(k, 0, 1.0);
        solver.set_dual_guess(k, 0, 0.1);
    }
    solver.set_control_guess(0, 0, 0.0);

    Access::set_linear_solver(solver, std::make_unique<FakeSolver>(1.0));

    const SolverStatus status = Access::postsolve(solver, SolverStatus::OPTIMAL);
    EXPECT_EQ(status, SolverStatus::FEASIBLE)
        << "postsolve must not trust an in-loop OPTIMAL verdict with stale dual residuals";
}

TEST(BarrierResidualContractTest, PostsolveRechecksPrimalResidualAfterLoopOptimal)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = FixedDualResidualRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    Solver solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_state_guess(0, 0, 0.0);
    solver.set_state_guess(1, 0, 100.0); // large multiple-shooting defect
    solver.set_control_guess(0, 0, 0.0);
    for (int k = 0; k <= N; ++k) {
        solver.set_slack_guess(k, 0, 1.0);
        solver.set_dual_guess(k, 0, 0.1);
    }

    Access::set_linear_solver(solver, std::make_unique<FakeSolver>(0.0));

    const SolverStatus status = Access::postsolve(solver, SolverStatus::OPTIMAL);
    EXPECT_EQ(status, SolverStatus::INFEASIBLE)
        << "postsolve must not trust an in-loop OPTIMAL verdict with stale primal residuals";
}
