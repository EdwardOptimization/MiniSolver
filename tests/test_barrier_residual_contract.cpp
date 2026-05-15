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

    LinearSolveResult solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
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

struct InfConstraintPostsolveModel {
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
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double /*dt*/)
    {
        kp.f_resid(0) = kp.x(0);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = 0.0;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = std::numeric_limits<double>::infinity();
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 0.0;
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 1.0;
        kp.H(0, 0) = 0.0;
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
    EXPECT_NEAR(residuals.max_complementarity_gap, 0.25 * 0.5, 1e-14);
}

TEST(BarrierResidualContractTest, ConvergenceUsesTrueComplementarityGapSnapshot)
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
    residuals.max_barrier_complementarity_residual = 0.0;
    residuals.max_complementarity_gap = 5e-6;

    // Simulate a later barrier update. Termination must use the residual snapshot's
    // true complementarity gap, not infer quality from the current internal barrier target.
    Access::mu(solver) = config.mu_final;

    EXPECT_FALSE(Access::check_convergence(solver, residuals, 0.0))
        << "termination must interpret residuals with their true complementarity gap";
}

TEST(BarrierResidualContractTest, ConvergenceUsesKktComplementarityNotBarrierTarget)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.mu_final = 1e-8;
    config.tol_con = 1e-8;
    config.tol_dual = 1e-8;
    config.tol_mu = 1e-6;

    Solver solver(N, Backend::CPU_SERIAL, config);

    StepResidualSummary residuals;
    residuals.barrier_mu = 1e-3;
    residuals.max_primal_inf = 0.0;
    residuals.max_barrier_complementarity_residual = 1e-3;
    residuals.max_complementarity_gap = 0.0;

    EXPECT_TRUE(Access::check_convergence(solver, residuals, 0.0))
        << "termination should certify KKT quality from true complementarity, not require "
           "the internal barrier target to have reached mu_final";
}

TEST(BarrierResidualContractTest, ConvergenceRejectsLargeTrueComplementarityAtMuFinal)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.mu_final = 1e-8;
    config.tol_con = 1e-8;
    config.tol_dual = 1e-8;
    config.tol_mu = 1e-6;

    Solver solver(N, Backend::CPU_SERIAL, config);

    StepResidualSummary residuals;
    residuals.barrier_mu = config.mu_final;
    residuals.max_primal_inf = 0.0;
    residuals.max_barrier_complementarity_residual = 0.0;
    residuals.max_complementarity_gap = 1e-3;

    EXPECT_FALSE(Access::check_convergence(solver, residuals, 0.0))
        << "small barrier centrality residual is not enough when true complementarity is large";
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
    EXPECT_NEAR(residuals.max_complementarity_gap, 0.25 * 0.5, 1e-14);
}

TEST(BarrierResidualContractTest, PostsolveRejectsInfConstraintResidual)
{
    constexpr int N = 1;
    using Solver = MiniSolver<InfConstraintPostsolveModel, 4>;
    using Access = minisolver::test::SolverInternalAccess<InfConstraintPostsolveModel, 4>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;

    Solver solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    for (int k = 0; k <= N; ++k) {
        solver.set_state_guess(k, 0, 0.0);
        solver.set_slack_guess(k, 0, 1.0);
        solver.set_dual_guess(k, 0, 0.1);
    }
    solver.set_control_guess(0, 0, 0.0);

    const SolverStatus status = Access::postsolve(solver, SolverStatus::MAX_ITER);

    EXPECT_EQ(status, SolverStatus::NUMERICAL_ERROR)
        << "postsolve must reject Inf model residuals instead of classifying them";
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
