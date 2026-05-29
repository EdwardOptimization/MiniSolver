#include "bugfix_common.h"

namespace {

template <typename TrajArray>
class AlwaysFailRiccatiSolver : public RiccatiSolver<TrajArray, BugTestModel> {
public:
    int calls = 0;

    LinearSolveResult solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        ++calls;
        return false;
    }
};

template <typename TrajArray>
class StalledDualRiccatiSolver : public RiccatiSolver<TrajArray, BugTestModel> {
public:
    int calls = 0;

    LinearSolveResult solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        ++calls;
        for (int k = 0; k <= N; ++k) {
            traj[k].dx.setZero();
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
            traj[k].dsoft_s.setZero();
            traj[k].r_bar.fill(1.0);
        }
        return true;
    }
};

ApiStatus no_op_bug_model_callback(MiniSolver<BugTestModel, 10>& /*solver*/, void* user)
{
    if (user != nullptr) {
        ++(*static_cast<int*>(user));
    }
    return ApiStatus::OK;
}

} // namespace

TEST(TerminationTest, RtiFixedIterationDoesNotMaskLinearSolveFailure)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = AlwaysFailRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.termination_profile = TerminationProfile::RTI_FIXED_ITERATION;
    config.max_iters = 10;

    Solver solver(N, Backend::CPU_SERIAL, config);
    Access::set_linear_solver(solver, std::make_unique<FakeSolver>());

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::LINEAR_SOLVE_FAILED)
        << "RTI fixed-iteration mode must not hide fatal direction-solve failures";
    EXPECT_EQ(solver.get_info().loop_status, SolverStatus::LINEAR_SOLVE_FAILED);
    EXPECT_EQ(solver.get_info().termination_reason, TerminationReason::LINEAR_SOLVE_FAILED);
}

TEST(TerminationTest, AcceptableNmpcPrimalFeasibleSkipsDirectionFailure)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = AlwaysFailRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.max_iters = 10;
    config.tol_con = 1e-9;

    Solver solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(0.1), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, 0, 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::FEASIBLE)
        << "ACCEPTABLE_NMPC should accept an already primal-feasible warm start "
           "before attempting a direction solve.";
    EXPECT_EQ(solver.get_info().loop_status, SolverStatus::FEASIBLE);
    EXPECT_EQ(solver.get_info().termination_reason, TerminationReason::PRIMAL_FEASIBLE);
    EXPECT_LE(solver.get_info().primal_inf, config.tol_con);
    EXPECT_EQ(fake_solver_ptr->calls, 0)
        << "The linear solver should not be called when ACCEPTABLE_NMPC already has "
           "fresh primal feasibility.";
}

TEST(TerminationTest, AcceptableNmpcCallbackDoesNotSkipDirectionSolve)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = AlwaysFailRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.max_iters = 10;
    config.tol_con = 1e-9;

    Solver solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(0.1), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, 0, 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    int callback_calls = 0;
    ASSERT_EQ(
        solver.set_model_update_callback(no_op_bug_model_callback, &callback_calls), ApiStatus::OK);

    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::LINEAR_SOLVE_FAILED)
        << "With a model-update callback installed, ACCEPTABLE_NMPC should not zero-step "
           "accept a warm start before trying to respond to callback-updated problem data.";
    EXPECT_GT(fake_solver_ptr->calls, 0);
    EXPECT_GT(callback_calls, 0);
}

TEST(TerminationTest, AcceptableNmpcInvalidReuseGuessDoesNotSkipDirectionSolve)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = AlwaysFailRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.max_iters = 10;
    config.tol_con = 1e-9;

    Solver solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(0.1), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, 0, 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    // The primal trajectory is feasible, but the stored primal-dual guess is not
    // reusable. ACCEPTABLE_NMPC must not treat this as a valid warm start.
    ASSERT_EQ(solver.set_dual_guess(0, 0, 0.0), ApiStatus::OK);

    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::LINEAR_SOLVE_FAILED)
        << "An effectively cold REUSE_PRIMAL_DUAL solve should still attempt a direction "
           "solve even if the primal trajectory is already feasible.";
    EXPECT_GT(fake_solver_ptr->calls, 0);
}

TEST(TerminationTest, ResidualStagnationLoopStatusWaitsForPostsolveFeasibleQuality)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = StalledDualRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::NONE;
    config.max_iters = 50;
    config.tol_con = 1e-9;
    config.tol_dual = 1e-9;
    config.tol_mu = 1e-9;
    config.tol_cost = 0.0;
    config.feasible_tol_scale = 10.0;
    config.residual_stagnation_min_iters = 2;
    config.residual_stagnation_window = 2;
    config.residual_stagnation_rel_tol = 1e-3;

    Solver solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(0.1), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, 0, 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_EQ(status, SolverStatus::FEASIBLE)
        << "Postsolve may still classify a residual-stagnated iterate as primal-feasible.";
    EXPECT_EQ(info.loop_status, SolverStatus::INSUFFICIENT_PROGRESS)
        << "Residual stagnation is a loop-stop reason, not a loop-level solution-quality "
           "certificate.";
    EXPECT_EQ(info.termination_reason, TerminationReason::RESIDUAL_STAGNATION);
    EXPECT_LT(info.iterations, config.max_iters);
    EXPECT_GT(fake_solver_ptr->calls, 0);
}

TEST(TerminationTest, AcceptableNmpcColdStartDoesNotSkipDirectionSolve)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = AlwaysFailRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.initialization = InitializationMode::COLD_START;
    config.max_iters = 10;
    config.tol_con = 1e-9;

    Solver solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(0.1), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, 0, 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::LINEAR_SOLVE_FAILED)
        << "Cold-start ACCEPTABLE_NMPC should still attempt at least one direction solve; "
           "pre-direction primal feasibility is only a warm-start shortcut.";
    EXPECT_GT(fake_solver_ptr->calls, 0);
}

TEST(TerminationTest, GenericInsufficientProgressReasonDoesNotPretendCostStagnation)
{
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    Solver solver(1, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(0.1), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, 0, 10.0), ApiStatus::OK);

    const SolverStatus status = Access::postsolve(solver, SolverStatus::INSUFFICIENT_PROGRESS);
    const SolverInfo& info = solver.get_info();

    EXPECT_EQ(status, SolverStatus::INSUFFICIENT_PROGRESS);
    EXPECT_EQ(info.loop_status, SolverStatus::INSUFFICIENT_PROGRESS);
    EXPECT_EQ(info.termination_reason, TerminationReason::INSUFFICIENT_PROGRESS)
        << "Generic insufficient-progress fallbacks must not be mislabeled as cost "
           "stagnation; concrete stagnation paths should set their own reason.";
}
