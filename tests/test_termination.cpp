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

struct AcceptedStepFeasibilityModel {
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

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0);
        kp.C(0, 0) = T(1);
        kp.D.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = T(0);
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

template <typename TrajArray>
class AcceptedStepFeasibilityRiccatiSolver
    : public RiccatiSolver<TrajArray, AcceptedStepFeasibilityModel> {
public:
    int calls = 0;

    LinearSolveResult solve(TrajArray& traj, int N, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/ = nullptr) override
    {
        ++calls;
        for (int k = 0; k <= N; ++k) {
            traj[k].dx(0) = -2.0;
            traj[k].du.setZero();
            traj[k].ds(0) = 1.0 - traj[k].s(0);
            traj[k].dlam.setZero();
            traj[k].dsoft_s.setZero();
            traj[k].r_bar.setZero();
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

TEST(TerminationTest, TinyStepStagnationDoesNotClaimStrictOptimality)
{
    SolverConfig config;
    config.tol_con = 1e-6;

    EXPECT_EQ(detail::TerminationKernel::classify_tiny_step_stagnation(config, 0.5e-6),
        SolverStatus::FEASIBLE)
        << "Tiny-step stagnation has no fresh complementarity proof; strict OPTIMAL must stay "
           "with the normal convergence/postsolve path.";
    EXPECT_EQ(detail::TerminationKernel::classify_tiny_step_stagnation(config, 2.0e-6),
        SolverStatus::UNSOLVED);
}

TEST(TerminationTest, ResidualStagnationMonitorRequiresConfiguredWindow)
{
    detail::ResidualStagnationMonitor monitor;
    const detail::ResidualStagnationConfigView config {
        true,
        false,
        false,
        1,
        2,
        1e-3,
        0.0,
        1e-3,
        1e-3,
        1e-3,
        10.0,
    };
    const detail::ResidualStagnationSample first {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        0,
    };
    EXPECT_EQ(monitor.update(first, config).status, SolverStatus::UNSOLVED);

    const detail::ResidualStagnationSample stalled {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        1,
    };
    EXPECT_EQ(monitor.update(stalled, config).status, SolverStatus::UNSOLVED);

    const detail::ResidualStagnationResult result = monitor.update(stalled, config);
    EXPECT_EQ(result.status, SolverStatus::INSUFFICIENT_PROGRESS);
    EXPECT_EQ(result.reason, TerminationReason::RESIDUAL_STAGNATION);
}

TEST(TerminationTest, ResidualStagnationMonitorHonorsMinIterations)
{
    detail::ResidualStagnationMonitor monitor;
    const detail::ResidualStagnationConfigView config {
        true,
        false,
        false,
        3,
        1,
        1e-3,
        0.0,
        1e-3,
        1e-3,
        1e-3,
        10.0,
    };
    const detail::ResidualStagnationSample first {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        0,
    };
    EXPECT_EQ(monitor.update(first, config).status, SolverStatus::UNSOLVED);

    const detail::ResidualStagnationSample before_min_iter {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        2,
    };
    EXPECT_EQ(monitor.update(before_min_iter, config).status, SolverStatus::UNSOLVED);
    EXPECT_EQ(monitor.stagnation_count(), 0);

    const detail::ResidualStagnationSample at_min_iter {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        3,
    };
    const detail::ResidualStagnationResult result = monitor.update(at_min_iter, config);
    EXPECT_EQ(result.status, SolverStatus::INSUFFICIENT_PROGRESS);
    EXPECT_EQ(result.reason, TerminationReason::RESIDUAL_STAGNATION);
}

TEST(TerminationTest, ResidualStagnationMonitorResetsOnFeasibleMuDecrease)
{
    detail::ResidualStagnationMonitor monitor;
    const detail::ResidualStagnationConfigView config {
        true,
        false,
        false,
        0,
        2,
        1e-3,
        0.0,
        1e-3,
        1e-3,
        1e-3,
        10.0,
    };
    const detail::ResidualStagnationSample first {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        0,
    };
    EXPECT_EQ(monitor.update(first, config).status, SolverStatus::UNSOLVED);

    const detail::ResidualStagnationSample stalled {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        1,
    };
    EXPECT_EQ(monitor.update(stalled, config).status, SolverStatus::UNSOLVED);
    EXPECT_EQ(monitor.stagnation_count(), 1);

    const detail::ResidualStagnationSample lower_mu {
        1e-4,
        1e-2,
        1e-2,
        5e-2,
        2,
    };
    EXPECT_EQ(monitor.update(lower_mu, config).status, SolverStatus::UNSOLVED);
    EXPECT_EQ(monitor.stagnation_count(), 0);
    EXPECT_TRUE(monitor.feasible_mode());

    EXPECT_EQ(monitor.update(lower_mu, config).status, SolverStatus::UNSOLVED);
    const detail::ResidualStagnationResult result = monitor.update(lower_mu, config);
    EXPECT_EQ(result.status, SolverStatus::INSUFFICIENT_PROGRESS);
    EXPECT_EQ(result.reason, TerminationReason::RESIDUAL_STAGNATION);
}

TEST(TerminationTest, ResidualStagnationMonitorSkipsCallbacksAndFixedIteration)
{
    detail::ResidualStagnationMonitor monitor;
    detail::ResidualStagnationConfigView config {
        true,
        false,
        true,
        0,
        1,
        1e-3,
        0.0,
        1e-3,
        1e-3,
        1e-3,
        10.0,
    };
    const detail::ResidualStagnationSample sample {
        1e-4,
        1e-2,
        1e-2,
        1e-1,
        3,
    };
    EXPECT_EQ(monitor.update(sample, config).status, SolverStatus::UNSOLVED);

    config.callback_installed = false;
    config.fixed_iteration_profile = true;
    EXPECT_EQ(monitor.update(sample, config).status, SolverStatus::UNSOLVED);
}

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

TEST(TerminationTest, LinearSolveRetriesEscalateRegularizationWithinBounds)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = AlwaysFailRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.max_iters = 1;
    config.linear_solve_max_attempts = 3;
    config.reg_init = 1e-4;
    config.reg_min = 1e-5;
    config.reg_max = 5e-3;
    config.reg_scale_up = 10.0;

    Solver solver(N, Backend::CPU_SERIAL, config);
    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    const SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::LINEAR_SOLVE_FAILED);
    EXPECT_EQ(fake_solver_ptr->calls, config.linear_solve_max_attempts);
    EXPECT_EQ(
        solver.get_info().regularization_escalation_count, config.linear_solve_max_attempts - 1);
    EXPECT_DOUBLE_EQ(Access::reg(solver), config.reg_max)
        << "Failed retries should scale regularization up and clamp at reg_max.";
}

TEST(TerminationTest, SuccessfulLinearSolveDecaysRegularizationWhenAlphaIsHealthy)
{
    constexpr int N = 1;
    using Solver = MiniSolver<BugTestModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<BugTestModel, 10>;
    using FakeSolver = StalledDualRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::NONE;
    config.max_iters = 1;
    config.linear_solve_max_attempts = 1;
    config.reg_init = 1e-2;
    config.reg_min = 1e-6;
    config.reg_scale_down = 10.0;
    config.enable_residual_stagnation_detection = false;

    Solver solver(N, Backend::CPU_SERIAL, config);
    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    (void)solver.solve();

    EXPECT_EQ(fake_solver_ptr->calls, 1);
    EXPECT_DOUBLE_EQ(Access::reg(solver), 1e-3)
        << "A successful linear solve with the default healthy alpha should cool reg.";
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

TEST(TerminationTest, AcceptableNmpcAcceptedStepRefreshesPrimalResidual)
{
    constexpr int N = 1;
    using Solver = MiniSolver<AcceptedStepFeasibilityModel, 10>;
    using Access = minisolver::test::SolverInternalAccess<AcceptedStepFeasibilityModel, 10>;
    using FakeSolver = AcceptedStepFeasibilityRiccatiSolver<Solver::TrajArray>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.line_search_type = LineSearchType::NONE;
    config.max_iters = 10;
    config.tol_con = 1e-9;

    Solver solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(0.1), ApiStatus::OK);
    for (int k = 0; k <= N; ++k) {
        ASSERT_EQ(solver.set_state_guess(k, 0, 1.0), ApiStatus::OK);
        ASSERT_EQ(solver.set_slack_guess(k, 0, 0.1), ApiStatus::OK);
        ASSERT_EQ(solver.set_dual_guess(k, 0, 0.1), ApiStatus::OK);
    }
    ASSERT_EQ(solver.set_control_guess(0, 0, 0.0), ApiStatus::OK);

    auto fake_solver = std::make_unique<FakeSolver>();
    FakeSolver* fake_solver_ptr = fake_solver.get();
    Access::set_linear_solver(solver, std::move(fake_solver));

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_EQ(status, SolverStatus::FEASIBLE)
        << "ACCEPTABLE_NMPC should stop only after the accepted trajectory has fresh primal "
           "residuals within tolerance.";
    EXPECT_EQ(info.loop_status, SolverStatus::FEASIBLE);
    EXPECT_EQ(info.termination_reason, TerminationReason::PRIMAL_FEASIBLE);
    EXPECT_EQ(info.iterations, 1);
    EXPECT_EQ(fake_solver_ptr->calls, 1);
    EXPECT_LE(info.primal_inf, config.tol_con);
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
