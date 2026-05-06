#include "fusedeulerimplicitregressionmodel.h"
#include "minisolver/algorithms/line_search.h"
#include "minisolver/algorithms/linear_solver.h"
#include "minisolver/debug/solver_snapshot.h"
#include "minisolver/solver/solver.h"
#include "test_reference_config.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace minisolver;

namespace {

std::string make_unique_filename(const char* stem, const char* ext)
{
    static std::atomic<unsigned long long> seq { 0 };
    const auto n = ++seq;
    const auto t = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::string name = stem;
    name += "_";
    name += std::to_string(t);
    name += "_";
    name += std::to_string(n);
    name += ext;
    return name;
}

bool file_exists(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    return in.good();
}

bool acceptable_status(SolverStatus status)
{
    return status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE;
}

void expect_finite_info(const SolverInfo& info)
{
    EXPECT_TRUE(valid_solver_status(info.status));
    EXPECT_TRUE(valid_solver_status(info.loop_status));
    EXPECT_TRUE(std::isfinite(info.primal_inf));
    EXPECT_TRUE(std::isfinite(info.unscaled_primal_inf));
    EXPECT_TRUE(std::isfinite(info.dual_inf));
    EXPECT_TRUE(std::isfinite(info.complementarity_inf));
    EXPECT_TRUE(std::isfinite(info.barrier_centrality_inf));
    EXPECT_TRUE(std::isfinite(info.mu));
    EXPECT_TRUE(std::isfinite(info.alpha));
    EXPECT_GE(info.iterations, 0);
    EXPECT_GE(info.regularization_escalation_count, 0);
    EXPECT_GE(info.soc_attempt_count, 0);
    EXPECT_GE(info.soc_accept_count, 0);
    EXPECT_GE(info.soc_reject_count, 0);
    EXPECT_GE(info.restoration_attempt_count, 0);
    EXPECT_GE(info.restoration_success_count, 0);
    EXPECT_GE(info.degraded_riccati_freeze_count, 0);
}

template <typename Solver> double total_unscaled_cost(const Solver& solver)
{
    double total = 0.0;
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        total += solver.get_stage_cost(k);
    }
    return total;
}

template <typename Solver> void shift_solution_guess_one_step(Solver& solver)
{
    const int N = solver.get_horizon();
    std::vector<double> states(static_cast<size_t>(N + 1));
    std::vector<double> controls(static_cast<size_t>(N));

    for (int k = 0; k <= N; ++k) {
        states[static_cast<size_t>(k)] = solver.get_state(k, 0);
    }
    for (int k = 0; k < N; ++k) {
        controls[static_cast<size_t>(k)] = solver.get_control(k, 0);
    }

    for (int k = 0; k < N; ++k) {
        ASSERT_EQ(solver.set_state_guess(k, 0, states[static_cast<size_t>(k + 1)]), ApiStatus::OK);
    }
    ASSERT_EQ(solver.set_state_guess(N, 0, states[static_cast<size_t>(N)]), ApiStatus::OK);

    for (int k = 0; k + 1 < N; ++k) {
        ASSERT_EQ(
            solver.set_control_guess(k, 0, controls[static_cast<size_t>(k + 1)]), ApiStatus::OK);
    }
    if (N > 0) {
        ASSERT_EQ(solver.set_control_guess(N - 1, 0, controls[static_cast<size_t>(N - 1)]),
            ApiStatus::OK);
    }
}

struct CorpusTrackingModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 0;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>&, const MSVec<T, NU>& u, const MSVec<T, NP>&)
    {
        MSVec<T, NX> xdot;
        xdot(0) = u(0);
        return xdot;
    }

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
        kp.cost
            = static_cast<T>(5.0) * kp.x(0) * kp.x(0) + static_cast<T>(0.25) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(10.0) * kp.x(0);
        kp.r(0) = static_cast<T>(0.5) * kp.u(0);
        kp.Q(0, 0) = 10.0;
        kp.R(0, 0) = 0.5;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct CorpusL1SoftModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 1.0 };
    static constexpr std::array<int, NC> constraint_types = { 1 };

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

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0) - static_cast<T>(1.0);
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff = kp.x(0) - static_cast<T>(2.0);
        kp.cost = diff * diff + static_cast<T>(1e-3) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(2.0) * diff;
        kp.r(0) = static_cast<T>(2e-3) * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-3;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct CorpusBadlyScaledModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 2;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0, 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0, 0 };

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
        const T residual = kp.x(0) - static_cast<T>(1.0);
        kp.g_val(0) = residual;
        kp.g_val(1) = static_cast<T>(1000.0) * residual;
        kp.C(0, 0) = 1.0;
        kp.C(1, 0) = 1000.0;
        kp.D.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0);
        kp.q(0) = static_cast<T>(2.0) * kp.x(0);
        kp.r.setZero();
        kp.Q(0, 0) = 2.0;
        kp.R.setIdentity();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct CorpusObstacleSocModel {
    static constexpr int NX = 2;
    static constexpr int NU = 2;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "px", "py" };
    static constexpr std::array<const char*, NU> control_names = { "vx", "vy" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>&,
        double dt, IntegratorType)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        xn(1) = x(1) + u(1) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.f_resid(1) = kp.x(1) + kp.u(1) * dt;
        kp.A.setIdentity();
        kp.B.setZero();
        kp.B(0, 0) = dt;
        kp.B(1, 1) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T radius = static_cast<T>(1.0);
        const T dx = kp.x(0);
        const T dy = kp.x(1);
        kp.g_val(0) = radius * radius - dx * dx - dy * dy;
        kp.C(0, 0) = static_cast<T>(-2.0) * dx;
        kp.C(0, 1) = static_cast<T>(-2.0) * dy;
        kp.D.setZero();
    }

    template <typename T> static void compute_true_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T radius = static_cast<T>(1.0);
        const T dx = kp.x(0);
        const T dy = kp.x(1);
        kp.g_true(0) = radius * radius - dx * dx - dy * dy;
    }

    template <typename T>
    static void compute_soc_constraints(
        const KnotPoint<T, NX, NU, NC, NP>&, KnotPoint<T, NX, NU, NC, NP>& trial_kp)
    {
        compute_constraints(trial_kp);
        compute_true_constraints(trial_kp);
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = static_cast<T>(0.0);
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

class CorpusObstacleSocLinearSolver
    : public LinearSolver<
          Trajectory<KnotPoint<double, CorpusObstacleSocModel::NX, CorpusObstacleSocModel::NU,
                         CorpusObstacleSocModel::NC, CorpusObstacleSocModel::NP>,
              1>::TrajArray> {
public:
    using TrajArray
        = Trajectory<KnotPoint<double, CorpusObstacleSocModel::NX, CorpusObstacleSocModel::NU,
                         CorpusObstacleSocModel::NC, CorpusObstacleSocModel::NP>,
            1>::TrajArray;

    bool soc_called = false;
    double observed_trial_obstacle_residual = 0.0;

    LinearSolveResult solve(TrajArray&, int, double, double, InertiaStrategy, const SolverConfig&,
        const TrajArray*) override
    {
        return true;
    }

    LinearSolveResult solve_soc(TrajArray& traj, const TrajArray& soc_rhs_traj, int N, double,
        double, InertiaStrategy, const SolverConfig&) override
    {
        soc_called = true;
        observed_trial_obstacle_residual = soc_rhs_traj[0].g_val(0);
        for (int k = 0; k <= N; ++k) {
            traj[k].dx.setZero();
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
            traj[k].dsoft_s.setZero();
        }
        traj[0].dx(0) = 1.0;
        traj[0].ds(0) = -1.0;
        return true;
    }
};

} // namespace

TEST(ReplayCorpusTest, UnconstrainedTrackingConvergesAndReplaysPreSolveSnapshot)
{
    constexpr int N = 12;
    constexpr double dt = 0.1;
    constexpr double x0 = 3.0;

    SolverConfig config = minisolver::test::make_reference_solver_config();
    config.max_iters = 120;
    config.tol_con = 1e-7;
    config.tol_dual = 1e-7;
    config.mu_final = 1e-9;

    MiniSolver<CorpusTrackingModel, 20> solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(dt), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", x0), ApiStatus::OK);
    solver.rollout_dynamics();

    using SnapshotIO = SolverSnapshotIO<CorpusTrackingModel, 20>;
    const auto pre_solve = SnapshotIO::capture_snapshot(solver);

    const std::string filename = make_unique_filename("replay_corpus_tracking", ".msnap");
    ASSERT_EQ(SnapshotIO::save_snapshot(filename, pre_solve).status, SnapshotStatus::OK);
    MiniSolver<CorpusTrackingModel, 20> replay(N, Backend::CPU_SERIAL);
    ASSERT_EQ(SnapshotIO::load_case(filename, replay).status, SnapshotStatus::OK);
    EXPECT_EQ(replay.get_horizon(), N);
    EXPECT_DOUBLE_EQ(replay.get_state(0, 0), x0);
    std::remove(filename.c_str());

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();
    ASSERT_TRUE(acceptable_status(status));
    EXPECT_EQ(info.status, status);
    expect_finite_info(info);

    const double total_cost = total_unscaled_cost(solver);
    EXPECT_TRUE(std::isfinite(total_cost));
    EXPECT_LT(std::abs(solver.get_state(N, 0)), std::abs(x0));

    for (int k = 0; k < N; ++k) {
        const double expected = solver.get_state(k, 0) + solver.get_control(k, 0) * dt;
        EXPECT_NEAR(solver.get_state(k + 1, 0), expected, 1e-6);
    }
}

TEST(ReplayCorpusTest, WarmStartTwoFrameSolveReachesAcceptableQualityInTwoIterations)
{
    constexpr int N = 12;
    constexpr double dt = 0.1;
    constexpr double x0 = 3.0;

    SolverConfig cold_config = minisolver::test::make_reference_solver_config();
    cold_config.max_iters = 120;
    cold_config.tol_con = 1e-7;
    cold_config.tol_dual = 1e-7;
    cold_config.mu_final = 1e-9;

    MiniSolver<CorpusTrackingModel, 20> solver(N, Backend::CPU_SERIAL, cold_config);
    ASSERT_EQ(solver.set_dt(dt), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", x0), ApiStatus::OK);
    solver.rollout_dynamics();

    const SolverStatus first_status = solver.solve();
    ASSERT_TRUE(acceptable_status(first_status));
    expect_finite_info(solver.get_info());

    shift_solution_guess_one_step(solver);
    const double shifted_x0 = solver.get_state(0, 0);
    const double shifted_cost_before = total_unscaled_cost(solver);

    SolverConfig warm_config = solver.get_config();
    warm_config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    warm_config.warm_start_barrier = WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP;
    warm_config.warm_start_regularization = WarmStartRegularizationMode::RESET_TO_REG_INIT;
    warm_config.barrier_strategy = BarrierStrategy::ADAPTIVE;
    warm_config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    warm_config.max_iters = 2;
    warm_config.tol_con = 1e-5;
    warm_config.tol_dual = 1e-4;
    warm_config.tol_mu = 1e-6;
    warm_config.mu_final = 1e-7;
    ASSERT_EQ(solver.set_config(warm_config), ApiStatus::OK);

    using SnapshotIO = SolverSnapshotIO<CorpusTrackingModel, 20>;
    const auto second_frame_snapshot = SnapshotIO::capture_snapshot(solver);

    const SolverStatus second_status = solver.solve();
    const SolverInfo& second_info = solver.get_info();
    ASSERT_TRUE(acceptable_status(second_status));
    expect_finite_info(second_info);
    EXPECT_LE(second_info.iterations, 2);
    EXPECT_LT(std::abs(solver.get_state(N, 0)), std::abs(shifted_x0));
    EXPECT_LE(total_unscaled_cost(solver), shifted_cost_before + 1e-9);

    const std::string filename = make_unique_filename("replay_corpus_warmstart", ".msnap");
    ASSERT_EQ(
        SnapshotIO::save_snapshot(filename, second_frame_snapshot).status, SnapshotStatus::OK);
    MiniSolver<CorpusTrackingModel, 20> replay(N, Backend::CPU_SERIAL);
    ASSERT_EQ(SnapshotIO::load_case(filename, replay).status, SnapshotStatus::OK);
    EXPECT_EQ(replay.get_config().initialization, InitializationMode::REUSE_PRIMAL_DUAL);
    EXPECT_EQ(
        replay.get_config().warm_start_barrier, WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP);
    EXPECT_DOUBLE_EQ(replay.get_state(0, 0), shifted_x0);
    std::remove(filename.c_str());
}

TEST(ReplayCorpusTest, L1SoftConstraintConvergesWithFiniteInteriorMetrics)
{
    constexpr int N = 3;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.max_iters = 120;
    config.tol_con = 1e-5;
    config.tol_dual = 1e-5;
    config.mu_final = 1e-7;

    MiniSolver<CorpusL1SoftModel, 10> solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(1.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    for (int k = 0; k < N; ++k) {
        ASSERT_EQ(solver.set_control_guess(k, "u", 0.5), ApiStatus::OK);
    }
    solver.rollout_dynamics();

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();
    ASSERT_TRUE(acceptable_status(status));
    expect_finite_info(info);
    EXPECT_TRUE(std::isfinite(total_unscaled_cost(solver)));

    for (int k = 0; k <= N; ++k) {
        const double slack = solver.get_slack(k, 0);
        const double dual = solver.get_dual(k, 0);
        EXPECT_TRUE(std::isfinite(slack));
        EXPECT_TRUE(std::isfinite(dual));
        EXPECT_GT(slack, 0.0);
        EXPECT_GT(dual, 0.0);
        EXPECT_LT(dual, CorpusL1SoftModel::constraint_weights[0]);
    }
}

TEST(ReplayCorpusTest, BadScalingCaseReportsScaledAndUnscaledFeasibility)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.max_iters = 0;
    config.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;

    MiniSolver<CorpusBadlyScaledModel, 1> solver(0, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_initial_state("x", 2.0), ApiStatus::OK);

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();
    EXPECT_EQ(status, SolverStatus::MAX_ITER);
    EXPECT_EQ(info.termination_reason, TerminationReason::MAX_ITERATIONS);
    expect_finite_info(info);
    EXPECT_TRUE(info.problem_scaling_active);
    EXPECT_TRUE(info.constraint_scaling_active);
    EXPECT_TRUE(info.objective_scaling_active);
    EXPECT_NEAR(info.primal_inf, 1.0, 1e-5);
    EXPECT_NEAR(info.unscaled_primal_inf, 1000.0, 1e-2);
}

TEST(ReplayCorpusTest, FailureSnapshotWorkflowPersistsPreSolveReplayState)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.max_iters = 0;
    config.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;

    MiniSolver<CorpusBadlyScaledModel, 1> solver(0, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_initial_state("x", 2.0), ApiStatus::OK);

    using SnapshotIO = SolverSnapshotIO<CorpusBadlyScaledModel, 1>;
    const auto pre_solve = SnapshotIO::capture_snapshot(solver);
    const SolverStatus status = solver.solve();
    ASSERT_EQ(status, SolverStatus::MAX_ITER);

    const std::string filename = make_unique_filename("replay_corpus_failure", ".msnap");
    ASSERT_EQ(
        SnapshotIO::save_failure_snapshot(filename, pre_solve, status).status, SnapshotStatus::OK);
    ASSERT_TRUE(file_exists(filename));

    MiniSolver<CorpusBadlyScaledModel, 1> replay(0, Backend::CPU_SERIAL);
    ASSERT_EQ(SnapshotIO::load_case(filename, replay).status, SnapshotStatus::OK);
    EXPECT_EQ(replay.get_horizon(), 0);
    EXPECT_DOUBLE_EQ(replay.get_state(0, 0), 2.0);
    EXPECT_EQ(replay.get_config().problem_scaling, ProblemScalingMethod::RUIZ_EQUILIBRATION);

    std::remove(filename.c_str());
}

TEST(ReplayCorpusTest, GeneratedImplicitIntegratorConvergesAndReplaysPreSolveSnapshot)
{
    constexpr int N = 10;
    constexpr int MAX_N = 16;
    using Model = FusedEulerImplicitRegressionModel;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.integrator = IntegratorType::EULER_IMPLICIT;
    config.default_dt = 0.08;
    config.max_iters = 80;
    config.tol_con = 1e-7;
    config.tol_dual = 1e-7;
    config.tol_mu = 1e-8;
    config.mu_final = 1e-8;
    config.tol_cost = 1e-10;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::FILTER;

    MiniSolver<Model, MAX_N> solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(config.default_dt), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x0", 1.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x1", -0.45), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x2", 0.30), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x3", -0.20), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x4", 0.12), ApiStatus::OK);
    solver.rollout_dynamics();

    using SnapshotIO = SolverSnapshotIO<Model, MAX_N>;
    const auto pre_solve = SnapshotIO::capture_snapshot(solver);

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();
    ASSERT_TRUE(acceptable_status(status));
    expect_finite_info(info);
    EXPECT_EQ(solver.get_config().integrator, IntegratorType::EULER_IMPLICIT);
    EXPECT_TRUE(std::isfinite(total_unscaled_cost(solver)));

    const std::string filename = make_unique_filename("replay_corpus_implicit", ".msnap");
    ASSERT_EQ(SnapshotIO::save_snapshot(filename, pre_solve).status, SnapshotStatus::OK);
    MiniSolver<Model, MAX_N> replay(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(SnapshotIO::load_case(filename, replay).status, SnapshotStatus::OK);
    EXPECT_EQ(replay.get_config().integrator, IntegratorType::EULER_IMPLICIT);
    EXPECT_DOUBLE_EQ(replay.get_state(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(replay.get_state(0, 1), -0.45);
    std::remove(filename.c_str());
}

TEST(ReplayCorpusTest, SocNonlinearObstaclePathAttemptsAndAcceptsCorrection)
{
    constexpr int N = 0;
    constexpr int MAX_N = 1;
    using Model = CorpusObstacleSocModel;
    using Knot = KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>;
    using TrajectoryType = Trajectory<Knot, MAX_N>;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = true;
    config.enable_line_search_rollout = false;
    config.soc_trigger_alpha = 0.1;
    config.line_search_tau = 0.9;

    FilterLineSearch<Model, MAX_N> line_search;
    CorpusObstacleSocLinearSolver linear_solver;
    TrajectoryType trajectory(N);
    std::array<double, MAX_N> dt_traj {};
    dt_traj.fill(0.1);

    auto& active = trajectory.active();
    active[0].set_zero();
    active[0].x(0) = 1.5;
    active[0].x(1) = 0.0;
    active[0].s(0) = 2.25;
    active[0].lam(0) = 1.0;
    active[0].dx(0) = -1.0;
    active[0].dx(1) = 0.0;
    active[0].du.setZero();
    active[0].ds(0) = 0.0;
    active[0].dlam(0) = 0.0;
    active[0].dsoft_s.setZero();
    Model::compute_dynamics(active[0], config.integrator, 0.0);
    Model::compute_constraints(active[0]);
    Model::compute_true_constraints(active[0]);
    Model::compute_cost_gn(active[0]);

    const LineSearchResult result
        = line_search.search(trajectory, linear_solver, dt_traj, 0.1, 1e-6, config);

    EXPECT_TRUE(result.soc_attempted);
    EXPECT_TRUE(result.soc_accepted);
    EXPECT_FALSE(result.soc_rejected);
    ASSERT_TRUE(linear_solver.soc_called);
    EXPECT_NEAR(linear_solver.observed_trial_obstacle_residual, 0.75, 1e-12)
        << "SOC must use the nonlinear obstacle residual evaluated at the rejected trial point.";
    EXPECT_NEAR(trajectory.active()[0].x(0), 1.5, 1e-12);
    EXPECT_NEAR(trajectory.active()[0].s(0), 1.25, 1e-12);
    EXPECT_NEAR(trajectory.active()[0].g_true(0) + trajectory.active()[0].s(0), 0.0, 1e-12);
}
