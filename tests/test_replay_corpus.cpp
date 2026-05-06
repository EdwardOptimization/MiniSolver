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
