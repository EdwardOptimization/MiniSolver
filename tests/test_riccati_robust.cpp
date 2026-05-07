// Riccati robustness mode contract tests.
//
// These tests pin the diagnostic-only contract introduced for Tier 4.2:
//   - SolverInfo::riccati_indefinite_blocks counts the number of backward-pass
//     stages that triggered an inertia-correction fallback (general-path
//     regularization escalation, small-Nu freeze, or
//     SATURATION/IGNORE_SINGULAR repair).
//   - SolverInfo::riccati_max_diagonal_perturbation reports the largest extra
//     diagonal value added beyond the user-supplied `reg`.
//   - In RiccatiRobustMode::STANDARD only the small-Nu freeze fallback
//     escalates to SolverInfo::degraded_step (via the pre-existing N-DEG-1
//     LinearSolveResult::degraded_step path); the other three fallback
//     paths leave degraded_step untouched even when their counters are
//     non-zero.
//   - RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS additionally flips
//     degraded_step whenever riccati_indefinite_blocks > 0, i.e. for *any*
//     of the four fallback paths, not only the freeze case.
//   - Validation accepts both modes and rejects unknown enum values.
//
// Behavioural coverage:
//   - tests/test_riccati.cpp pins the small-Nu freeze fallback at the
//     LinearSolveResult level (NU=2, REGULARIZATION strategy, indefinite
//     R(1,1)).
//   - tests/test_status.cpp pins the freeze-fallback end-to-end surface
//     through SolverInfo::degraded_step / degraded_riccati_freeze_count.
//   - This file pins the per-mode escalation contract:
//       * clean-problem invariants (counters stay zero, both modes leave
//         degraded_step false);
//       * config-validation gate;
//       * non-freeze general-path REGULARIZATION retry at the
//         RiccatiSolver level on a NU=4 model with mildly indefinite
//         R(0,0): counter bumps, degraded_step stays false at the
//         per-call (LinearSolveResult) level;
//       * Solver-level mode escalation: STANDARD does not promote
//         non-freeze events to SolverInfo::degraded_step,
//         INERTIA_AWARE_DIAGNOSTICS does, and the freeze path escalates
//         in BOTH modes.

#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

using namespace minisolver;

namespace minisolver {
namespace test {
    template <typename Model, int MAX_N> struct SolverInternalAccess {
        using Solver = MiniSolver<Model, MAX_N>;
        static void record_linear_solver_diagnostics(Solver& s, const LinearSolveResult& result)
        {
            s.record_linear_solver_diagnostics_(result);
        }
    };
} // namespace test
} // namespace minisolver

namespace {

struct SmoothTrackingModel {
    static constexpr int NX = 2;
    static constexpr int NU = 1;
    static constexpr int NC = 0;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x", "v" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>&,
        double dt, IntegratorType)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + x(1) * dt;
        xn(1) = x(1) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.x(1) * dt;
        kp.f_resid(1) = kp.x(1) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.A(0, 1) = dt;
        kp.A(1, 0) = 0.0;
        kp.A(1, 1) = 1.0;
        kp.B(0, 0) = 0.0;
        kp.B(1, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = static_cast<T>(2.0) * kp.x(0) * kp.x(0) + static_cast<T>(0.1) * kp.x(1) * kp.x(1)
            + static_cast<T>(0.5) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(4.0) * kp.x(0);
        kp.q(1) = static_cast<T>(0.2) * kp.x(1);
        kp.r(0) = static_cast<T>(1.0) * kp.u(0);
        kp.Q(0, 0) = 4.0;
        kp.Q(0, 1) = 0.0;
        kp.Q(1, 0) = 0.0;
        kp.Q(1, 1) = 0.2;
        kp.R(0, 0) = 1.0;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

bool acceptable(SolverStatus s)
{
    return s == SolverStatus::OPTIMAL || s == SolverStatus::FEASIBLE;
}

} // namespace

TEST(RiccatiRobustModeTest, DefaultConfigUsesStandardMode)
{
    SolverConfig config;
    EXPECT_EQ(config.riccati_robust_mode, RiccatiRobustMode::STANDARD);
}

TEST(RiccatiRobustModeTest, ConfigValidationAcceptsBothModesAndRejectsUnknown)
{
    SolverConfig config;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::OK);

    config.riccati_robust_mode = RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::OK);

    // Force an out-of-range enum value to confirm the enum gate fires.
    auto& mode = config.riccati_robust_mode;
    std::int32_t raw = 0;
    std::memcpy(&raw, &mode, sizeof(raw));
    raw = 7;
    std::memcpy(&mode, &raw, sizeof(mode));
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
}

TEST(RiccatiRobustModeTest, StandardModeKeepsInertiaCountersAtZeroOnSmoothProblem)
{
    constexpr int N = 10;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.riccati_robust_mode = RiccatiRobustMode::STANDARD;

    MiniSolver<SmoothTrackingModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    EXPECT_EQ(solver.set_initial_state("x", 0.5), ApiStatus::OK);
    EXPECT_EQ(solver.set_initial_state("v", 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    const SolverStatus status = solver.solve();
    EXPECT_TRUE(acceptable(status)) << status_to_string(status);

    const SolverInfo& info = solver.get_info();
    EXPECT_EQ(info.riccati_indefinite_blocks, 0);
    EXPECT_DOUBLE_EQ(info.riccati_max_diagonal_perturbation, 0.0);
    EXPECT_FALSE(info.degraded_step)
        << "STANDARD mode must not flag a clean solve with zero inertia events as degraded";
}

TEST(RiccatiRobustModeTest, InertiaAwareModeAlsoZeroOnCleanProblemAndKeepsDegradedFalse)
{
    constexpr int N = 10;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.riccati_robust_mode = RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS;

    MiniSolver<SmoothTrackingModel, 16> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    EXPECT_EQ(solver.set_initial_state("x", 0.5), ApiStatus::OK);
    EXPECT_EQ(solver.set_initial_state("v", 0.0), ApiStatus::OK);
    solver.rollout_dynamics();

    const SolverStatus status = solver.solve();
    EXPECT_TRUE(acceptable(status)) << status_to_string(status);

    const SolverInfo& info = solver.get_info();
    // Contract: counters always populated (here zero because the problem is
    // SPD); INERTIA_AWARE only escalates *non-zero* corrections to
    // degraded_step.
    EXPECT_EQ(info.riccati_indefinite_blocks, 0);
    EXPECT_DOUBLE_EQ(info.riccati_max_diagonal_perturbation, 0.0);
    EXPECT_FALSE(info.degraded_step)
        << "INERTIA_AWARE mode must not flip degraded_step when no inertia correction occurred";
}

TEST(RiccatiRobustModeTest, InfoResetClearsInertiaCounters)
{
    SolverInfo info;
    info.riccati_indefinite_blocks = 7;
    info.riccati_max_diagonal_perturbation = 1e3;
    info.degraded_step = true;
    info.reset();
    EXPECT_EQ(info.riccati_indefinite_blocks, 0);
    EXPECT_DOUBLE_EQ(info.riccati_max_diagonal_perturbation, 0.0);
    EXPECT_FALSE(info.degraded_step);
}

namespace {
// Direct LinearSolveResult-level RED test for the *non-freeze* fallback
// path. We construct a 4-control / 2-state Riccati problem with a mildly
// indefinite R(0,0) so:
//   * Knot::NU == 4 forces the general path (line 600 onwards in riccati.h)
//     instead of the small-Nu freeze fallback (which only fires for
//     NU <= 3 + REGULARIZATION).
//   * R(0,0) = -1e-6 with regularization_step >> 1e-6 means the SPD retry
//     succeeds after adding regularization_step to every diagonal, so
//     LinearSolveResult::ok stays true and the result is *not* a freeze
//     (degraded_step stays false at the per-call level).
// The contract under test:
//   * riccati_indefinite_blocks > 0 (one stage saw a non-freeze fallback
//     fire).
//   * riccati_max_diagonal_perturbation matches the diagonal addend used
//     by the general-path REGULARIZATION retry.
//   * LinearSolveResult::degraded_step stays FALSE (the Solver-level
//     escalation that depends on RiccatiRobustMode is layered on top of
//     this in record_linear_solver_diagnostics_; here we pin the
//     per-call invariant).
struct NonFreezeFallbackModel {
    static constexpr int NX = 2;
    static constexpr int NU = 4;
    static constexpr int NC = 0;
    static constexpr int NP = 0;
    static constexpr std::array<const char*, NX> state_names = { "x", "v" };
    static constexpr std::array<const char*, NU> control_names = { "u0", "u1", "u2", "u3" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};
};
} // namespace

TEST(RiccatiRobustModeTest, GeneralPathRegularizationFallbackBumpsCounterButLeavesDegradedFalse)
{
    using Knot = KnotPoint<double, NonFreezeFallbackModel::NX, NonFreezeFallbackModel::NU,
        NonFreezeFallbackModel::NC, NonFreezeFallbackModel::NP>;
    using TrajArray = std::array<Knot, 3>; // N = 2

    TrajArray traj;
    constexpr int N = 2;

    SolverConfig config;
    config.reg_min = 1.0e-9;

    // Make Quu(0,0) mildly indefinite by isolating the u0 column. With
    // B[:,0] = 0 the (0,0) entry of Quu = R(0,0) + (B[:,0])^T * V_xx * B[:,0]
    // collapses to R(0,0) alone (any cost-to-go contribution for the
    // remaining controls cannot rescue u0). Pick R(0,0) just below the
    // amount the general-path REGULARIZATION retry will inject so the
    // SECOND Cholesky succeeds and result.ok stays true.
    const double indef_r00 = -0.5 * config.regularization_step;

    for (int k = 0; k <= N; ++k) {
        traj[k].set_zero();
        traj[k].Q.setIdentity();
        traj[k].R.setIdentity();
        traj[k].R(0, 0) = indef_r00;
        traj[k].A.setIdentity();
        traj[k].B.setZero();
        traj[k].B(1, 1) = 1.0; // only u1 affects v.
        traj[k].x.setZero();
        traj[k].u.setZero();
    }

    traj[N].q(0) = 1.0; // Drive a non-trivial backward gradient.

    RiccatiSolver<TrajArray, NonFreezeFallbackModel> solver;
    const LinearSolveResult result
        = solver.solve(traj, N, 0.01, 1.0e-9, InertiaStrategy::REGULARIZATION, config);

    EXPECT_TRUE(result.ok)
        << "General-path REGULARIZATION retry should recover from a mildly indefinite Quu.";
    EXPECT_FALSE(result.degraded_step)
        << "The non-freeze general-path retry must NOT set LinearSolveResult::degraded_step "
           "(only the small-Nu freeze fallback owns the per-call degraded_step bit).";
    EXPECT_EQ(result.degraded_riccati_freeze_count, 0)
        << "The general-path retry must not be confused with a freeze fallback.";
    EXPECT_GT(result.riccati_indefinite_blocks, 0)
        << "The general-path REGULARIZATION retry must bump riccati_indefinite_blocks.";
    EXPECT_NEAR(result.riccati_max_diagonal_perturbation, config.regularization_step, 1.0e-12)
        << "The recorded perturbation must match the diagonal addend the retry actually used.";
}

// Solver-level per-mode escalation contract. Inject a hand-crafted
// LinearSolveResult that carries a non-freeze inertia event
// (riccati_indefinite_blocks = 2, degraded_step = false,
// degraded_riccati_freeze_count = 0) into record_linear_solver_diagnostics_
// and check what propagates to SolverInfo:
//   * STANDARD: counters propagate, degraded_step stays false.
//   * INERTIA_AWARE_DIAGNOSTICS: counters propagate AND degraded_step flips
//     to true (mode-driven escalation).
TEST(RiccatiRobustModeTest, StandardModeDoesNotEscalateNonFreezeInertiaEvents)
{
    using Access = minisolver::test::SolverInternalAccess<SmoothTrackingModel, 16>;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.riccati_robust_mode = RiccatiRobustMode::STANDARD;

    MiniSolver<SmoothTrackingModel, 16> solver(/*initial_N=*/4, Backend::CPU_SERIAL, config);

    LinearSolveResult fake_result(true, /*degraded=*/false, /*freeze_count=*/0);
    fake_result.riccati_indefinite_blocks = 2;
    fake_result.riccati_max_diagonal_perturbation = 1.5e-3;
    Access::record_linear_solver_diagnostics(solver, fake_result);

    const SolverInfo& info = solver.get_info();
    EXPECT_EQ(info.riccati_indefinite_blocks, 2)
        << "Non-freeze counters must propagate to SolverInfo regardless of mode.";
    EXPECT_DOUBLE_EQ(info.riccati_max_diagonal_perturbation, 1.5e-3)
        << "The Solver-level max-perturbation must mirror the per-call value.";
    EXPECT_FALSE(info.degraded_step)
        << "STANDARD mode must NOT escalate non-freeze inertia events to degraded_step.";
    EXPECT_EQ(info.degraded_riccati_freeze_count, 0)
        << "Non-freeze events must not increment the freeze-specific counter.";
}

TEST(RiccatiRobustModeTest, InertiaAwareModeEscalatesNonFreezeInertiaEvents)
{
    using Access = minisolver::test::SolverInternalAccess<SmoothTrackingModel, 16>;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.riccati_robust_mode = RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS;

    MiniSolver<SmoothTrackingModel, 16> solver(/*initial_N=*/4, Backend::CPU_SERIAL, config);

    LinearSolveResult fake_result(true, /*degraded=*/false, /*freeze_count=*/0);
    fake_result.riccati_indefinite_blocks = 1;
    fake_result.riccati_max_diagonal_perturbation = 4.2e-4;
    Access::record_linear_solver_diagnostics(solver, fake_result);

    const SolverInfo& info = solver.get_info();
    EXPECT_EQ(info.riccati_indefinite_blocks, 1)
        << "Non-freeze counters must propagate to SolverInfo regardless of mode.";
    EXPECT_DOUBLE_EQ(info.riccati_max_diagonal_perturbation, 4.2e-4);
    EXPECT_TRUE(info.degraded_step)
        << "INERTIA_AWARE_DIAGNOSTICS must flip degraded_step whenever "
           "riccati_indefinite_blocks > 0, even on a non-freeze fallback.";
    EXPECT_EQ(info.degraded_riccati_freeze_count, 0)
        << "Mode escalation must not synthesize a freeze count for a non-freeze event.";
}

TEST(RiccatiRobustModeTest, FreezeFallbackEscalatesInBothModes)
{
    using Access = minisolver::test::SolverInternalAccess<SmoothTrackingModel, 16>;

    for (auto mode :
        { RiccatiRobustMode::STANDARD, RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS }) {
        SolverConfig config;
        config.print_level = PrintLevel::NONE;
        config.riccati_robust_mode = mode;

        MiniSolver<SmoothTrackingModel, 16> solver(/*initial_N=*/4, Backend::CPU_SERIAL, config);

        LinearSolveResult fake_result(true, /*degraded=*/true, /*freeze_count=*/3);
        fake_result.riccati_indefinite_blocks = 3;
        fake_result.riccati_max_diagonal_perturbation = 1.0e2;
        Access::record_linear_solver_diagnostics(solver, fake_result);

        const SolverInfo& info = solver.get_info();
        EXPECT_TRUE(info.degraded_step)
            << "Freeze fallback (LinearSolveResult::degraded_step=true) must flip "
               "SolverInfo::degraded_step in BOTH modes (pre-existing N-DEG-1 contract).";
        EXPECT_EQ(info.degraded_riccati_freeze_count, 3)
            << "Freeze count must propagate verbatim regardless of mode.";
        EXPECT_EQ(info.riccati_indefinite_blocks, 3);
        EXPECT_DOUBLE_EQ(info.riccati_max_diagonal_perturbation, 1.0e2);
    }
}
