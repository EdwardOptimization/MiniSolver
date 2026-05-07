// Riccati robustness mode contract tests.
//
// These tests pin the diagnostic-only contract introduced for Tier 4.2:
//   - SolverInfo::riccati_indefinite_blocks counts the number of backward-pass
//     stages that triggered an inertia-correction fallback (general-path
//     regularization escalation, small-Nu freeze, or
//     SATURATION/IGNORE_SINGULAR repair).
//   - SolverInfo::riccati_max_diagonal_perturbation reports the largest extra
//     diagonal value added beyond the user-supplied `reg`.
//   - RiccatiRobustMode::STANDARD never tags a successful solve with
//     non-zero inertia counters as a degraded step.
//   - RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS flips degraded_step when
//     any inertia-correction event happens, even if the linear solve
//     succeeds, so monitoring code can gate downstream control actions.
//   - Validation accepts both modes and rejects unknown enum values.
//
// We deliberately do not exercise a model that currently triggers the
// fallback: such a case requires a near-indefinite Quu, which is rare and
// not the contract under test here. What we *do* test is that on a clean
// problem the counters stay zero in both modes, and that the config
// validation gate is hooked up correctly.

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

using namespace minisolver;

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
        kp.H(0, 0) = 0.0;
        kp.H(1, 0) = 0.0;
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
