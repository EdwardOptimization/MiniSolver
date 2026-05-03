/**
 * @file test_scaling_regressions.cpp
 * @brief Evidence cases for future scaling / normalization work.
 */

#include "minisolver/solver/solver.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

namespace {

template <int Scale> struct EquivalentConstraintScaleModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 2;
    static const int NP = 0;

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
        const T physical_residual = kp.x(0) - static_cast<T>(1.0);
        kp.g_val(0) = physical_residual;
        kp.g_val(1) = static_cast<T>(Scale) * physical_residual;
        kp.C(0, 0) = 1.0;
        kp.C(1, 0) = static_cast<T>(Scale);
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

using WellScaledEquivalentModel = EquivalentConstraintScaleModel<1>;
using BadlyScaledEquivalentModel = EquivalentConstraintScaleModel<1000>;

struct ScalingBaselineReport {
    SolverStatus status = SolverStatus::UNSOLVED;
    SolverInfo info;
};

template <typename Model> ScalingBaselineReport run_initial_feasibility_snapshot(double x0)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0;
    config.enable_profiling = false;

    MiniSolver<Model, 1> solver(0, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", x0);

    ScalingBaselineReport report;
    report.status = solver.solve();
    report.info = solver.get_info();
    return report;
}

} // namespace

TEST(ScalingRegressionTest, EquivalentConstraintRowsExposeUnscaledPrimalMetricDistortion)
{
    const ScalingBaselineReport well_scaled
        = run_initial_feasibility_snapshot<WellScaledEquivalentModel>(2.0);
    const ScalingBaselineReport badly_scaled
        = run_initial_feasibility_snapshot<BadlyScaledEquivalentModel>(2.0);

    ASSERT_EQ(well_scaled.status, SolverStatus::MAX_ITER);
    ASSERT_EQ(badly_scaled.status, SolverStatus::MAX_ITER);
    ASSERT_EQ(well_scaled.info.iterations, 0);
    ASSERT_EQ(badly_scaled.info.iterations, 0);

    ASSERT_TRUE(std::isfinite(well_scaled.info.primal_inf));
    ASSERT_TRUE(std::isfinite(badly_scaled.info.primal_inf));

    EXPECT_NEAR(well_scaled.info.primal_inf, 1.0, 1e-5);
    EXPECT_NEAR(badly_scaled.info.primal_inf, 1000.0, 1e-2);
    EXPECT_GT(badly_scaled.info.primal_inf / well_scaled.info.primal_inf, 900.0)
        << "These two models encode the same physical feasible set. The large ratio captures "
           "the current unscaled-row limitation that N-MOD-2 must address.";
}

TEST(ScalingRegressionTest, BadlyScaledBaselineExposesAvailableSolveMetrics)
{
    const ScalingBaselineReport report
        = run_initial_feasibility_snapshot<BadlyScaledEquivalentModel>(2.0);

    EXPECT_EQ(report.status, SolverStatus::MAX_ITER);
    EXPECT_EQ(report.info.loop_status, SolverStatus::MAX_ITER);
    EXPECT_EQ(report.info.termination_reason, TerminationReason::MAX_ITERATIONS);
    EXPECT_EQ(report.info.iterations, 0);
    EXPECT_DOUBLE_EQ(report.info.alpha, 1.0);
    EXPECT_EQ(report.info.regularization_escalation_count, 0);
    EXPECT_EQ(report.info.soc_attempt_count, 0);
    EXPECT_EQ(report.info.soc_accept_count, 0);
    EXPECT_EQ(report.info.restoration_attempt_count, 0);
    EXPECT_FALSE(report.info.degraded_step);
    EXPECT_TRUE(std::isfinite(report.info.primal_inf));
    EXPECT_TRUE(std::isfinite(report.info.dual_inf));
    EXPECT_TRUE(std::isfinite(report.info.complementarity_inf));
}
