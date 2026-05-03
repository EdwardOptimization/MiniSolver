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

struct LargeObjectiveCurvatureModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

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
        (void)kp;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost
            = static_cast<T>(50.0) * kp.x(0) * kp.x(0) + static_cast<T>(12.5) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(100.0) * kp.x(0);
        kp.r(0) = static_cast<T>(25.0) * kp.u(0);
        kp.Q(0, 0) = 100.0;
        kp.R(0, 0) = 25.0;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

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

template <typename Model>
ScalingBaselineReport run_initial_feasibility_snapshot(double x0, const SolverConfig& config)
{
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

TEST(ScalingRegressionTest, AutomaticRowScalingNormalizesInternalPrimalMetric)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0;
    config.constraint_scaling = ConstraintScalingMethod::ROW_INF_NORM;

    const ScalingBaselineReport report
        = run_initial_feasibility_snapshot<BadlyScaledEquivalentModel>(2.0, config);

    EXPECT_EQ(report.status, SolverStatus::MAX_ITER);
    EXPECT_TRUE(report.info.constraint_scaling_active);
    EXPECT_NEAR(report.info.primal_inf, 1.0, 1e-5)
        << "Automatic row scaling should normalize equivalent constraint rows without "
           "manual model metadata.";
    EXPECT_NEAR(report.info.unscaled_primal_inf, 1000.0, 1e-2)
        << "Diagnostics should still expose the raw model residual.";
}

TEST(ScalingRegressionTest, AutomaticScalingRejectsInvalidScaleBounds)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.constraint_scaling = ConstraintScalingMethod::ROW_INF_NORM;
    config.constraint_row_scale_min = 0.0;

    MiniSolver<BadlyScaledEquivalentModel, 1> solver(0, Backend::CPU_SERIAL, config);

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_EQ(status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(info.status, SolverStatus::INVALID_INPUT);
    EXPECT_EQ(info.termination_reason, TerminationReason::INVALID_INPUT);
    EXPECT_FALSE(info.constraint_scaling_active);
}

TEST(ScalingRegressionTest, HessianGershgorinScalesObjectivePacketOnly)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.objective_scaling = ObjectiveScalingMethod::HESSIAN_GERSHGORIN;

    KnotPoint<double, LargeObjectiveCurvatureModel::NX, LargeObjectiveCurvatureModel::NU,
        LargeObjectiveCurvatureModel::NC, LargeObjectiveCurvatureModel::NP>
        kp;
    kp.x(0) = 2.0;
    kp.u(0) = 3.0;

    detail::evaluate_model_stage<LargeObjectiveCurvatureModel>(kp, config, 0.1, false);

    const double unscaled_cost = 50.0 * 2.0 * 2.0 + 12.5 * 3.0 * 3.0;
    EXPECT_NEAR(kp.objective_scale, 0.01, 1e-12);
    EXPECT_NEAR(kp.cost_unscaled, unscaled_cost, 1e-12);
    EXPECT_NEAR(kp.cost, unscaled_cost * 0.01, 1e-12);
    EXPECT_NEAR(kp.q(0), 2.0, 1e-12);
    EXPECT_NEAR(kp.r(0), 0.75, 1e-12);
    EXPECT_NEAR(kp.Q(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(kp.R(0, 0), 0.25, 1e-12);
}

TEST(ScalingRegressionTest, ProblemScalingActivatesBoundedConstraintAndObjectiveScaling)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0;
    config.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;

    const ScalingBaselineReport report
        = run_initial_feasibility_snapshot<BadlyScaledEquivalentModel>(2.0, config);

    EXPECT_EQ(report.status, SolverStatus::MAX_ITER);
    EXPECT_TRUE(report.info.problem_scaling_active);
    EXPECT_TRUE(report.info.constraint_scaling_active);
    EXPECT_TRUE(report.info.objective_scaling_active);
    EXPECT_NEAR(report.info.primal_inf, 1.0, 1e-5);
    EXPECT_NEAR(report.info.unscaled_primal_inf, 1000.0, 1e-2);
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
