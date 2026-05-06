// Coordinate-scaling hint contract tests (Stage 5 minimal viable).
//
// These tests pin the API behaviour of the per-coordinate scale setters and
// the resulting termination metric. They deliberately do NOT assert that the
// search direction or Riccati recursion changes -- the hint is documented as
// affecting only the dual-stationarity termination metric.

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <gtest/gtest.h>

using namespace minisolver;

namespace {

// Two-state, two-control tracking model with a deliberate scale mismatch
// between the two control coordinates so weighted vs unweighted dual norms
// produce different values.
struct CoordScalingModel {
    static constexpr int NX = 2;
    static constexpr int NU = 2;
    static constexpr int NC = 0;
    static constexpr int NP = 2;

    static constexpr std::array<const char*, NX> state_names = { "x_pos", "x_vel" };
    static constexpr std::array<const char*, NU> control_names = { "u_force", "u_steer" };
    static constexpr std::array<const char*, NP> param_names = { "x_pos_ref", "x_vel_ref" };
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>&,
        double dt, IntegratorType)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + x(1) * dt;
        xn(1) = x(1) + (u(0) + 0.1 * u(1)) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.x(1) * dt;
        kp.f_resid(1) = kp.x(1) + (kp.u(0) + static_cast<T>(0.1) * kp.u(1)) * dt;
        kp.A(0, 0) = 1.0;
        kp.A(0, 1) = dt;
        kp.A(1, 0) = 0.0;
        kp.A(1, 1) = 1.0;
        kp.B(0, 0) = 0.0;
        kp.B(0, 1) = 0.0;
        kp.B(1, 0) = dt;
        kp.B(1, 1) = 0.1 * dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff_pos = kp.x(0) - kp.p(0);
        const T diff_vel = kp.x(1) - kp.p(1);
        kp.cost = static_cast<T>(5.0) * diff_pos * diff_pos
            + static_cast<T>(0.1) * diff_vel * diff_vel + static_cast<T>(0.5) * kp.u(0) * kp.u(0)
            + static_cast<T>(0.5) * kp.u(1) * kp.u(1);
        kp.q(0) = static_cast<T>(10.0) * diff_pos;
        kp.q(1) = static_cast<T>(0.2) * diff_vel;
        kp.r(0) = static_cast<T>(1.0) * kp.u(0);
        kp.r(1) = static_cast<T>(1.0) * kp.u(1);
        kp.Q(0, 0) = 10.0;
        kp.Q(0, 1) = 0.0;
        kp.Q(1, 0) = 0.0;
        kp.Q(1, 1) = 0.2;
        kp.R(0, 0) = 1.0;
        kp.R(0, 1) = 0.0;
        kp.R(1, 0) = 0.0;
        kp.R(1, 1) = 1.0;
        kp.H(0, 0) = 0.0;
        kp.H(0, 1) = 0.0;
        kp.H(1, 0) = 0.0;
        kp.H(1, 1) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

void configure_problem(MiniSolver<CoordScalingModel, 16>& solver, int N)
{
    solver.set_dt(0.1);
    solver.set_initial_state("x_pos", 1.0);
    solver.set_initial_state("x_vel", 0.0);
    for (int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "x_pos_ref", 0.0);
        solver.set_parameter(k, "x_vel_ref", 0.0);
    }
    solver.rollout_dynamics();
}

} // namespace

TEST(CoordinateScalingTest, DefaultConfigKeepsScaleAtUnity)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    MiniSolver<CoordScalingModel, 16> solver(8, Backend::CPU_SERIAL, config);

    EXPECT_EQ(config.coordinate_scaling, CoordinateScalingMethod::NONE);
    EXPECT_DOUBLE_EQ(solver.get_state_scale(0), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_state_scale(1), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_control_scale(0), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_control_scale(1), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_parameter_scale(0), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_parameter_scale(1), 1.0);
}

TEST(CoordinateScalingTest, SetScaleByIndexAndNameAreEquivalent)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    MiniSolver<CoordScalingModel, 16> solver(8, Backend::CPU_SERIAL, config);

    EXPECT_EQ(solver.set_state_scale(0, 100.0), ApiStatus::OK);
    EXPECT_EQ(solver.set_state_scale("x_vel", 0.5), ApiStatus::OK);
    EXPECT_EQ(solver.set_control_scale("u_force", 2.0), ApiStatus::OK);
    EXPECT_EQ(solver.set_control_scale(1, 0.1), ApiStatus::OK);
    EXPECT_EQ(solver.set_parameter_scale("x_pos_ref", 100.0), ApiStatus::OK);

    EXPECT_DOUBLE_EQ(solver.get_state_scale(0), 100.0);
    EXPECT_DOUBLE_EQ(solver.get_state_scale(1), 0.5);
    EXPECT_DOUBLE_EQ(solver.get_control_scale(0), 2.0);
    EXPECT_DOUBLE_EQ(solver.get_control_scale(1), 0.1);
    EXPECT_DOUBLE_EQ(solver.get_parameter_scale(0), 100.0);
}

TEST(CoordinateScalingTest, RejectInvalidScaleValuesAndIndices)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    MiniSolver<CoordScalingModel, 16> solver(8, Backend::CPU_SERIAL, config);

    EXPECT_EQ(solver.set_state_scale(-1, 1.0), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.set_state_scale(7, 1.0), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.set_control_scale(-1, 1.0), ApiStatus::InvalidIndex);
    EXPECT_EQ(solver.set_parameter_scale(99, 1.0), ApiStatus::InvalidIndex);

    EXPECT_EQ(solver.set_state_scale("nonexistent", 1.0), ApiStatus::UnknownName);

    EXPECT_EQ(solver.set_state_scale(0, 0.0), ApiStatus::InvalidArgument);
    EXPECT_EQ(solver.set_state_scale(0, -1.0), ApiStatus::InvalidArgument);
    EXPECT_EQ(solver.set_state_scale(0, std::numeric_limits<double>::infinity()),
        ApiStatus::InvalidArgument);

    EXPECT_EQ(solver.set_control_scale(0, 1e-30), ApiStatus::InvalidArgument);
    EXPECT_EQ(solver.set_control_scale(0, 1e30), ApiStatus::InvalidArgument);

    EXPECT_DOUBLE_EQ(solver.get_state_scale(0), 1.0)
        << "Failed setters must not mutate stored scale";
    EXPECT_DOUBLE_EQ(solver.get_control_scale(0), 1.0);
}

TEST(CoordinateScalingTest, ResetCoordinateScalingRestoresUnity)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    MiniSolver<CoordScalingModel, 16> solver(8, Backend::CPU_SERIAL, config);

    EXPECT_EQ(solver.set_state_scale(0, 100.0), ApiStatus::OK);
    EXPECT_EQ(solver.set_control_scale(1, 0.5), ApiStatus::OK);
    EXPECT_EQ(solver.set_parameter_scale(1, 4.0), ApiStatus::OK);

    solver.reset_coordinate_scaling();

    EXPECT_DOUBLE_EQ(solver.get_state_scale(0), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_control_scale(1), 1.0);
    EXPECT_DOUBLE_EQ(solver.get_parameter_scale(1), 1.0);
}

TEST(CoordinateScalingTest, NoneStrategyKeepsBaselineDualInfBitForBit)
{
    constexpr int N = 8;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.coordinate_scaling = CoordinateScalingMethod::NONE;

    MiniSolver<CoordScalingModel, 16> baseline(N, Backend::CPU_SERIAL, config);
    configure_problem(baseline, N);
    const SolverStatus status_a = baseline.solve();
    const double baseline_dual_inf = baseline.get_info().dual_inf;
    const bool baseline_active = baseline.get_info().coordinate_scaling_active;

    MiniSolver<CoordScalingModel, 16> with_scale(N, Backend::CPU_SERIAL, config);
    configure_problem(with_scale, N);
    EXPECT_EQ(with_scale.set_control_scale(0, 100.0), ApiStatus::OK);
    EXPECT_EQ(with_scale.set_control_scale(1, 0.01), ApiStatus::OK);
    const SolverStatus status_b = with_scale.solve();
    const double dual_inf_with_scale = with_scale.get_info().dual_inf;

    EXPECT_EQ(status_a, status_b)
        << "Setting coordinate scales while CoordinateScalingMethod::NONE is active must not "
           "alter the solve outcome";
    EXPECT_DOUBLE_EQ(dual_inf_with_scale, baseline_dual_inf)
        << "NONE strategy must keep the legacy dual-stationarity inf-norm bit-for-bit";
    EXPECT_FALSE(baseline_active);
    EXPECT_FALSE(with_scale.get_info().coordinate_scaling_active);
}

TEST(CoordinateScalingTest, UserSuppliedStrategyAppliesWeightedDualNorm)
{
    constexpr int N = 8;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.coordinate_scaling = CoordinateScalingMethod::USER_SUPPLIED;
    config.max_iters = 1; // Stop after one outer iteration so r_bar is large enough to compare.
    config.tol_dual = 1e-30;
    config.tol_con = 1e-30;
    config.tol_mu = 1e-30;

    MiniSolver<CoordScalingModel, 16> baseline(N, Backend::CPU_SERIAL, config);
    configure_problem(baseline, N);
    baseline.solve();
    const double baseline_dual_inf = baseline.get_info().dual_inf;
    EXPECT_FALSE(baseline.get_info().coordinate_scaling_active)
        << "All scales at 1.0 must keep coordinate_scaling_active false even with USER_SUPPLIED";

    MiniSolver<CoordScalingModel, 16> shrunk(N, Backend::CPU_SERIAL, config);
    configure_problem(shrunk, N);
    EXPECT_EQ(shrunk.set_control_scale(0, 0.5), ApiStatus::OK);
    EXPECT_EQ(shrunk.set_control_scale(1, 0.5), ApiStatus::OK);
    shrunk.solve();
    const double shrunk_dual_inf = shrunk.get_info().dual_inf;
    EXPECT_TRUE(shrunk.get_info().coordinate_scaling_active);
    EXPECT_LT(shrunk_dual_inf, baseline_dual_inf)
        << "Halving every control scale must halve the weighted dual-stationarity infinity norm";
    EXPECT_NEAR(shrunk_dual_inf, 0.5 * baseline_dual_inf, 1e-9 * std::max(1.0, baseline_dual_inf));

    MiniSolver<CoordScalingModel, 16> grown(N, Backend::CPU_SERIAL, config);
    configure_problem(grown, N);
    EXPECT_EQ(grown.set_control_scale(0, 4.0), ApiStatus::OK);
    EXPECT_EQ(grown.set_control_scale(1, 4.0), ApiStatus::OK);
    grown.solve();
    const double grown_dual_inf = grown.get_info().dual_inf;
    EXPECT_GT(grown_dual_inf, baseline_dual_inf);
    EXPECT_NEAR(grown_dual_inf, 4.0 * baseline_dual_inf, 1e-9 * std::max(1.0, grown_dual_inf));
}

TEST(CoordinateScalingTest, ConfigValidationRejectsInvalidScaleBounds)
{
    SolverConfig config;
    config.coordinate_scale_min = 1e-3;
    config.coordinate_scale_max = 1e3;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::OK);

    config.coordinate_scale_min = 0.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);

    config.coordinate_scale_min = 1e2;
    config.coordinate_scale_max = 1e1;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
}
