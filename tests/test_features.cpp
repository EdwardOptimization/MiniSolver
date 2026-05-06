/**
 * @file test_features.cpp
 * @brief Tests for individual solver features: cost stagnation, parameter
 *        persistence, explicit GPU unsupported behavior, and direction refinement.
 */
#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

// =============================================================================
// Model: Flat Cost for stagnation termination test
// =============================================================================
struct FlatCostModel {
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
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& /*u*/,
        const MSVec<T, NP>& /*p*/, double /*dt*/, IntegratorType /*type*/)
    {
        return x;
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double /*dt*/)
    {
        kp.f_resid(0) = kp.x(0);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = 0.0;
        double w = 1e-7;
        kp.cost = w * kp.x(0) * kp.x(0);
        kp.Q(0, 0) = 2 * w;
        kp.q(0) = 2 * w * kp.x(0);
        kp.R.setIdentity();
        kp.g_val(0) = -10.0 - kp.x(0);
        kp.C(0, 0) = -1.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute(kp, type, dt);
    }
    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute(kp, IntegratorType::EULER_EXPLICIT, 0.1);
    }
};

// =============================================================================
// Feature: Cost Stagnation Termination
// =============================================================================
TEST(FeaturesTest, CostStagnationTermination)
{
    int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.tol_con = 1e-12;
    config.tol_mu = 1e-12;
    config.mu_final = 1e-9;
    config.tol_cost = 1e-5;
    config.max_iters = 50;

    MiniSolver<FlatCostModel, 10> solver(N, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 1.0);

    SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE)
        << "Cost stagnation should stop early; strict OPTIMAL now requires true "
           "complementarity, while this flat-cost case is only primal acceptable.";
    EXPECT_LT(solver.get_iteration_count(), config.max_iters);
    EXPECT_EQ(info.status, status);
    EXPECT_EQ(info.termination_reason, TerminationReason::COST_STAGNATION);
    EXPECT_LT(info.iterations, config.max_iters);
    EXPECT_TRUE(std::isfinite(info.primal_inf));
    EXPECT_TRUE(std::isfinite(info.complementarity_inf));
}

// =============================================================================
// Feature: Parameter Persistence Through Solve (Ghost Cost Bug prevention)
// =============================================================================
TEST(FeaturesTest, ParameterPersistenceCheck)
{
    int N = 5;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.max_iters = 1;

    MiniSolver<CarModel, 10> solver(N, Backend::CPU_SERIAL, config);

    double magic_val = 123.456;
    solver.set_parameter(2, "x_ref", magic_val);

    EXPECT_DOUBLE_EQ(solver.get_parameter(2, "x_ref"), magic_val);

    solver.solve();

    double val_after = solver.get_parameter(2, "x_ref");
    EXPECT_DOUBLE_EQ(val_after, magic_val)
        << "Parameter lost after solve() iteration (Ghost Cost Bug)";
}

// =============================================================================
// Feature: GPU Backend Unsupported
// =============================================================================
TEST(FeaturesTest, GPUBackendUnsupportedFailsExplicitly)
{
    int N = 1;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.backend = Backend::GPU_MPX;
    config.max_iters = 1;

    MiniSolver<FlatCostModel, 10> solver(N, config.backend, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 1.0);

    SolverStatus status = solver.solve();

    EXPECT_EQ(status, SolverStatus::INVALID_INPUT)
        << "GPU backend is not implemented; it must not silently benchmark as CPU";
}

// =============================================================================
// Feature: dynamics-defect rollout refinement (runs without crash, maintains convergence)
// =============================================================================
TEST(FeaturesTest, DynamicsDefectRolloutRefinement)
{
    int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::DEBUG;
    config.integrator = IntegratorType::RK4_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.inertia_strategy = InertiaStrategy::REGULARIZATION;
    config.direction_refinement = DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT;

    MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("v", 0.0);

    for (int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "v_ref", 1.0);
        solver.set_parameter(k, "x_ref", k * 0.1);
        solver.set_parameter(k, "y_ref", 0.0);
        solver.set_parameter(k, "obs_x", 100.0);
        solver.set_parameter(k, "obs_y", 100.0);
        solver.set_parameter(k, "obs_rad", 1.0);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        solver.set_parameter(k, "w_pos", 10.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_theta", 0.1);
        solver.set_parameter(k, "w_acc", 0.1);
        solver.set_parameter(k, "w_steer", 1.0);
    }

    solver.rollout_dynamics();
    SolverStatus status = solver.solve();

    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
}

// =============================================================================
// Constrained-direction consistency anchor for OD-005.
//
// `DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT` only corrects dx/du via
// the existing Riccati feedback gains; it does not refine ds/dlam/dsoft_s.
// The overdesign-ledger flagged that this leaves slack/dual directions less
// consistent in constrained cases (potentially driving fraction-to-boundary
// violations or dual sign flips), but auto-disabling refinement was rejected
// as too strong. This test pins the runtime contract: with rollout refinement
// enabled and the upper control bound u <= 0.5 strongly active for the
// majority of the horizon, the solver must still
//   * reach OPTIMAL/FEASIBLE,
//   * respect both hard control bounds (no fraction-to-boundary breakage),
//   * keep every slack strictly positive (interior),
//   * keep every inequality dual finite and non-negative.
// If a future change makes rollout refinement break any of these invariants
// in the constrained regime, this test will fail and force either the
// auto-disable decision or a switch to full KKT iterative refinement
// rather than silently shipping inconsistency.
// =============================================================================
namespace {

struct ConstrainedRolloutModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 2;
    static constexpr int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "x_ref" };
    static constexpr std::array<double, NC> constraint_weights = { 0.0, 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0, 0 };

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
        kp.g_val(0) = kp.u(0) - static_cast<T>(0.5);
        kp.g_val(1) = static_cast<T>(-1.0) * kp.u(0) - static_cast<T>(0.5);
        kp.C.setZero();
        kp.D.setZero();
        kp.D(0, 0) = 1.0;
        kp.D(1, 0) = -1.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff = kp.x(0) - kp.p(0);
        const T u = kp.u(0);
        kp.cost = static_cast<T>(50.0) * diff * diff + static_cast<T>(0.1) * u * u;
        kp.q(0) = static_cast<T>(100.0) * diff;
        kp.r(0) = static_cast<T>(0.2) * u;
        kp.Q(0, 0) = 100.0;
        kp.R(0, 0) = 0.2;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

} // namespace

TEST(FeaturesTest, DefectRolloutRefinementKeepsConstrainedDirectionConsistent)
{
    constexpr int N = 12;
    constexpr double dt = 0.1;
    using Model = ConstrainedRolloutModel;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.barrier_strategy = BarrierStrategy::ADAPTIVE;
    config.line_search_type = LineSearchType::FILTER;
    config.direction_refinement = DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT;
    config.max_iters = 80;
    config.tol_con = 1e-6;
    config.tol_dual = 1e-5;
    config.mu_final = 1e-8;

    MiniSolver<Model, 16> solver(N, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(dt), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    for (int k = 0; k <= N; ++k) {
        ASSERT_EQ(solver.set_parameter(k, "x_ref", 5.0), ApiStatus::OK);
    }
    solver.rollout_dynamics();

    const SolverStatus status = solver.solve();
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE)
        << "Rollout refinement on a constrained problem must reach OPTIMAL/FEASIBLE; "
           "got "
        << status_to_string(status);

    int active_upper = 0;
    for (int k = 0; k < N; ++k) {
        const double u_k = solver.get_control(k, 0);
        EXPECT_LE(u_k, 0.5 + 1e-6) << "Hard upper bound must be respected at stage " << k;
        EXPECT_GE(u_k, -0.5 - 1e-6) << "Hard lower bound must be respected at stage " << k;
        if (u_k > 0.5 - 1e-3) {
            ++active_upper;
        }
        for (int c = 0; c < Model::NC; ++c) {
            const double slack = solver.get_slack(k, c);
            const double dual = solver.get_dual(k, c);
            EXPECT_GT(slack, 0.0) << "Slack must remain strictly positive (interior)";
            EXPECT_TRUE(std::isfinite(slack));
            EXPECT_TRUE(std::isfinite(dual));
            EXPECT_GE(dual, 0.0) << "Inequality dual must remain non-negative";
        }
    }
    EXPECT_GE(active_upper, N / 2)
        << "Test setup invariant: x_ref=5 must keep the upper bound active for at "
           "least half the stages so refinement actually exercises the constrained path.";
}
