/**
 * @file test_soft_constraints.cpp
 * @brief Tests for L1/L2 soft constraints: basic convergence + comparison
 *        between interface-based and manual-slack implementations.
 */
#include "minisolver/algorithms/initialization.h"
#include "minisolver/solver/riccati.h"
#include "minisolver/solver/solver.h"
#include "solver_internal_access.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace minisolver;

// Define SoftModel with mutable test-only soft structure.
struct SoftModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static std::array<bool, NC> constraint_has_l1;
    static std::array<bool, NC> constraint_has_l2;
    static std::array<double, NC> l1_weights;
    static std::array<double, NC> l2_weights;
    static constexpr bool any_l1_constraints = true;
    static constexpr bool any_l2_constraints = true;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};

    static void set_l1(double weight)
    {
        constraint_has_l1[0] = true;
        constraint_has_l2[0] = false;
        l1_weights[0] = weight;
        l2_weights[0] = 0.0;
    }

    static void set_l2(double weight)
    {
        constraint_has_l1[0] = false;
        constraint_has_l2[0] = true;
        l1_weights[0] = 0.0;
        l2_weights[0] = weight;
    }

    static void set_l1_l2(double l1_weight, double l2_weight)
    {
        constraint_has_l1[0] = true;
        constraint_has_l2[0] = true;
        l1_weights[0] = l1_weight;
        l2_weights[0] = l2_weight;
    }

    template <typename T>
    static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.l1_weight(0) = T(l1_weights[0]);
        kp.l2_weight(0) = T(l2_weights[0]);
    }

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt;
        return x_next;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        T x = kp.x(0);
        T u = kp.u(0);
        kp.f_resid(0) = x + u * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T x = kp.x(0);
        // x <= 5 -> x - 5 <= 0
        kp.g_val(0) = x - 5.0;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_impl(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T x = kp.x(0);
        T u = kp.u(0);
        // Cost: (x - 10)^2 + 1e-4 * u^2 (small regularization)
        T diff = x - 10.0;
        kp.cost = diff * diff + 1e-4 * u * u;

        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * u;

        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-4;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// Define static members
std::array<bool, SoftModel::NC> SoftModel::constraint_has_l1 = { false };
std::array<bool, SoftModel::NC> SoftModel::constraint_has_l2 = { false };
std::array<double, SoftModel::NC> SoftModel::l1_weights = { 0.0 };
std::array<double, SoftModel::NC> SoftModel::l2_weights = { 0.0 };

// New-style soft metadata: Python/codegen fixes the L1/L2 structure, and a
// generated updater fills per-knot runtime weights from parameters.
struct ParameterWeightedL1Model {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 1;

    static constexpr std::array<bool, NC> constraint_has_l1 = { true };
    static constexpr std::array<bool, NC> constraint_has_l2 = { false };
    static constexpr bool any_l1_constraints = true;
    static constexpr bool any_l2_constraints = false;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "w" };

    template <typename T>
    static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.l1_weight(0) = kp.p(0);
        kp.l2_weight(0) = T(0);
    }

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt;
        return x_next;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0) - 5.0;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_impl(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff = kp.x(0) - 10.0;
        kp.cost = diff * diff + 1e-4 * kp.u(0) * kp.u(0);
        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-4;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
};

TEST(SoftConstraintTest, L1_Convergence)
{
    // Setup L1
    SoftModel::set_l1(1.0);

    SolverConfig config;
    // config.print_level = PrintLevel::DEBUG;
    config.tol_con = 1e-4;
    config.tol_dual = 1e-4;
    config.max_iters = 50;

    // N=1
    MiniSolver<SoftModel, 5> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0); // dt=1

    // Initial guess x=0, u=10 (to reach 10)
    solver.set_initial_state("x", 0.0);
    solver.set_control_guess(0, "u", 10.0);
    solver.rollout_dynamics();

    // Theoretical L1 Opt: min (x-10)^2 + 1*max(0, x-5)
    // At x=5: cost 25.
    // For x > 5: cost (x-10)^2 + (x-5). deriv 2(x-10)+1 = 2x-19=0 -> x=9.5. Cost 0.25+4.5 = 4.75.
    // For x <= 5: cost (x-10)^2. min at x=5 (constrained).
    // Global min x=9.5.

    solver.solve();

    double x_final = solver.get_state(1, 0);
    std::cout << "L1 Final X: " << x_final << std::endl;

    EXPECT_NEAR(x_final, 9.5, 1.0e-3);
}

TEST(SoftConstraintTest, L1RuntimeParameterWeightAffectsSolve)
{
    SolverConfig config;
    config.tol_con = 1e-4;
    config.tol_dual = 1e-4;
    config.max_iters = 80;

    MiniSolver<ParameterWeightedL1Model, 5> solver(1, Backend::CPU_SERIAL, config);
    ASSERT_EQ(solver.set_dt(1.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_initial_state("x", 0.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_control_guess(0, "u", 10.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_parameter(0, "w", 2.0), ApiStatus::OK);
    ASSERT_EQ(solver.set_parameter(1, "w", 2.0), ApiStatus::OK);
    solver.rollout_dynamics();

    ASSERT_NE(solver.solve(), SolverStatus::NUMERICAL_ERROR);

    const double x_final = solver.get_state(1, 0);
    EXPECT_NEAR(x_final, 9.0, 1.0e-3);
}

TEST(SoftConstraintTest, L1InvalidDualWarmStartFallsBackSafely)
{
    SoftModel::set_l1(1.0);

    SolverConfig config;
    config.tol_con = 1e-4;
    config.tol_dual = 1e-4;
    config.max_iters = 50;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;

    MiniSolver<SoftModel, 5> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    solver.set_initial_state("x", 0.0);
    solver.set_control_guess(0, "u", 10.0);
    solver.rollout_dynamics();

    solver.set_slack_guess(0, 0, 0.1);
    solver.set_dual_guess(0, 0, SoftModel::l1_weights[0]);
    solver.set_slack_guess(1, 0, 0.1);
    solver.set_dual_guess(1, 0, SoftModel::l1_weights[0]);

    SolverStatus status = solver.solve();
    EXPECT_NE(status, SolverStatus::NUMERICAL_ERROR);
    EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
    EXPECT_NEAR(solver.get_state(1, 0), 9.5, 1.0e-3);
}

TEST(SoftConstraintTest, L1TinyWeightInitializationStaysFinite)
{
    SoftModel::set_l1(1e-7);

    KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP> kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-10;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu);

    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_TRUE(std::isfinite(kp.soft_s(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_LT(kp.lam(0), SoftModel::l1_weights[0]);
    EXPECT_GT(kp.soft_s(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-14);
    EXPECT_NEAR(kp.soft_s(0) * (SoftModel::l1_weights[0] - kp.lam(0)), mu, 1e-14);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.soft_s(0), 0.0, 1e-8);
}

TEST(SoftConstraintTest, L1BarrierDerivativeUsesSharedSoftDualFloor)
{
    SoftModel::set_l1(1e-7);

    SolverConfig config;
    config.min_barrier_slack = 1e-12;

    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.s(0) = 1.0;
    kp.lam(0) = SoftModel::l1_weights[0] - 1.5e-12;
    kp.soft_s(0) = 1.0;
    kp.C(0, 0) = 1.0;
    SoftModel::update_soft_constraint_weights(kp);

    RiccatiWorkspace<Knot> workspace;
    compute_barrier_derivatives<Knot, SoftModel>(kp, 1e-10, config, workspace);

    const double soft_dual_floor = detail::l1_soft_dual_floor(SoftModel::l1_weights[0], config);
    const double expected_sigma = 1.0 / (kp.s(0) / kp.lam(0) + kp.soft_s(0) / soft_dual_floor);

    EXPECT_NEAR(kp.Q_bar(0, 0) - kp.Q(0, 0), expected_sigma, 1e-15)
        << "Riccati barrier derivatives must use the same L1 soft-dual floor as "
           "initialization/restoration.";
}

TEST(SoftConstraintTest, MixedL1L2InitializationUsesCombinedSoftDual)
{
    SoftModel::set_l1_l2(2.0, 3.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.35;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double soft_dual
        = SoftModel::l1_weights[0] + SoftModel::l2_weights[0] * kp.soft_s(0) - kp.lam(0);
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_GT(kp.soft_s(0), 0.0);
    EXPECT_GT(soft_dual, detail::l1_soft_dual_floor(SoftModel::l1_weights[0], config));
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.soft_s(0) * soft_dual, mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.soft_s(0), 0.0, 1e-12);
}

TEST(SoftConstraintTest, MixedL1L2BarrierDerivativeUsesBothWeights)
{
    SoftModel::set_l1_l2(2.0, 3.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.s(0) = 1.25;
    kp.lam(0) = 0.7;
    kp.soft_s(0) = 0.4;
    kp.C(0, 0) = 1.0;
    SoftModel::update_soft_constraint_weights(kp);

    RiccatiWorkspace<Knot> workspace;
    compute_barrier_derivatives<Knot, SoftModel>(kp, 5e-2, config, workspace);

    const double soft_dual
        = SoftModel::l1_weights[0] + SoftModel::l2_weights[0] * kp.soft_s(0) - kp.lam(0);
    const double soft_denom = soft_dual + SoftModel::l2_weights[0] * kp.soft_s(0);
    const double expected_sigma = 1.0 / (kp.s(0) / kp.lam(0) + kp.soft_s(0) / soft_denom);

    EXPECT_NEAR(kp.Q_bar(0, 0) - kp.Q(0, 0), expected_sigma, 1e-15)
        << "Same-row L1+L2 must use the combined soft dual "
           "w1 + w2*soft_s - lam, not the pure L1 or pure L2 branch.";
}

TEST(SoftConstraintTest, MixedL1L2DualRecoveryUsesBothWeights)
{
    SoftModel::set_l1_l2(2.0, 3.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.s(0) = 1.25;
    kp.lam(0) = 0.7;
    kp.soft_s(0) = 0.4;
    kp.g_val(0) = -0.85;
    SoftModel::update_soft_constraint_weights(kp);

    constexpr double mu = 5e-2;
    const double soft_dual
        = SoftModel::l1_weights[0] + SoftModel::l2_weights[0] * kp.soft_s(0) - kp.lam(0);
    const double soft_denom = soft_dual + SoftModel::l2_weights[0] * kp.soft_s(0);
    const double r_y = kp.s(0) * kp.lam(0) - mu;
    const double r_eq = kp.g_val(0) + kp.s(0) - kp.soft_s(0);
    const double r_z = kp.soft_s(0) * soft_dual - mu;
    const double sigma = 1.0 / (kp.s(0) / kp.lam(0) + kp.soft_s(0) / soft_denom);
    const double expected_dlam = sigma * (r_eq - r_y / kp.lam(0) + r_z / soft_denom);
    const double expected_ds = (-r_y - kp.s(0) * expected_dlam) / kp.lam(0);
    const double expected_dsoft_s = -(r_z - kp.soft_s(0) * expected_dlam) / soft_denom;

    recover_dual_search_directions<Knot, SoftModel>(kp, mu, config);

    EXPECT_NEAR(kp.dlam(0), expected_dlam, 1e-15);
    EXPECT_NEAR(kp.ds(0), expected_ds, 1e-15);
    EXPECT_NEAR(kp.dsoft_s(0), expected_dsoft_s, 1e-15);
}

TEST(SoftConstraintTest, L1NegativeSoftDualDoesNotReduceAverageComplementarity)
{
    SoftModel::set_l1(1.0);

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.min_barrier_slack = 1e-12;
    config.mu_init = 1e-4;

    MiniSolver<SoftModel, 5> solver(0, Backend::CPU_SERIAL, config);
    solver.set_initial_state("x", 5.0);

    using Access = minisolver::test::SolverInternalAccess<SoftModel, 5>;
    auto& traj = Access::get_trajectory(solver);
    traj[0].s(0) = 1.0;
    traj[0].lam(0) = 2.0;
    traj[0].soft_s(0) = 3.0;

    const StepResidualSummary residuals = Access::evaluate_step_model(solver, traj);
    const double expected_floor_gap
        = 3.0 * detail::l1_soft_dual_floor(SoftModel::l1_weights[0], config);

    EXPECT_NEAR(residuals.max_complementarity_gap, 3.0, 1e-12)
        << "Diagnostics should still expose the raw invalid soft dual box.";
    EXPECT_NEAR(residuals.avg_complementarity_gap, (2.0 + expected_floor_gap) / 2.0, 1e-12)
        << "The barrier-update average should not be reduced by negative L1 soft-dual gap.";
}

TEST(SoftConstraintTest, L2_Convergence)
{
    // Setup L2
    SoftModel::set_l2(1.0);

    SolverConfig config;
    config.tol_con = 1e-4;
    config.max_iters = 50;

    MiniSolver<SoftModel, 5> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    // Start at x=10 (Cost min, but Feasibility violation)
    solver.set_initial_state("x", 10.0);
    solver.set_control_guess(0, "u", 0.0);
    solver.rollout_dynamics();

    // Theoretical L2 Opt: x = 8.333
    // From x=10, cost increases, but violation decreases.
    // If phi doesn't include penalty, cost increases -> phi increases.
    // theta decreases.
    // Filter accepts if theta decreases enough.

    solver.solve();

    double x_final = solver.get_state(1, 0);
    std::cout << "L2 Final X (from x=10): " << x_final << std::endl;

    EXPECT_NEAR(x_final, 8.333, 1.0e-3);
}

TEST(SoftConstraintTest, MixedL1L2_Convergence)
{
    SoftModel::set_l1_l2(1.0, 2.0);

    SolverConfig config;
    config.tol_con = 1e-4;
    config.tol_dual = 1e-4;
    config.max_iters = 80;

    MiniSolver<SoftModel, 5> solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    solver.set_initial_state("x", 0.0);
    solver.set_control_guess(0, "u", 10.0);
    solver.rollout_dynamics();

    const SolverStatus status = solver.solve();

    const double x_final = solver.get_state(1, 0);
    std::cout << "Mixed L1/L2 Final X: " << x_final << std::endl;

    EXPECT_NE(status, SolverStatus::NUMERICAL_ERROR);
    EXPECT_NEAR(x_final, 7.25, 1.0e-3) << "For x > 5, the same-row penalty has derivative "
                                          "1 + 2 * (x - 5), not just the pure L1 term.";
}

TEST(SoftConstraintTest, ZeroL2WeightInitializesAsRegularizedSoftRow)
{
    SoftModel::set_l2(0.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double w_eff = detail::barrier_floor(config);
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / w_eff, 0.0, 1e-8)
        << "A structurally soft L2 row with zero runtime weight should use the "
           "regularized L2 path, not fall back to hard-constraint initialization.";
}

TEST(SoftConstraintTest, ZeroL1WeightInitializesAsRegularizedSoftRow)
{
    SoftModel::set_l1(0.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double w_eff = detail::barrier_floor(config);
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / w_eff, 0.0, 1e-8)
        << "An inactive structural L1 row should use the same regularized "
           "soft-row path as zero L2 weight.";
}

TEST(SoftConstraintTest, MixedL1L2WithZeroL1UsesL2Path)
{
    SoftModel::set_l1_l2(0.0, 3.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / SoftModel::l2_weights[0], 0.0, 1e-12);
}

TEST(SoftConstraintTest, MixedL1L2WithZeroL2UsesL1Path)
{
    SoftModel::set_l1_l2(2.0, 0.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double soft_dual = SoftModel::l1_weights[0] - kp.lam(0);
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_TRUE(std::isfinite(kp.soft_s(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_GT(kp.soft_s(0), 0.0);
    EXPECT_GT(soft_dual, detail::l1_soft_dual_floor(SoftModel::l1_weights[0], config));
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.soft_s(0) * soft_dual, mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.soft_s(0), 0.0, 1e-12);
}

TEST(SoftConstraintTest, MixedL1L2WithBothZeroUsesRegularizedSoftRow)
{
    SoftModel::set_l1_l2(0.0, 0.0);

    SolverConfig config;
    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double w_eff = detail::barrier_floor(config);
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / w_eff, 0.0, 1e-8);
}

TEST(SoftConstraintTest, TinyInactiveL1WeightUsesRegularizedSoftRow)
{
    SolverConfig config;
    SoftModel::set_l1(detail::barrier_floor(config));

    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double w_eff = detail::barrier_floor(config);
    EXPECT_FALSE(detail::active_l1_soft_constraint<SoftModel>(kp, 0, config));
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / w_eff, 0.0, 1e-8);
}

TEST(SoftConstraintTest, TinyL2WeightUsesEffectiveFloor)
{
    SolverConfig config;
    SoftModel::set_l2(0.5 * detail::barrier_floor(config));

    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double w_eff = detail::barrier_floor(config);
    EXPECT_TRUE(detail::active_l2_soft_constraint<SoftModel>(kp, 0));
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / w_eff, 0.0, 1e-8);
}

TEST(SoftConstraintTest, MixedL1L2WithTinyInactiveL1UsesL2Path)
{
    SolverConfig config;
    SoftModel::set_l1_l2(detail::barrier_floor(config), 3.0);

    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    EXPECT_FALSE(detail::active_l1_soft_constraint<SoftModel>(kp, 0, config));
    EXPECT_TRUE(detail::active_l2_soft_constraint<SoftModel>(kp, 0));
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / SoftModel::l2_weights[0], 0.0, 1e-12);
}

TEST(SoftConstraintTest, MixedL1L2WithTinyPositiveL2StillUsesMixedPath)
{
    SolverConfig config;
    SoftModel::set_l1_l2(2.0, 0.5 * detail::barrier_floor(config));

    using Knot = KnotPoint<double, SoftModel::NX, SoftModel::NU, SoftModel::NC, SoftModel::NP>;
    Knot kp;
    kp.set_zero();
    kp.g_val(0) = -0.5;
    constexpr double mu = 1e-4;

    detail::InitializationKernel::initialize_constraint_primal_dual<SoftModel>(kp, 0, mu, config);

    const double soft_dual
        = SoftModel::l1_weights[0] + SoftModel::l2_weights[0] * kp.soft_s(0) - kp.lam(0);
    EXPECT_TRUE(detail::active_mixed_l1_l2_soft_constraint<SoftModel>(kp, 0, config));
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_TRUE(std::isfinite(kp.soft_s(0)));
    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_GT(kp.soft_s(0), 0.0);
    EXPECT_GT(soft_dual, detail::l1_soft_dual_floor(SoftModel::l1_weights[0], config));
    EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    EXPECT_NEAR(kp.soft_s(0) * soft_dual, mu, 1e-12);
    EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.soft_s(0), 0.0, 1e-12);
}

// ==========================================
// 1. Interface Model (Benchmark)
// Uses built-in L1/L2 soft constraint logic
// ==========================================
struct InterfaceModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static std::array<bool, NC> constraint_has_l1;
    static std::array<bool, NC> constraint_has_l2;
    static std::array<double, NC> l1_weights;
    static std::array<double, NC> l2_weights;
    static constexpr bool any_l1_constraints = true;
    static constexpr bool any_l2_constraints = true;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};

    static void set_l1(double weight)
    {
        constraint_has_l1[0] = true;
        constraint_has_l2[0] = false;
        l1_weights[0] = weight;
        l2_weights[0] = 0.0;
    }

    static void set_l2(double weight)
    {
        constraint_has_l1[0] = false;
        constraint_has_l2[0] = true;
        l1_weights[0] = 0.0;
        l2_weights[0] = weight;
    }

    static void set_l1_l2(double l1_weight, double l2_weight)
    {
        constraint_has_l1[0] = true;
        constraint_has_l2[0] = true;
        l1_weights[0] = l1_weight;
        l2_weights[0] = l2_weight;
    }

    template <typename T>
    static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.l1_weight(0) = T(l1_weights[0]);
        kp.l2_weight(0) = T(l2_weights[0]);
    }

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        return x + u * dt;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        // Constraint: x <= 5.0
        kp.g_val(0) = kp.x(0) - 5.0;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T diff = kp.x(0) - 10.0; // Target x = 10
        kp.cost = diff * diff + 1e-4 * kp.u(0) * kp.u(0);
        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-4;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

std::array<bool, 1> InterfaceModel::constraint_has_l1 = { false };
std::array<bool, 1> InterfaceModel::constraint_has_l2 = { false };
std::array<double, 1> InterfaceModel::l1_weights = { 0.0 };
std::array<double, 1> InterfaceModel::l2_weights = { 0.0 };

// ==========================================
// 2. Manual L1 Model
// Explicitly adds 'slk' as a control variable
// ==========================================
struct ManualL1Model {
    static const int NX = 1;
    static const int NU = 2; // [u, slk]
    static const int NC = 2; // [g-slk, -slk]
    static const int NP = 0;

    static constexpr std::array<double, NC> constraint_weights = { 0.0, 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0, 0 };

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u", "slk" };
    static constexpr std::array<const char*, NP> param_names = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt; // slk (u[1]) does not affect dynamics
        return x_next;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
        kp.B(0, 1) = 0.0;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T x = kp.x(0);
        T slk = kp.u(1);

        // 1. x - 5 - slk <= 0
        kp.g_val(0) = x - 5.0 - slk;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
        kp.D(0, 1) = -1.0;

        // 2. -slk <= 0 (Non-negative slack)
        kp.g_val(1) = -slk;
        kp.C(1, 0) = 0.0;
        kp.D(1, 0) = 0.0;
        kp.D(1, 1) = -1.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T diff = kp.x(0) - 10.0;
        T slk = kp.u(1);
        double w = 1.0; // Fixed L1 Weight

        // Cost: (x-10)^2 + w*slk
        kp.cost = diff * diff + 1e-4 * kp.u(0) * kp.u(0) + w * slk;

        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.r(1) = w; // Gradient w.r.t slk is constant w

        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-4;
        kp.R(1, 1) = 0.0; // Linear cost -> 0 Hessian
        kp.R(0, 1) = 0.0;
        kp.R(1, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// ==========================================
// 3. Manual L2 Model
// Uses 0.5 * w * slk^2 to match Interface
// ==========================================
struct ManualL2Model {
    static const int NX = 1;
    static const int NU = 2;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };
    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u", "slk" };
    static constexpr std::array<const char*, NP> param_names = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt;
        return x_next;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
        kp.B(0, 1) = 0.0;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T x = kp.x(0);
        T slk = kp.u(1);
        // x - 5 - slk <= 0
        kp.g_val(0) = x - 5.0 - slk;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
        kp.D(0, 1) = -1.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T diff = kp.x(0) - 10.0;
        T slk = kp.u(1);
        double w = 1.0; // Fixed L2 Weight

        // Cost: 0.5 * w * slk^2 (Matches MiniSolver Interface L2 formulation)
        kp.cost = diff * diff + 1e-4 * kp.u(0) * kp.u(0) + 0.5 * w * slk * slk;

        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.r(1) = w * slk;

        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-4;
        kp.R(1, 1) = w;
        kp.R(0, 1) = 0.0;
        kp.R(1, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// ==========================================
// 4. Manual Mixed L1+L2 Model
// Uses w1 * slk + 0.5 * w2 * slk^2 to match Interface
// ==========================================
struct ManualMixedL1L2Model {
    static const int NX = 1;
    static const int NU = 2; // [u, slk]
    static const int NC = 2; // [g-slk, -slk]
    static const int NP = 0;

    static constexpr std::array<double, NC> constraint_weights = { 0.0, 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0, 0 };

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u", "slk" };
    static constexpr std::array<const char*, NP> param_names = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt;
        return x_next;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
        kp.B(0, 1) = 0.0;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T x = kp.x(0);
        const T slk = kp.u(1);

        kp.g_val(0) = x - 5.0 - slk;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
        kp.D(0, 1) = -1.0;

        kp.g_val(1) = -slk;
        kp.C(1, 0) = 0.0;
        kp.D(1, 0) = 0.0;
        kp.D(1, 1) = -1.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff = kp.x(0) - 10.0;
        const T slk = kp.u(1);
        constexpr double w1 = 1.0;
        constexpr double w2 = 2.0;

        kp.cost = diff * diff + 1e-4 * kp.u(0) * kp.u(0) + w1 * slk + 0.5 * w2 * slk * slk;

        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.r(1) = w1 + w2 * slk;

        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-4;
        kp.R(1, 1) = w2;
        kp.R(0, 1) = 0.0;
        kp.R(1, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_exact(kp);
    }
    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// ==========================================
// 5. Comparison Tests
// ==========================================

TEST(ComparisonTest, L1_SoftConstraint)
{
    SolverConfig config;
    config.tol_con = 1e-4;
    config.max_iters = 100;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    // We use N=2 to ensure 'slk' (as a control input) is optimized at the intermediate node.
    // k=0: x0 -> u0 -> x1
    // k=1: x1 -> u1(slk) -> x2. Constraint applies here.
    int N = 2;

    // --- 1. Interface (Ground Truth) ---
    InterfaceModel::set_l1(1.0);

    MiniSolver<InterfaceModel, 5> solver_if(N, Backend::CPU_SERIAL, config);
    solver_if.set_dt(1.0);
    solver_if.set_initial_state("x", 0.0);
    solver_if.set_control_guess(0, "u", 10.0); // Reach 10 at k=1
    solver_if.set_control_guess(1, "u", 0.0);
    solver_if.rollout_dynamics();
    solver_if.solve();
    double x_if = solver_if.get_state(1, 0); // Check x at k=1

    // --- 2. Manual Model ---
    MiniSolver<ManualL1Model, 5> solver_man(N, Backend::CPU_SERIAL, config);
    solver_man.set_dt(1.0);
    solver_man.set_initial_state("x", 0.0);
    solver_man.set_control_guess(0, "u", 10.0);
    solver_man.set_control_guess(1, "u", 0.0);

    // Manual Initialization for k=1 (where x=10 approx)
    // x ~ 10. Constraint x - 5 - slk <= 0.
    // To satisfy, slk needs to be 5.
    double slk_init = 5.0;
    solver_man.set_control_guess(1, "slk", slk_init);

    solver_man.rollout_dynamics();

    // [Init Dual Variables]
    // For L1: Stationarity implies Lambda = Weight = 1.0
    solver_man.set_slack_guess(1, 0, 0.01); // Small slack for active constraint
    solver_man.set_dual_guess(1, 0, 1.0); // L1 Dual = Weight

    solver_man.set_slack_guess(1, 1, slk_init); // Inactive non-negative constraint
    solver_man.set_dual_guess(1, 1, 0.01);

    SolverConfig cfg_l1 = solver_man.get_config();
    cfg_l1.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    solver_man.set_config(cfg_l1);
    solver_man.solve();
    double x_man = solver_man.get_state(1, 0);

    std::cout << "[L1 Comparison N=2] Interface x1: " << x_if << " vs Manual x1: " << x_man
              << std::endl;

    // Theoretical Opt for L1: x = 9.5
    EXPECT_NEAR(x_if, 9.5, 1e-3);
    EXPECT_NEAR(x_if, x_man, 1e-3);
}

TEST(ComparisonTest, L2_SoftConstraint)
{
    SolverConfig config;
    config.tol_con = 1e-4;
    config.max_iters = 100;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    int N = 2;

    // --- 1. Interface ---
    InterfaceModel::set_l2(1.0);

    MiniSolver<InterfaceModel, 5> solver_if(N, Backend::CPU_SERIAL, config);
    solver_if.set_dt(1.0);
    solver_if.set_initial_state("x", 0.0);
    solver_if.set_control_guess(0, "u", 10.0);
    solver_if.set_control_guess(1, "u", 0.0);
    solver_if.rollout_dynamics();
    solver_if.solve();
    double x_if = solver_if.get_state(1, 0);

    // --- 2. Manual Model ---
    MiniSolver<ManualL2Model, 5> solver_man(N, Backend::CPU_SERIAL, config);
    solver_man.set_dt(1.0);
    solver_man.set_initial_state("x", 0.0);
    solver_man.set_control_guess(0, "u", 10.0);
    solver_man.set_control_guess(1, "u", 0.0);

    // Initialize slk at k=1
    double slk_init = 5.0;
    solver_man.set_control_guess(1, "slk", slk_init);

    solver_man.rollout_dynamics();

    // [Init Dual Variables]
    // For L2: Stationarity implies Lambda = Weight * Slack
    // Lam = 1.0 * 5.0 = 5.0
    solver_man.set_slack_guess(1, 0, 0.01);
    solver_man.set_dual_guess(1, 0, 5.0);

    SolverConfig cfg_l2 = solver_man.get_config();
    cfg_l2.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    solver_man.set_config(cfg_l2);
    solver_man.solve();
    double x_man = solver_man.get_state(1, 0);

    std::cout << "[L2 Comparison N=2] Interface x1: " << x_if << " vs Manual x1: " << x_man
              << std::endl;

    // Theoretical Opt for L2: x = 25/3 = 8.333...
    EXPECT_NEAR(x_if, 8.333, 1e-3);
    EXPECT_NEAR(x_if, x_man, 1e-3);
}

TEST(ComparisonTest, MixedL1L2_SoftConstraint)
{
    SolverConfig config;
    config.tol_con = 1e-4;
    config.tol_dual = 1e-4;
    config.max_iters = 100;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    int N = 2;

    InterfaceModel::set_l1_l2(1.0, 2.0);

    MiniSolver<InterfaceModel, 5> solver_if(N, Backend::CPU_SERIAL, config);
    solver_if.set_dt(1.0);
    solver_if.set_initial_state("x", 0.0);
    solver_if.set_control_guess(0, "u", 10.0);
    solver_if.set_control_guess(1, "u", 0.0);
    solver_if.rollout_dynamics();
    const SolverStatus status_if = solver_if.solve();
    const double x_if = solver_if.get_state(1, 0);
    const double soft_s_if = solver_if.get_constraint_val(1, 0) + solver_if.get_slack(1, 0);

    MiniSolver<ManualMixedL1L2Model, 5> solver_man(N, Backend::CPU_SERIAL, config);
    solver_man.set_dt(1.0);
    solver_man.set_initial_state("x", 0.0);
    solver_man.set_control_guess(0, "u", 10.0);
    solver_man.set_control_guess(1, "u", 0.0);

    double slk_init = 5.0;
    solver_man.set_control_guess(1, "slk", slk_init);
    solver_man.rollout_dynamics();

    solver_man.set_slack_guess(1, 0, 0.01);
    solver_man.set_dual_guess(1, 0, 1.0 + 2.0 * slk_init);
    solver_man.set_slack_guess(1, 1, slk_init);
    solver_man.set_dual_guess(1, 1, 0.01);

    SolverConfig cfg_mixed = solver_man.get_config();
    cfg_mixed.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    solver_man.set_config(cfg_mixed);
    const SolverStatus status_man = solver_man.solve();
    const double x_man = solver_man.get_state(1, 0);
    const double slk_man = solver_man.get_control(1, 1);

    std::cout << "[Mixed L1/L2 Comparison N=2] Interface x1: " << x_if << " soft_s: " << soft_s_if
              << " vs Manual x1: " << x_man << " slk: " << slk_man << std::endl;

    EXPECT_NE(status_if, SolverStatus::NUMERICAL_ERROR);
    EXPECT_NE(status_man, SolverStatus::NUMERICAL_ERROR);
    EXPECT_NEAR(x_if, 7.25, 1e-3);
    EXPECT_NEAR(x_if, x_man, 1e-3);
    EXPECT_NEAR(soft_s_if, slk_man, 1e-3)
        << "Mixed soft constraints must use one shared relaxation variable, "
           "matching an explicit slk in w1*slk + 0.5*w2*slk^2.";
}
