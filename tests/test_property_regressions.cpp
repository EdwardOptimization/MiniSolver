#include "minisolver/algorithms/initialization.h"
#include "minisolver/algorithms/model_evaluation.h"
#include "minisolver/core/trajectory.h"
#include "minisolver/solver/line_search_utils.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace minisolver;

namespace {

struct HardConstraintModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
};

struct L1SoftConstraintModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr double soft_weight = 2.0;
    static constexpr std::array<bool, NC> constraint_has_l1 = { true };
    static constexpr std::array<bool, NC> constraint_has_l2 = { false };

    template <typename T>
    static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.l1_weight(0) = T(soft_weight);
        kp.l2_weight(0) = T(0);
    }
};

struct L2SoftConstraintModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr double soft_weight = 2.0;
    static constexpr std::array<bool, NC> constraint_has_l1 = { false };
    static constexpr std::array<bool, NC> constraint_has_l2 = { true };

    template <typename T>
    static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.l1_weight(0) = T(0);
        kp.l2_weight(0) = T(soft_weight);
    }
};

struct MixedL1L2SoftConstraintModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr double l1_weight = 2.0;
    static constexpr double l2_weight = 3.0;
    static constexpr std::array<bool, NC> constraint_has_l1 = { true };
    static constexpr std::array<bool, NC> constraint_has_l2 = { true };

    template <typename T>
    static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.l1_weight(0) = T(l1_weight);
        kp.l2_weight(0) = T(l2_weight);
    }
};

struct ScalingPropertyModel {
    static constexpr int NX = 2;
    static constexpr int NU = 1;
    static constexpr int NC = 2;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x0", "x1" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
};

double row_inf_norm(const KnotPoint<double, ScalingPropertyModel::NX, ScalingPropertyModel::NU,
                        ScalingPropertyModel::NC, ScalingPropertyModel::NP>& kp,
    int row)
{
    double norm = std::abs(kp.g_unscaled(row));
    for (int col = 0; col < ScalingPropertyModel::NX; ++col) {
        norm = std::max(norm, std::abs(kp.C(row, col)));
    }
    for (int col = 0; col < ScalingPropertyModel::NU; ++col) {
        norm = std::max(norm, std::abs(kp.D(row, col)));
    }
    return norm;
}

double objective_hessian_row_bound(const KnotPoint<double, ScalingPropertyModel::NX,
    ScalingPropertyModel::NU, 0, ScalingPropertyModel::NP>& kp)
{
    double bound = 0.0;
    for (int row = 0; row < ScalingPropertyModel::NX; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < ScalingPropertyModel::NX; ++col) {
            row_sum += std::abs(kp.Q(row, col));
        }
        for (int col = 0; col < ScalingPropertyModel::NU; ++col) {
            row_sum += std::abs(kp.H(col, row));
        }
        bound = std::max(bound, row_sum);
    }
    for (int row = 0; row < ScalingPropertyModel::NU; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < ScalingPropertyModel::NX; ++col) {
            row_sum += std::abs(kp.H(row, col));
        }
        for (int col = 0; col < ScalingPropertyModel::NU; ++col) {
            row_sum += std::abs(kp.R(row, col));
        }
        bound = std::max(bound, row_sum);
    }
    return bound;
}

} // namespace

TEST(PropertyRegressionTest, HardConstraintInitializationStaysOnCentralPath)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<double> g_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> log_mu_dist(-8.0, -1.0);

    for (int sample = 0; sample < 200; ++sample) {
        const double mu = std::pow(10.0, log_mu_dist(rng));

        Knot kp;
        kp.set_zero();
        kp.g_val(0) = g_dist(rng);

        detail::InitializationKernel::initialize_constraint_primal_dual<HardConstraintModel>(
            kp, 0, mu);

        EXPECT_GT(kp.s(0), 0.0);
        EXPECT_GT(kp.lam(0), 0.0);
        EXPECT_TRUE(std::isfinite(kp.s(0)));
        EXPECT_TRUE(std::isfinite(kp.lam(0)));
        EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-12);
    }
}

TEST(PropertyRegressionTest, ViolatedHardConstraintInitializationScalesSlackWithResidual)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;

    SolverConfig config;
    config.mu_init = 1e-1;
    config.warm_start_slack_init = 1e-6;
    config.min_barrier_slack = 1e-12;

    constexpr double g = 120.0;

    Knot kp;
    kp.set_zero();
    kp.g_val(0) = g;

    detail::InitializationKernel::initialize_constraint_primal_dual<HardConstraintModel>(
        kp, 0, config.mu_init, config);

    EXPECT_GT(kp.s(0), 0.0);
    EXPECT_GT(kp.lam(0), 0.0);
    EXPECT_TRUE(std::isfinite(kp.s(0)));
    EXPECT_TRUE(std::isfinite(kp.lam(0)));
    EXPECT_NEAR(kp.s(0) * kp.lam(0), config.mu_init, 1e-12);

    // With g > 0 there is no positive slack that can satisfy g + s = 0.
    // Initialization should therefore keep the barrier variables interior at
    // the violation scale instead of putting s at the tiny floor, which would
    // make lambda / s explode in the first Riccati/IPM system.
    EXPECT_GE(kp.s(0), g);
    EXPECT_LE(kp.lam(0) / kp.s(0), 10.0 * config.mu_init / (g * g));
}

TEST(PropertyRegressionTest, L1SoftInitializationStaysInsideDualBox)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    constexpr double w = L1SoftConstraintModel::soft_weight;

    std::mt19937 rng(0x51A7E);
    std::uniform_real_distribution<double> g_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> log_mu_dist(-8.0, -4.0);

    for (int sample = 0; sample < 200; ++sample) {
        const double mu = std::pow(10.0, log_mu_dist(rng));

        Knot kp;
        kp.set_zero();
        kp.g_val(0) = g_dist(rng);

        detail::InitializationKernel::initialize_constraint_primal_dual<L1SoftConstraintModel>(
            kp, 0, mu);

        EXPECT_GT(kp.s(0), 0.0);
        EXPECT_GT(kp.lam(0), 0.0);
        EXPECT_LT(kp.lam(0), w);
        EXPECT_GT(kp.soft_s(0), 0.0);
        EXPECT_TRUE(std::isfinite(kp.s(0)));
        EXPECT_TRUE(std::isfinite(kp.lam(0)));
        EXPECT_TRUE(std::isfinite(kp.soft_s(0)));
        EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-10);
        EXPECT_NEAR(kp.soft_s(0) * (w - kp.lam(0)), mu, 1e-10);
        EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.soft_s(0), 0.0, 1e-8);
    }
}

TEST(PropertyRegressionTest, L2SoftInitializationSatisfiesCentralPathResidual)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    constexpr double w = L2SoftConstraintModel::soft_weight;

    std::mt19937 rng(0x1202);
    std::uniform_real_distribution<double> g_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> log_mu_dist(-8.0, -3.0);

    for (int sample = 0; sample < 200; ++sample) {
        const double mu = std::pow(10.0, log_mu_dist(rng));

        Knot kp;
        kp.set_zero();
        kp.g_val(0) = g_dist(rng);

        detail::InitializationKernel::initialize_constraint_primal_dual<L2SoftConstraintModel>(
            kp, 0, mu);

        EXPECT_GT(kp.s(0), 0.0);
        EXPECT_GT(kp.lam(0), 0.0);
        EXPECT_TRUE(std::isfinite(kp.s(0)));
        EXPECT_TRUE(std::isfinite(kp.lam(0)));
        EXPECT_NEAR(kp.s(0) * kp.lam(0), mu, 1e-10);
        EXPECT_NEAR(kp.g_val(0) + kp.s(0) - kp.lam(0) / w, 0.0, 1e-8);
    }
}

TEST(PropertyRegressionTest, FractionToBoundaryKeepsHardAndL1VariablesInterior)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    constexpr int MaxN = 3;
    constexpr int N = 3;
    constexpr double tau = 0.95;
    constexpr double w = L1SoftConstraintModel::soft_weight;

    std::mt19937 rng(0xB0A7);
    std::uniform_real_distribution<double> positive_dist(1e-3, 1.0);
    std::uniform_real_distribution<double> dual_dist(1e-3, w - 1e-3);
    std::uniform_real_distribution<double> direction_dist(-3.0, 3.0);

    for (int sample = 0; sample < 200; ++sample) {
        std::array<Knot, MaxN + 1> traj;
        for (int k = 0; k <= N; ++k) {
            traj[k].set_zero();
            traj[k].s(0) = positive_dist(rng);
            traj[k].lam(0) = dual_dist(rng);
            traj[k].soft_s(0) = positive_dist(rng);
            traj[k].ds(0) = direction_dist(rng);
            traj[k].dlam(0) = direction_dist(rng);
            traj[k].dsoft_s(0) = direction_dist(rng);
            L1SoftConstraintModel::update_soft_constraint_weights(traj[k]);
        }

        const double alpha
            = fraction_to_boundary_rule<decltype(traj), L1SoftConstraintModel>(traj, N, tau);

        ASSERT_GE(alpha, 0.0);
        ASSERT_LE(alpha, 1.0);

        for (int k = 0; k <= N; ++k) {
            const double s_new = traj[k].s(0) + alpha * traj[k].ds(0);
            const double lam_new = traj[k].lam(0) + alpha * traj[k].dlam(0);
            const double soft_s_new = traj[k].soft_s(0) + alpha * traj[k].dsoft_s(0);
            const double soft_dual_new = w - lam_new;

            EXPECT_GT(s_new, -1e-12);
            EXPECT_GT(lam_new, -1e-12);
            EXPECT_GT(soft_s_new, -1e-12);
            EXPECT_GT(soft_dual_new, -1e-12);
        }
    }
}

TEST(PropertyRegressionTest, FractionToBoundaryKeepsMixedL1L2SoftDualInterior)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    constexpr int MaxN = 0;
    constexpr int N = 0;
    constexpr double tau = 0.95;

    std::array<Knot, MaxN + 1> traj;
    traj[0].set_zero();
    traj[0].s(0) = 1.0;
    traj[0].lam(0) = 31.0;
    traj[0].soft_s(0) = 10.0;
    traj[0].ds(0) = 0.0;
    traj[0].dlam(0) = 0.0;
    traj[0].dsoft_s(0) = -1.0;
    MixedL1L2SoftConstraintModel::update_soft_constraint_weights(traj[0]);

    const double alpha
        = fraction_to_boundary_rule<decltype(traj), MixedL1L2SoftConstraintModel>(traj, N, tau);

    const double soft_dual_new = MixedL1L2SoftConstraintModel::l1_weight
        + MixedL1L2SoftConstraintModel::l2_weight * (traj[0].soft_s(0) + alpha * traj[0].dsoft_s(0))
        - (traj[0].lam(0) + alpha * traj[0].dlam(0));

    EXPECT_LT(alpha, 0.34) << "The mixed soft dual w1 + w2*soft_s - lam becomes nonpositive "
                              "before soft_s itself reaches zero.";
    EXPECT_GT(soft_dual_new, -1e-12);
}

TEST(PropertyRegressionTest, AutomaticConstraintRowScalingBoundsRandomRows)
{
    using Knot = KnotPoint<double, ScalingPropertyModel::NX, ScalingPropertyModel::NU,
        ScalingPropertyModel::NC, ScalingPropertyModel::NP>;

    SolverConfig config;
    config.constraint_row_scale_min = 1e-4;
    config.constraint_row_scale_max = 1e4;

    std::mt19937 rng(0x5CA1E);
    std::uniform_real_distribution<double> value_dist(-1000.0, 1000.0);

    for (int sample = 0; sample < 200; ++sample) {
        Knot kp;
        kp.set_zero();
        for (int row = 0; row < ScalingPropertyModel::NC; ++row) {
            kp.g_unscaled(row) = value_dist(rng);
            for (int col = 0; col < ScalingPropertyModel::NX; ++col) {
                kp.C(row, col) = value_dist(rng);
            }
            for (int col = 0; col < ScalingPropertyModel::NU; ++col) {
                kp.D(row, col) = value_dist(rng);
            }

            const double row_norm = row_inf_norm(kp, row);
            const double scale = detail::compute_auto_constraint_row_scale(kp, config, row);

            EXPECT_TRUE(std::isfinite(scale));
            EXPECT_GE(scale, config.constraint_row_scale_min);
            EXPECT_LE(scale, config.constraint_row_scale_max);
            EXPECT_LE(scale * std::max(1.0, row_norm), 1.0 + 1e-12)
                << "row-inf scaling should down-scale large rows to O(1)";
        }
    }
}

TEST(PropertyRegressionTest, ObjectiveGershgorinScalingBoundsRandomCurvature)
{
    using Knot = KnotPoint<double, ScalingPropertyModel::NX, ScalingPropertyModel::NU, 0,
        ScalingPropertyModel::NP>;

    SolverConfig config;
    config.objective_scale_min = 1e-4;
    config.objective_scale_max = 1.0;

    std::mt19937 rng(0x9E125);
    std::uniform_real_distribution<double> value_dist(-1000.0, 1000.0);

    for (int sample = 0; sample < 200; ++sample) {
        Knot kp;
        kp.set_zero();
        for (int row = 0; row < ScalingPropertyModel::NX; ++row) {
            for (int col = 0; col < ScalingPropertyModel::NX; ++col) {
                kp.Q(row, col) = value_dist(rng);
            }
        }
        for (int row = 0; row < ScalingPropertyModel::NU; ++row) {
            for (int col = 0; col < ScalingPropertyModel::NU; ++col) {
                kp.R(row, col) = value_dist(rng);
            }
            for (int col = 0; col < ScalingPropertyModel::NX; ++col) {
                kp.H(row, col) = value_dist(rng);
            }
        }

        const double row_bound = objective_hessian_row_bound(kp);
        const double scale = detail::compute_hessian_gershgorin_objective_scale(kp, config);

        EXPECT_TRUE(std::isfinite(scale));
        EXPECT_GE(scale, config.objective_scale_min);
        EXPECT_LE(scale, config.objective_scale_max);
        EXPECT_LE(scale * std::max(1.0, row_bound), 1.0 + 1e-12)
            << "Gershgorin objective scaling should bound the active Hessian row sum";
    }
}
