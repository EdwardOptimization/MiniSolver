#include "minisolver/algorithms/initialization.h"
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
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };
};

struct L1SoftConstraintModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 2.0 };
    static constexpr std::array<int, NC> constraint_types = { 1 };
};

struct L2SoftConstraintModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 2.0 };
    static constexpr std::array<int, NC> constraint_types = { 2 };
};

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

TEST(PropertyRegressionTest, L1SoftInitializationStaysInsideDualBox)
{
    using Knot = KnotPoint<double, 1, 1, 1, 0>;
    constexpr double w = L1SoftConstraintModel::constraint_weights[0];

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
    constexpr double w = L2SoftConstraintModel::constraint_weights[0];

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
    constexpr double w = L1SoftConstraintModel::constraint_weights[0];

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
