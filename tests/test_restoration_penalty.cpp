// Restoration penalty rho contract tests.
//
// These tests pin the Tier 4.4 contract for the quadratic-penalty
// feasibility restoration:
//   - Default config keeps RestorationPenaltyMode::FIXED with rho_init =
//     1000.0, preserving the legacy hardcoded behaviour.
//   - Validation rejects non-positive rho_init / rho_min / rho_max,
//     non-positive violation floor, rho_max < rho_min, and unknown enum
//     values.
//   - SolverInfo::reset clears the restoration_rho_* counters.

#include "minisolver/core/config_validation.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

using namespace minisolver;

TEST(RestorationPenaltyTest, DefaultsPreserveLegacyBehaviour)
{
    SolverConfig config;
    EXPECT_EQ(config.restoration_penalty_mode, SolverConfig::RestorationPenaltyMode::FIXED);
    EXPECT_DOUBLE_EQ(config.restoration_rho_init, 1000.0);
    EXPECT_GT(config.restoration_rho_min, 0.0);
    EXPECT_GT(config.restoration_rho_max, config.restoration_rho_min);
    EXPECT_GT(config.restoration_rho_violation_floor, 0.0);
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::OK);
}

TEST(RestorationPenaltyTest, ValidationRejectsNonPositiveRho)
{
    SolverConfig config;

    config.restoration_rho_init = 0.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
    config.restoration_rho_init = 1000.0;

    config.restoration_rho_min = -1.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
    config.restoration_rho_min = 1.0;

    config.restoration_rho_max = 0.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
    config.restoration_rho_max = 1e6;

    config.restoration_rho_violation_floor = 0.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
    config.restoration_rho_violation_floor = 1e-6;
}

TEST(RestorationPenaltyTest, ValidationRejectsRhoMaxBelowRhoMin)
{
    SolverConfig config;
    config.restoration_rho_min = 100.0;
    config.restoration_rho_max = 1.0;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
}

TEST(RestorationPenaltyTest, ValidationRejectsUnknownPenaltyMode)
{
    SolverConfig config;
    auto& mode = config.restoration_penalty_mode;
    std::int32_t raw = 0;
    std::memcpy(&raw, &mode, sizeof(raw));
    raw = 99;
    std::memcpy(&mode, &raw, sizeof(mode));
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::InvalidArgument);
}

TEST(RestorationPenaltyTest, ValidationAcceptsViolationAdaptiveMode)
{
    SolverConfig config;
    config.restoration_penalty_mode = SolverConfig::RestorationPenaltyMode::VIOLATION_ADAPTIVE;
    EXPECT_EQ(detail::validate_solver_config(config), ApiStatus::OK);
}

TEST(RestorationPenaltyTest, InfoResetClearsRestorationRhoCounters)
{
    SolverInfo info;
    info.restoration_rho_min_used = 12.0;
    info.restoration_rho_max_used = 3400.0;
    info.restoration_rho_adaptive_steps = 5;
    info.reset();
    EXPECT_DOUBLE_EQ(info.restoration_rho_min_used, 0.0);
    EXPECT_DOUBLE_EQ(info.restoration_rho_max_used, 0.0);
    EXPECT_EQ(info.restoration_rho_adaptive_steps, 0);
}
