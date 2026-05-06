#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/core/config_validation.h"
#include "minisolver/core/solver_config_profiles.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/solver/solver.h"
#include <gtest/gtest.h>

using namespace minisolver;

namespace {

void configure_simple_car(MiniSolver<CarModel, 20>& solver, int N)
{
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("v", 0.0);
    for (int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "v_ref", 1.0);
        solver.set_parameter(k, "x_ref", 0.5 * k * 0.1);
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
}

} // namespace

TEST(SolverConfigProfilesTest, AllProfilesPassConfigValidation)
{
    EXPECT_EQ(detail::validate_solver_config(make_reference_config()), ApiStatus::OK);
    EXPECT_EQ(detail::validate_solver_config(make_default_config()), ApiStatus::OK);
    EXPECT_EQ(detail::validate_solver_config(make_speed_config()), ApiStatus::OK);
    EXPECT_EQ(detail::validate_solver_config(make_robust_config()), ApiStatus::OK);
}

TEST(SolverConfigProfilesTest, ProfileFieldsReflectIntent)
{
    const SolverConfig reference = make_reference_config();
    EXPECT_EQ(reference.barrier_strategy, BarrierStrategy::MONOTONE);
    EXPECT_EQ(reference.line_search_type, LineSearchType::MERIT);
    EXPECT_FALSE(reference.enable_soc);
    EXPECT_FALSE(reference.enable_feasibility_restoration);
    EXPECT_EQ(reference.direction_refinement, DirectionRefinementMode::NONE);

    const SolverConfig speed = make_speed_config();
    EXPECT_EQ(speed.termination_profile, TerminationProfile::ACCEPTABLE_NMPC)
        << "Speed profile must allow ACCEPTABLE_NMPC fallback";
    EXPECT_LE(speed.max_iters, 30) << "Speed profile must run a tight iteration budget";
    EXPECT_FALSE(speed.enable_soc);
    EXPECT_FALSE(speed.enable_feasibility_restoration);

    const SolverConfig robust = make_robust_config();
    EXPECT_EQ(robust.barrier_strategy, BarrierStrategy::MEHROTRA);
    EXPECT_EQ(robust.line_search_type, LineSearchType::FILTER);
    EXPECT_TRUE(robust.enable_soc);
    EXPECT_TRUE(robust.enable_feasibility_restoration);
    EXPECT_EQ(robust.problem_scaling, ProblemScalingMethod::RUIZ_EQUILIBRATION);
    EXPECT_EQ(robust.direction_refinement, DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT);
    EXPECT_LE(robust.mu_final, 1e-7);
    EXPECT_LE(robust.tol_con, 1e-6);

    // Profiles are not all the same: there must be observable differences.
    EXPECT_NE(reference.barrier_strategy, robust.barrier_strategy);
    EXPECT_NE(speed.termination_profile, robust.termination_profile);
    EXPECT_LT(speed.max_iters, robust.max_iters);
}

TEST(SolverConfigProfilesTest, AllProfilesSolveCarModelToAcceptableQuality)
{
    constexpr int N = 10;
    const SolverConfig profiles[] = { make_reference_config(), make_default_config(),
        make_speed_config(), make_robust_config() };
    const char* names[] = { "reference", "default", "speed", "robust" };

    for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); ++i) {
        SolverConfig cfg = profiles[i];
        MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, cfg);
        configure_simple_car(solver, N);

        const SolverStatus status = solver.solve();
        EXPECT_TRUE(status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE)
            << "Profile " << names[i] << " must reach OPTIMAL/FEASIBLE on the basic CarModel; "
            << "got " << status_to_string(status);
    }
}
