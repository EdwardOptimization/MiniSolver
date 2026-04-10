#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "asset_regression_reference_data.h"
#include "minisolver/solver/solver.h"
#include "doubleintegrator3dregressionmodel.h"
#include "kinematicbicycleregressionmodel.h"

using namespace minisolver;

namespace {

bool is_success(SolverStatus status) {
    return status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE;
}

double wrap_angle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

template <typename SolverT>
double total_cost(const SolverT& solver, int horizon) {
    double cost = 0.0;
    for (int k = 0; k <= horizon; ++k) {
        cost += solver.get_stage_cost(k);
    }
    return cost;
}

template <size_t N>
void expect_state_near(const std::array<double, N>& expected,
                       const std::array<double, N>& actual,
                       double tol) {
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "state index " << i;
    }
}

void shift_bicycle_guess(MiniSolver<KinematicBicycleRegressionModel, 24>& solver, int horizon) {
    for (int k = 0; k <= horizon; ++k) {
        int src_k = std::min(k + 1, horizon);
        for (int i = 0; i < KinematicBicycleRegressionModel::NX; ++i) {
            solver.set_state_guess(k, i, solver.get_state(src_k, i));
        }
    }
    for (int k = 0; k < horizon; ++k) {
        int src_k = std::min(k + 1, horizon - 1);
        for (int i = 0; i < KinematicBicycleRegressionModel::NU; ++i) {
            solver.set_control_guess(k, i, solver.get_control(src_k, i));
        }
    }
}

void shift_double_integrator_guess(MiniSolver<DoubleIntegrator3DRegressionModel, 24>& solver, int horizon) {
    for (int k = 0; k <= horizon; ++k) {
        int src_k = std::min(k + 1, horizon);
        for (int i = 0; i < DoubleIntegrator3DRegressionModel::NX; ++i) {
            solver.set_state_guess(k, i, solver.get_state(src_k, i));
        }
    }
    for (int k = 0; k < horizon; ++k) {
        int src_k = std::min(k + 1, horizon - 1);
        for (int i = 0; i < DoubleIntegrator3DRegressionModel::NU; ++i) {
            solver.set_control_guess(k, i, solver.get_control(src_k, i));
        }
    }
}

void setup_straight_track(MiniSolver<KinematicBicycleRegressionModel, 24>& solver,
                          int horizon,
                          double dt,
                          double stage_offset) {
    constexpr double kRefSpeed = 2.0;
    constexpr double kTrackHalfWidth = 1.0;
    for (int k = 0; k <= horizon; ++k) {
        double t = (stage_offset + k) * dt;
        solver.set_parameter(k, "x_ref", kRefSpeed * t);
        solver.set_parameter(k, "y_ref", 0.0);
        solver.set_parameter(k, "psi_ref", 0.0);
        solver.set_parameter(k, "v_ref", kRefSpeed);
        solver.set_parameter(k, "n_x", 0.0);
        solver.set_parameter(k, "n_y", 1.0);
        solver.set_parameter(k, "w_left", kTrackHalfWidth);
        solver.set_parameter(k, "w_right", kTrackHalfWidth);
    }
}

void setup_curved_track(MiniSolver<KinematicBicycleRegressionModel, 24>& solver,
                        int horizon,
                        double dt,
                        double stage_offset) {
    constexpr double kRefSpeed = 1.8;
    constexpr double kAmp = 0.28;
    constexpr double kFreq = 0.55;
    for (int k = 0; k <= horizon; ++k) {
        double t = (stage_offset + k) * dt;
        double x_ref = kRefSpeed * t;
        double y_ref = kAmp * std::sin(kFreq * x_ref);
        double dy_dx = kAmp * kFreq * std::cos(kFreq * x_ref);
        double psi_ref = std::atan(dy_dx);
        double n_x = -std::sin(psi_ref);
        double n_y = std::cos(psi_ref);
        double width = 0.55 - 0.08 * std::sin(0.35 * x_ref);

        solver.set_parameter(k, "x_ref", x_ref);
        solver.set_parameter(k, "y_ref", y_ref);
        solver.set_parameter(k, "psi_ref", psi_ref);
        solver.set_parameter(k, "v_ref", kRefSpeed);
        solver.set_parameter(k, "n_x", n_x);
        solver.set_parameter(k, "n_y", n_y);
        solver.set_parameter(k, "w_left", width);
        solver.set_parameter(k, "w_right", width);
    }
}

void setup_3d_reference(MiniSolver<DoubleIntegrator3DRegressionModel, 24>& solver,
                        int horizon,
                        double dt,
                        double phase = 0.0,
                        double amplitude_scale = 1.0) {
    for (int k = 0; k <= horizon; ++k) {
        double t = phase + k * dt;
        double x_ref = 0.5 + 0.35 * t;
        double y_ref = amplitude_scale * 0.15 * std::sin(0.9 * t);
        double z_ref = 1.0 + amplitude_scale * 0.12 * std::cos(0.7 * t);
        double vx_ref = 0.35;
        double vy_ref = amplitude_scale * 0.15 * 0.9 * std::cos(0.9 * t);
        double vz_ref = -amplitude_scale * 0.12 * 0.7 * std::sin(0.7 * t);

        solver.set_parameter(k, "x_ref", x_ref);
        solver.set_parameter(k, "y_ref", y_ref);
        solver.set_parameter(k, "z_ref", z_ref);
        solver.set_parameter(k, "vx_ref", vx_ref);
        solver.set_parameter(k, "vy_ref", vy_ref);
        solver.set_parameter(k, "vz_ref", vz_ref);
    }
}

}  // namespace

TEST(AssetRegressionTest, KinematicBicycleStraightTrackRecovery) {
    constexpr int N = 12;
    constexpr double dt = 0.1;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 60;
    config.tol_con = 1e-4;
    config.mu_final = 1e-6;
    config.integrator = IntegratorType::RK2_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.line_search_type = LineSearchType::FILTER;

    MiniSolver<KinematicBicycleRegressionModel, 24> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(dt);
    setup_straight_track(solver, N, dt, 0.0);

    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.35);
    solver.set_initial_state("psi", 0.12);
    solver.set_initial_state("v", 0.8);
    solver.set_initial_state("delta", 0.0);
    solver.rollout_dynamics();

    SolverStatus status = solver.solve();
    ASSERT_TRUE(is_success(status));

    double y_final = solver.get_state(N, solver.get_state_idx("y"));
    double psi_final = solver.get_state(N, solver.get_state_idx("psi"));
    double v_final = solver.get_state(N, solver.get_state_idx("v"));
    double cost = total_cost(solver, N);

    std::array<double, 5> actual_terminal = {
        solver.get_state(N, solver.get_state_idx("x")),
        solver.get_state(N, solver.get_state_idx("y")),
        solver.get_state(N, solver.get_state_idx("psi")),
        solver.get_state(N, solver.get_state_idx("v")),
        solver.get_state(N, solver.get_state_idx("delta")),
    };
    std::array<double, 2> actual_first_control = {
        solver.get_control(0, solver.get_control_idx("a")),
        solver.get_control(0, solver.get_control_idx("delta_rate")),
    };

    EXPECT_LT(std::abs(y_final), 0.12);
    EXPECT_LT(std::abs(wrap_angle(psi_final)), 0.12);
    EXPECT_GT(v_final, 1.5);
    EXPECT_LE(cost, testdata::kKinematicBicycleStraightReference.objective * 1.02);
    expect_state_near(testdata::kKinematicBicycleStraightReference.terminal_state, actual_terminal, 5e-3);
    EXPECT_NEAR(actual_first_control[0], testdata::kKinematicBicycleStraightReference.first_control[0], 1e-3);
    EXPECT_NEAR(actual_first_control[1], testdata::kKinematicBicycleStraightReference.first_control[1], 1e-3);

    for (int k = 0; k <= N; ++k) {
        double x_ref = solver.get_parameter(k, "x_ref");
        double y_ref = solver.get_parameter(k, "y_ref");
        double n_x = solver.get_parameter(k, "n_x");
        double n_y = solver.get_parameter(k, "n_y");
        double w_left = solver.get_parameter(k, "w_left");
        double w_right = solver.get_parameter(k, "w_right");
        double x = solver.get_state(k, solver.get_state_idx("x"));
        double y = solver.get_state(k, solver.get_state_idx("y"));
        double lateral = n_x * (x - x_ref) + n_y * (y - y_ref);
        EXPECT_LE(lateral, w_left + 1e-3);
        EXPECT_GE(lateral, -w_right - 1e-3);
    }
}

TEST(AssetRegressionTest, DoubleIntegrator3DReferenceTracking) {
    constexpr int N = 15;
    constexpr double dt = 0.1;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 50;
    config.tol_con = 1e-5;
    config.mu_final = 1e-6;
    config.integrator = IntegratorType::RK2_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.line_search_type = LineSearchType::FILTER;

    MiniSolver<DoubleIntegrator3DRegressionModel, 24> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(dt);
    setup_3d_reference(solver, N, dt);

    solver.set_initial_state("x", -0.35);
    solver.set_initial_state("y", 0.25);
    solver.set_initial_state("z", 0.75);
    solver.set_initial_state("vx", 0.0);
    solver.set_initial_state("vy", 0.0);
    solver.set_initial_state("vz", 0.0);
    solver.rollout_dynamics();

    double initial_error =
        std::hypot(solver.get_state(0, solver.get_state_idx("x")) - solver.get_parameter(0, "x_ref"),
                   solver.get_state(0, solver.get_state_idx("y")) - solver.get_parameter(0, "y_ref"));
    initial_error = std::hypot(initial_error,
                               solver.get_state(0, solver.get_state_idx("z")) - solver.get_parameter(0, "z_ref"));

    SolverStatus status = solver.solve();
    ASSERT_TRUE(is_success(status));

    double x_final = solver.get_state(N, solver.get_state_idx("x"));
    double y_final = solver.get_state(N, solver.get_state_idx("y"));
    double z_final = solver.get_state(N, solver.get_state_idx("z"));
    double vx_final = solver.get_state(N, solver.get_state_idx("vx"));
    double vy_final = solver.get_state(N, solver.get_state_idx("vy"));
    double vz_final = solver.get_state(N, solver.get_state_idx("vz"));

    double terminal_error =
        std::hypot(x_final - solver.get_parameter(N, "x_ref"), y_final - solver.get_parameter(N, "y_ref"));
    terminal_error = std::hypot(terminal_error, z_final - solver.get_parameter(N, "z_ref"));

    double terminal_velocity_error =
        std::hypot(vx_final - solver.get_parameter(N, "vx_ref"), vy_final - solver.get_parameter(N, "vy_ref"));
    terminal_velocity_error =
        std::hypot(terminal_velocity_error, vz_final - solver.get_parameter(N, "vz_ref"));
    double cost = total_cost(solver, N);

    std::array<double, 6> actual_terminal = {x_final, y_final, z_final, vx_final, vy_final, vz_final};
    std::array<double, 3> actual_first_control = {
        solver.get_control(0, solver.get_control_idx("ax")),
        solver.get_control(0, solver.get_control_idx("ay")),
        solver.get_control(0, solver.get_control_idx("az")),
    };

    EXPECT_LT(terminal_error, initial_error * 0.35);
    EXPECT_LT(terminal_velocity_error, 0.5);
    EXPECT_NEAR(cost, testdata::kDoubleIntegrator3DTrackingReference.objective, 5e-3);
    expect_state_near(testdata::kDoubleIntegrator3DTrackingReference.terminal_state, actual_terminal, 5e-4);
    EXPECT_NEAR(actual_first_control[0], testdata::kDoubleIntegrator3DTrackingReference.first_control[0], 5e-3);
    EXPECT_NEAR(actual_first_control[1], testdata::kDoubleIntegrator3DTrackingReference.first_control[1], 5e-3);
    EXPECT_NEAR(actual_first_control[2], testdata::kDoubleIntegrator3DTrackingReference.first_control[2], 5e-3);

    for (int k = 0; k < N; ++k) {
        EXPECT_LE(std::abs(solver.get_control(k, solver.get_control_idx("ax"))), 12.0 + 1e-4);
        EXPECT_LE(std::abs(solver.get_control(k, solver.get_control_idx("ay"))), 12.0 + 1e-4);
        EXPECT_LE(std::abs(solver.get_control(k, solver.get_control_idx("az"))), 12.0 + 1e-4);
    }
}

TEST(AssetRegressionTest, KinematicBicycleCurvedTrackClosedLoop) {
    constexpr int N = 14;
    constexpr int MPC_STEPS = 6;
    constexpr double dt = 0.1;
    constexpr double wheelbase = 0.33;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 70;
    config.tol_con = 2e-4;
    config.mu_final = 1e-6;
    config.integrator = IntegratorType::RK2_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.line_search_type = LineSearchType::FILTER;

    MiniSolver<KinematicBicycleRegressionModel, 24> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(dt);
    setup_curved_track(solver, N, dt, 0.0);

    double sim_x = 0.0;
    double sim_y = 0.24;
    double sim_psi = 0.18;
    double sim_v = 1.0;
    double sim_delta = 0.02;

    solver.set_initial_state("x", sim_x);
    solver.set_initial_state("y", sim_y);
    solver.set_initial_state("psi", sim_psi);
    solver.set_initial_state("v", sim_v);
    solver.set_initial_state("delta", sim_delta);
    solver.rollout_dynamics();

    int success_count = 0;
    for (int step = 0; step < MPC_STEPS; ++step) {
        setup_curved_track(solver, N, dt, static_cast<double>(step));

        SolverStatus status = solver.solve();
        if (is_success(status)) {
            ++success_count;
        }

        ASSERT_TRUE(is_success(status));

        double accel = solver.get_control(0, solver.get_control_idx("a"));
        double delta_rate = solver.get_control(0, solver.get_control_idx("delta_rate"));
        sim_x += sim_v * std::cos(sim_psi) * dt;
        sim_y += sim_v * std::sin(sim_psi) * dt;
        sim_psi += sim_v * std::tan(sim_delta) / wheelbase * dt;
        sim_v += accel * dt;
        sim_delta += delta_rate * dt;
        sim_delta = std::clamp(sim_delta, -0.5, 0.5);

        shift_bicycle_guess(solver, N);
        solver.set_initial_state("x", sim_x);
        solver.set_initial_state("y", sim_y);
        solver.set_initial_state("psi", sim_psi);
        solver.set_initial_state("v", sim_v);
        solver.set_initial_state("delta", sim_delta);
        solver.reset(ResetOption::ALG_STATE);
    }

    EXPECT_GE(success_count, MPC_STEPS);
    EXPECT_LT(std::abs(sim_y), 0.35);
    EXPECT_LT(std::abs(wrap_angle(sim_psi)), 0.30);
    EXPECT_GT(sim_v, 1.2);
    expect_state_near(
        testdata::kKinematicBicycleCurvedClosedLoopFinalState,
        std::array<double, 5>{sim_x, sim_y, sim_psi, sim_v, sim_delta},
        5e-3);
}

TEST(AssetRegressionTest, DoubleIntegrator3DShiftedResolveMatchesReference) {
    constexpr int N = 16;
    constexpr double dt = 0.1;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 60;
    config.tol_con = 1e-5;
    config.mu_final = 1e-6;
    config.integrator = IntegratorType::RK2_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.line_search_type = LineSearchType::FILTER;

    MiniSolver<DoubleIntegrator3DRegressionModel, 24> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(dt);
    setup_3d_reference(solver, N, dt, 0.0, 1.0);
    solver.set_initial_state("x", -0.30);
    solver.set_initial_state("y", 0.22);
    solver.set_initial_state("z", 0.72);
    solver.set_initial_state("vx", 0.0);
    solver.set_initial_state("vy", 0.0);
    solver.set_initial_state("vz", 0.0);
    solver.rollout_dynamics();

    SolverStatus base_status = solver.solve();
    ASSERT_TRUE(is_success(base_status));

    shift_double_integrator_guess(solver, N);
    setup_3d_reference(solver, N, dt, 0.15, 1.15);
    solver.set_initial_state("x", -0.20);
    solver.set_initial_state("y", 0.28);
    solver.set_initial_state("z", 0.80);
    solver.set_initial_state("vx", 0.04);
    solver.set_initial_state("vy", -0.01);
    solver.set_initial_state("vz", 0.02);
    solver.reset(ResetOption::ALG_STATE);

    SolverStatus shifted_status = solver.solve();
    ASSERT_TRUE(is_success(shifted_status));

    std::array<double, 6> actual_terminal = {
        solver.get_state(N, solver.get_state_idx("x")),
        solver.get_state(N, solver.get_state_idx("y")),
        solver.get_state(N, solver.get_state_idx("z")),
        solver.get_state(N, solver.get_state_idx("vx")),
        solver.get_state(N, solver.get_state_idx("vy")),
        solver.get_state(N, solver.get_state_idx("vz")),
    };
    std::array<double, 3> actual_first_control = {
        solver.get_control(0, solver.get_control_idx("ax")),
        solver.get_control(0, solver.get_control_idx("ay")),
        solver.get_control(0, solver.get_control_idx("az")),
    };
    double cost = total_cost(solver, N);

    EXPECT_NEAR(cost, testdata::kDoubleIntegrator3DShiftedReference.objective, 5e-3);
    expect_state_near(testdata::kDoubleIntegrator3DShiftedReference.terminal_state, actual_terminal, 5e-4);
    EXPECT_NEAR(actual_first_control[0], testdata::kDoubleIntegrator3DShiftedReference.first_control[0], 5e-3);
    EXPECT_NEAR(actual_first_control[1], testdata::kDoubleIntegrator3DShiftedReference.first_control[1], 5e-3);
    EXPECT_NEAR(actual_first_control[2], testdata::kDoubleIntegrator3DShiftedReference.first_control[2], 5e-3);
}
