#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "generated/dcoltwocarsmodel.h"
#include "generated/innerdcolnorm2model.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kLaneBump = 2.4;
constexpr int kPoseDim = 6;
constexpr int kOuterHorizon = 50;
constexpr double kCarLength = 4.5;
constexpr double kCarWidth = 1.9;
constexpr double kDcolMinDistance = 1.0;

enum StateIndex {
    X1 = 0,
    Y1 = 1,
    THETA1 = 2,
    V1 = 3,
    X2 = 4,
    Y2 = 5,
    THETA2 = 6,
    V2 = 7,
};

enum ControlIndex {
    A1 = 0,
    OMEGA1 = 1,
    A2 = 2,
    OMEGA2 = 3,
};

enum ParamIndex {
    X1_REF = 0,
    Y1_REF = 1,
    THETA1_REF = 2,
    V1_REF = 3,
    X2_REF = 4,
    Y2_REF = 5,
    THETA2_REF = 6,
    V2_REF = 7,
    X1_LIN = 8,
    Y1_LIN = 9,
    THETA1_LIN = 10,
    X2_LIN = 11,
    Y2_LIN = 12,
    THETA2_LIN = 13,
    DCOL_ALPHA = 14,
    DCOL_GX1 = 15,
    DCOL_GY1 = 16,
    DCOL_GTHETA1 = 17,
    DCOL_GX2 = 18,
    DCOL_GY2 = 19,
    DCOL_GTHETA2 = 20,
    DCOL_H00 = 21,
    DCOL_H01 = 22,
    DCOL_H02 = 23,
    DCOL_H03 = 24,
    DCOL_H04 = 25,
    DCOL_H05 = 26,
    DCOL_H11 = 27,
    DCOL_H12 = 28,
    DCOL_H13 = 29,
    DCOL_H14 = 30,
    DCOL_H15 = 31,
    DCOL_H22 = 32,
    DCOL_H23 = 33,
    DCOL_H24 = 34,
    DCOL_H25 = 35,
    DCOL_H33 = 36,
    DCOL_H34 = 37,
    DCOL_H35 = 38,
    DCOL_H44 = 39,
    DCOL_H45 = 40,
    DCOL_H55 = 41,
};

using DcolSolver = MiniSolver<DcolTwoCarsModel, kOuterHorizon>;
using InnerDcolSolver = MiniSolver<InnerDcolNorm2Model, 1>;

enum class DcolOracleMode {
    ANALYTIC_SUPPORT,
    INNER_MINISOLVER,
};

struct Pose2 {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
};

struct LocalQuadratic {
    double alpha = 0.0;
    std::array<double, kPoseDim> g {};
    std::array<std::array<double, kPoseDim>, kPoseDim> H {};
};

enum InnerParamIndex {
    INNER_X1 = 0,
    INNER_Y1 = 1,
    INNER_THETA1 = 2,
    INNER_X2 = 3,
    INNER_Y2 = 4,
    INNER_THETA2 = 5,
};

enum InnerControlIndex {
    INNER_P1X = 0,
    INNER_P1Y = 1,
    INNER_P2X = 2,
    INNER_P2Y = 3,
    INNER_ALPHA = 4,
};

struct DcolCallbackContext {
    DcolOracleMode oracle_mode = DcolOracleMode::ANALYTIC_SUPPORT;
    double dt = 0.15;
    double car1_v_ref = 3.0;
    double car2_v_ref = 3.0;
    double car2_x0 = 17.35;
    int calls = 0;
    int analytic_evals = 0;
    int inner_solves = 0;
    int inner_failures = 0;
    int inner_total_iterations = 0;
    int inner_max_iterations = 0;
    std::array<double, kOuterHorizon + 1> last_alpha {};
    std::array<bool, kOuterHorizon + 1> inner_seeded {};
    std::vector<std::unique_ptr<InnerDcolSolver>> inner_solvers;
};

SolverConfig make_inner_dcol_config()
{
    SolverConfig config;
    config.max_iters = 60;
    config.default_dt = 1.0;
    config.print_level = PrintLevel::NONE;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.line_search_type = LineSearchType::MERIT;
    config.hessian_approximation = HessianApproximation::EXACT;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.warm_start_barrier = WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP;
    config.warm_start_regularization = WarmStartRegularizationMode::RESET_TO_REG_INIT;
    config.tol_con = 1.0e-7;
    config.tol_dual = 1.0e-3;
    config.tol_mu = 1.0e-7;
    return config;
}

void initialize_inner_solvers(DcolCallbackContext& ctx, int horizon)
{
    ctx.inner_solvers.clear();
    ctx.inner_solvers.reserve(static_cast<size_t>(horizon + 1));
    const SolverConfig inner_config = make_inner_dcol_config();
    for (int k = 0; k <= horizon; ++k) {
        auto solver = std::make_unique<InnerDcolSolver>(1, Backend::CPU_SERIAL, inner_config);
        solver->set_dt(1.0);
        solver->set_initial_state({ 0.0 });
        solver->set_control_guess(0, INNER_ALPHA, 1.0);
        ctx.inner_solvers.push_back(std::move(solver));
        ctx.last_alpha[static_cast<size_t>(k)] = 1.0;
        ctx.inner_seeded[static_cast<size_t>(k)] = false;
    }
}

double passing_bump(const DcolCallbackContext& ctx, int horizon, int stage)
{
    const double horizon_time = ctx.dt * static_cast<double>(horizon);
    const double t = ctx.dt * static_cast<double>(stage);
    return kLaneBump * std::sin(kPi * t / horizon_time);
}

double passing_bump_rate(const DcolCallbackContext& ctx, int horizon, int stage)
{
    const double horizon_time = ctx.dt * static_cast<double>(horizon);
    const double t = ctx.dt * static_cast<double>(stage);
    return kLaneBump * kPi * std::cos(kPi * t / horizon_time) / horizon_time;
}

double unwrap_angle_near(double angle, double reference)
{
    while (angle - reference > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle - reference < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}

double car1_reference_heading(const DcolCallbackContext& ctx, int horizon, int stage)
{
    if (stage == 0) {
        return 0.0;
    }
    const double bump_rate_ref = passing_bump_rate(ctx, horizon, stage);
    return std::atan2(-bump_rate_ref, ctx.car1_v_ref);
}

double car2_reference_heading(const DcolCallbackContext& ctx, int horizon, int stage)
{
    if (stage == 0) {
        return kPi;
    }
    const double bump_rate_ref = passing_bump_rate(ctx, horizon, stage);
    return unwrap_angle_near(std::atan2(bump_rate_ref, -ctx.car2_v_ref), kPi);
}

Pose2 car1_pose(const DcolSolver& solver, int stage)
{
    return Pose2 {
        solver.get_state(stage, X1),
        solver.get_state(stage, Y1),
        solver.get_state(stage, THETA1),
    };
}

Pose2 car2_pose(const DcolSolver& solver, int stage)
{
    return Pose2 {
        solver.get_state(stage, X2),
        solver.get_state(stage, Y2),
        solver.get_state(stage, THETA2),
    };
}

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

Vec2 car_axis_x(const Pose2& car)
{
    return Vec2 { std::cos(car.theta), std::sin(car.theta) };
}

Vec2 car_axis_y(const Pose2& car)
{
    return Vec2 { -std::sin(car.theta), std::cos(car.theta) };
}

double dot(const Vec2& a, const Vec2& b)
{
    return a.x * b.x + a.y * b.y;
}

double sign_or_zero(double value)
{
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

double rectangle_support(const Pose2& car, const Vec2& n)
{
    const Vec2 ex = car_axis_x(car);
    const Vec2 ey = car_axis_y(car);
    return 0.5 * kCarLength * std::abs(dot(n, ex)) + 0.5 * kCarWidth * std::abs(dot(n, ey));
}

double rectangle_support_theta_derivative(const Pose2& car, const Vec2& n)
{
    const Vec2 ex = car_axis_x(car);
    const Vec2 ey = car_axis_y(car);
    const double nx = dot(n, ex);
    const double ny = dot(n, ey);
    return 0.5 * kCarLength * sign_or_zero(nx) * ny - 0.5 * kCarWidth * sign_or_zero(ny) * nx;
}

double cross(const Vec2& a, const Vec2& b)
{
    return a.x * b.y - a.y * b.x;
}

double norm(const Vec2& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

double normalize_angle(double angle)
{
    constexpr double kTwoPi = 2.0 * kPi;
    double result = std::fmod(angle, kTwoPi);
    if (result < 0.0) {
        result += kTwoPi;
    }
    return result;
}

double dcol_alpha_for_direction(const Pose2& car1, const Pose2& car2, double angle)
{
    const Vec2 n { std::cos(angle), std::sin(angle) };
    const Vec2 delta { car2.x - car1.x, car2.y - car1.y };
    const double numerator = dot(n, delta);
    const double denominator
        = rectangle_support(car1, n) + rectangle_support(car2, n) + kDcolMinDistance;
    return numerator / denominator;
}

Vec2 support_linear_part_for_interval(const Pose2& car1, const Pose2& car2, double angle)
{
    const Vec2 n { std::cos(angle), std::sin(angle) };
    Vec2 a {};
    auto add_axis = [&](const Vec2& axis, double half_extent) {
        const double sigma = sign_or_zero(dot(n, axis));
        a.x += half_extent * sigma * axis.x;
        a.y += half_extent * sigma * axis.y;
    };

    add_axis(car_axis_x(car1), 0.5 * kCarLength);
    add_axis(car_axis_y(car1), 0.5 * kCarWidth);
    add_axis(car_axis_x(car2), 0.5 * kCarLength);
    add_axis(car_axis_y(car2), 0.5 * kCarWidth);
    return a;
}

double maximize_dcol_angle(const Pose2& car1, const Pose2& car2)
{
    constexpr double kTwoPi = 2.0 * kPi;
    constexpr double kAngleTol = 1.0e-12;
    std::array<double, 16> breakpoints {};
    int breakpoint_count = 0;

    auto add_breakpoint = [&](double angle) {
        angle = normalize_angle(angle);
        for (int i = 0; i < breakpoint_count; ++i) {
            const double diff = std::abs(breakpoints[static_cast<size_t>(i)] - angle);
            if (diff < 1.0e-10 || std::abs(diff - kTwoPi) < 1.0e-10) {
                return;
            }
        }
        breakpoints[static_cast<size_t>(breakpoint_count++)] = angle;
    };

    auto add_car_breakpoints = [&](const Pose2& car) {
        // Non-smooth support changes happen when the search normal is
        // perpendicular to either rectangle axis. Evaluate these angles
        // explicitly instead of relying on a smooth line search across them.
        add_breakpoint(car.theta);
        add_breakpoint(car.theta + 0.5 * kPi);
        add_breakpoint(car.theta + kPi);
        add_breakpoint(car.theta + 1.5 * kPi);
    };
    add_car_breakpoints(car1);
    add_car_breakpoints(car2);

    std::sort(breakpoints.begin(), breakpoints.begin() + breakpoint_count);

    double best_angle = 0.0;
    double best_value = dcol_alpha_for_direction(car1, car2, best_angle);
    auto consider = [&](double angle) {
        const double value = dcol_alpha_for_direction(car1, car2, angle);
        if (value > best_value) {
            best_value = value;
            best_angle = normalize_angle(angle);
        }
    };

    const Vec2 delta { car2.x - car1.x, car2.y - car1.y };
    const double delta_norm = norm(delta);

    for (int i = 0; i < breakpoint_count; ++i) {
        consider(breakpoints[static_cast<size_t>(i)]);
    }

    for (int i = 0; i < breakpoint_count; ++i) {
        const double lo = breakpoints[static_cast<size_t>(i)];
        const double hi = (i + 1 < breakpoint_count) ? breakpoints[static_cast<size_t>(i + 1)]
                                                     : breakpoints[0] + kTwoPi;
        if (hi <= lo + kAngleTol) {
            continue;
        }

        const double mid = 0.5 * (lo + hi);
        consider(mid);
        if (delta_norm <= 1.0e-12) {
            continue;
        }

        const Vec2 a = support_linear_part_for_interval(car1, car2, mid);
        const double rhs = -cross(a, delta) / (kDcolMinDistance * delta_norm);
        if (rhs < -1.0 - 1.0e-12 || rhs > 1.0 + 1.0e-12) {
            continue;
        }
        const double clamped_rhs = std::max(-1.0, std::min(1.0, rhs));
        const double phi = std::atan2(delta.y, delta.x);
        const double root_a = phi - std::asin(clamped_rhs);
        const double root_b = phi - (kPi - std::asin(clamped_rhs));

        auto consider_root_in_interval = [&](double root) {
            while (root < lo - kAngleTol) {
                root += kTwoPi;
            }
            while (root > hi + kAngleTol) {
                root -= kTwoPi;
            }
            if (root > lo + kAngleTol && root < hi - kAngleTol) {
                consider(root);
            }
        };
        consider_root_in_interval(root_a);
        consider_root_in_interval(root_b);
    }

    return best_angle;
}

LocalQuadratic solve_analytic_dcol(DcolCallbackContext& ctx, const Pose2& car1, const Pose2& car2)
{
    ++ctx.analytic_evals;
    LocalQuadratic out;

    const double angle = maximize_dcol_angle(car1, car2);
    const Vec2 n { std::cos(angle), std::sin(angle) };
    const Vec2 delta { car2.x - car1.x, car2.y - car1.y };
    const double numerator = dot(n, delta);
    const double denominator
        = rectangle_support(car1, n) + rectangle_support(car2, n) + kDcolMinDistance;

    if (denominator <= 0.0) {
        return out;
    }
    out.alpha = std::max(0.0, numerator / denominator);

    // Envelope theorem with the maximizing normal fixed. The normal acts as the
    // active separating certificate, so no inner IPM dual solve is needed.
    const double inv_den = 1.0 / denominator;
    const double alpha_over_den = out.alpha * inv_den;
    out.g[0] = -n.x * inv_den;
    out.g[1] = -n.y * inv_den;
    out.g[2] = -alpha_over_den * rectangle_support_theta_derivative(car1, n);
    out.g[3] = n.x * inv_den;
    out.g[4] = n.y * inv_den;
    out.g[5] = -alpha_over_den * rectangle_support_theta_derivative(car2, n);

    return out;
}

void set_inner_pose(InnerDcolSolver& solver, const Pose2& car1, const Pose2& car2)
{
    solver.set_parameter(0, INNER_X1, car1.x);
    solver.set_parameter(0, INNER_Y1, car1.y);
    solver.set_parameter(0, INNER_THETA1, car1.theta);
    solver.set_parameter(0, INNER_X2, car2.x);
    solver.set_parameter(0, INNER_Y2, car2.y);
    solver.set_parameter(0, INNER_THETA2, car2.theta);
}

void seed_inner_solver(InnerDcolSolver& solver, const Pose2& car1, const Pose2& car2)
{
    const double dx = car1.x - car2.x;
    const double dy = car1.y - car2.y;
    const double alpha = std::max(1.0, std::sqrt(dx * dx + dy * dy) / kDcolMinDistance);
    solver.set_control_guess(0, INNER_P1X, car1.x);
    solver.set_control_guess(0, INNER_P1Y, car1.y);
    solver.set_control_guess(0, INNER_P2X, car2.x);
    solver.set_control_guess(0, INNER_P2Y, car2.y);
    solver.set_control_guess(0, INNER_ALPHA, alpha);
}

void add_rect_dual_derivatives(InnerDcolSolver& solver, int constraint_offset, const Pose2& car,
    double px, double py, int pose_offset, LocalQuadratic& out)
{
    const double c = std::cos(car.theta);
    const double s = std::sin(car.theta);
    const double dx = px - car.x;
    const double dy = py - car.y;
    const double local_x = c * dx + s * dy;
    const double local_y = -s * dx + c * dy;

    const std::array<std::array<double, 3>, 4> grad { {
        { -c, -s, local_y },
        { c, s, -local_y },
        { s, -c, -local_x },
        { -s, c, local_x },
    } };
    const std::array<double, 4> h_x_theta { s, -s, c, -c };
    const std::array<double, 4> h_y_theta { -c, c, s, -s };
    const std::array<double, 4> h_theta_theta {
        -local_x,
        local_x,
        -local_y,
        local_y,
    };

    for (int row = 0; row < 4; ++row) {
        const double lambda = solver.get_dual(0, constraint_offset + row);
        for (int i = 0; i < 3; ++i) {
            out.g[static_cast<size_t>(pose_offset + i)]
                += lambda * grad[static_cast<size_t>(row)][static_cast<size_t>(i)];
        }
        const int ix = pose_offset;
        const int iy = pose_offset + 1;
        const int itheta = pose_offset + 2;
        out.H[static_cast<size_t>(ix)][static_cast<size_t>(itheta)]
            += lambda * h_x_theta[static_cast<size_t>(row)];
        out.H[static_cast<size_t>(itheta)][static_cast<size_t>(ix)]
            = out.H[static_cast<size_t>(ix)][static_cast<size_t>(itheta)];
        out.H[static_cast<size_t>(iy)][static_cast<size_t>(itheta)]
            += lambda * h_y_theta[static_cast<size_t>(row)];
        out.H[static_cast<size_t>(itheta)][static_cast<size_t>(iy)]
            = out.H[static_cast<size_t>(iy)][static_cast<size_t>(itheta)];
        out.H[static_cast<size_t>(itheta)][static_cast<size_t>(itheta)]
            += lambda * h_theta_theta[static_cast<size_t>(row)];
    }
}

LocalQuadratic solve_inner_dcol(
    DcolCallbackContext& ctx, int stage, const Pose2& car1, const Pose2& car2)
{
    auto& solver = *ctx.inner_solvers.at(static_cast<size_t>(stage));
    set_inner_pose(solver, car1, car2);
    if (!ctx.inner_seeded[static_cast<size_t>(stage)]) {
        seed_inner_solver(solver, car1, car2);
        ctx.inner_seeded[static_cast<size_t>(stage)] = true;
    }

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();
    ++ctx.inner_solves;
    ctx.inner_total_iterations += info.iterations;
    ctx.inner_max_iterations = std::max(ctx.inner_max_iterations, info.iterations);
    if (status != SolverStatus::OPTIMAL && status != SolverStatus::FEASIBLE) {
        ++ctx.inner_failures;
    }

    LocalQuadratic out;
    out.alpha = solver.get_control(0, INNER_ALPHA);
    const double p1x = solver.get_control(0, INNER_P1X);
    const double p1y = solver.get_control(0, INNER_P1Y);
    const double p2x = solver.get_control(0, INNER_P2X);
    const double p2y = solver.get_control(0, INNER_P2Y);

    // Envelope theorem: d alpha*/d q = sum_i lambda_i * d g_i/d q.
    // The Hessian block below is the dual-weighted direct q curvature of the
    // rectangle constraints; full KKT sensitivity curvature is intentionally
    // not embedded in this demo.
    add_rect_dual_derivatives(solver, 0, car1, p1x, p1y, 0, out);
    add_rect_dual_derivatives(solver, 4, car2, p2x, p2y, 3, out);

    return out;
}

LocalQuadratic solve_dcol_oracle(
    DcolCallbackContext& ctx, int stage, const Pose2& car1, const Pose2& car2)
{
    LocalQuadratic out;
    if (ctx.oracle_mode == DcolOracleMode::ANALYTIC_SUPPORT) {
        out = solve_analytic_dcol(ctx, car1, car2);
    } else {
        out = solve_inner_dcol(ctx, stage, car1, car2);
    }
    ctx.last_alpha[static_cast<size_t>(stage)] = out.alpha;
    return out;
}

ApiStatus refresh_dcol_quadratics(DcolSolver& solver, DcolCallbackContext& ctx, bool count_callback)
{
    if (count_callback) {
        ++ctx.calls;
    }

    constexpr std::array<int, kPoseDim> grad_params { DCOL_GX1, DCOL_GY1, DCOL_GTHETA1, DCOL_GX2,
        DCOL_GY2, DCOL_GTHETA2 };
    constexpr int hess_params[kPoseDim][kPoseDim] = {
        { DCOL_H00, DCOL_H01, DCOL_H02, DCOL_H03, DCOL_H04, DCOL_H05 },
        { DCOL_H01, DCOL_H11, DCOL_H12, DCOL_H13, DCOL_H14, DCOL_H15 },
        { DCOL_H02, DCOL_H12, DCOL_H22, DCOL_H23, DCOL_H24, DCOL_H25 },
        { DCOL_H03, DCOL_H13, DCOL_H23, DCOL_H33, DCOL_H34, DCOL_H35 },
        { DCOL_H04, DCOL_H14, DCOL_H24, DCOL_H34, DCOL_H44, DCOL_H45 },
        { DCOL_H05, DCOL_H15, DCOL_H25, DCOL_H35, DCOL_H45, DCOL_H55 },
    };

    for (int k = 0; k <= solver.get_horizon(); ++k) {
        const double t = ctx.dt * static_cast<double>(k);
        const double bump_ref = passing_bump(ctx, solver.get_horizon(), k);
        const double theta1_ref = car1_reference_heading(ctx, solver.get_horizon(), k);
        const double theta2_ref = car2_reference_heading(ctx, solver.get_horizon(), k);
        const Pose2 car1 = car1_pose(solver, k);
        const Pose2 car2 = car2_pose(solver, k);
        const LocalQuadratic quad = solve_dcol_oracle(ctx, k, car1, car2);

        const std::array<std::pair<int, double>, 15> base_assignments = { {
            { X1_REF, ctx.car1_v_ref * t },
            { Y1_REF, -bump_ref },
            { THETA1_REF, theta1_ref },
            { V1_REF, ctx.car1_v_ref },
            { X2_REF, ctx.car2_x0 - ctx.car2_v_ref * t },
            { Y2_REF, bump_ref },
            { THETA2_REF, theta2_ref },
            { V2_REF, ctx.car2_v_ref },
            { X1_LIN, car1.x },
            { Y1_LIN, car1.y },
            { THETA1_LIN, car1.theta },
            { X2_LIN, car2.x },
            { Y2_LIN, car2.y },
            { THETA2_LIN, car2.theta },
            { DCOL_ALPHA, quad.alpha },
        } };
        for (const auto& item : base_assignments) {
            const ApiStatus status = solver.set_parameter(k, item.first, item.second);
            if (status != ApiStatus::OK) {
                return status;
            }
        }

        for (int i = 0; i < kPoseDim; ++i) {
            const ApiStatus status = solver.set_parameter(k, grad_params[i], quad.g[i]);
            if (status != ApiStatus::OK) {
                return status;
            }
        }
        for (int i = 0; i < kPoseDim; ++i) {
            for (int j = i; j < kPoseDim; ++j) {
                const ApiStatus status = solver.set_parameter(k, hess_params[i][j], quad.H[i][j]);
                if (status != ApiStatus::OK) {
                    return status;
                }
            }
        }
    }

    return ApiStatus::OK;
}

ApiStatus update_dcol_quadratics(DcolSolver& solver, void* user)
{
    auto* ctx = static_cast<DcolCallbackContext*>(user);
    return refresh_dcol_quadratics(solver, *ctx, true);
}

void seed_joint_avoidance_guess(DcolSolver& solver, const DcolCallbackContext& ctx)
{
    solver.set_initial_state(
        { 0.0, 0.0, 0.0, ctx.car1_v_ref, ctx.car2_x0, 0.0, kPi, ctx.car2_v_ref });
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        const double t = ctx.dt * static_cast<double>(k);
        const double bump = passing_bump(ctx, solver.get_horizon(), k);
        const double theta1_guess = car1_reference_heading(ctx, solver.get_horizon(), k);
        const double theta2_guess = car2_reference_heading(ctx, solver.get_horizon(), k);
        solver.set_state_guess(k, X1, ctx.car1_v_ref * t);
        solver.set_state_guess(k, Y1, -bump);
        solver.set_state_guess(k, THETA1, theta1_guess);
        solver.set_state_guess(k, V1, ctx.car1_v_ref);
        solver.set_state_guess(k, X2, ctx.car2_x0 - ctx.car2_v_ref * t);
        solver.set_state_guess(k, Y2, bump);
        solver.set_state_guess(k, THETA2, theta2_guess);
        solver.set_state_guess(k, V2, ctx.car2_v_ref);
        if (k < solver.get_horizon()) {
            solver.set_control_guess(k, A1, 0.0);
            solver.set_control_guess(k, OMEGA1, 0.0);
            solver.set_control_guess(k, A2, 0.0);
            solver.set_control_guess(k, OMEGA2, 0.0);
        }
    }
}

double minimum_alpha(const DcolCallbackContext& ctx, int horizon)
{
    double min_alpha = 1.0e100;
    for (int k = 0; k <= horizon; ++k) {
        min_alpha = std::min(min_alpha, ctx.last_alpha[static_cast<size_t>(k)]);
    }
    return min_alpha;
}

double max_abs_y(const DcolSolver& solver)
{
    double max_y = 0.0;
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        max_y = std::max(max_y, std::abs(solver.get_state(k, Y1)));
        max_y = std::max(max_y, std::abs(solver.get_state(k, Y2)));
    }
    return max_y;
}

} // namespace

int main(int argc, char** argv)
{
    DcolCallbackContext ctx;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--inner-minisolver") {
            ctx.oracle_mode = DcolOracleMode::INNER_MINISOLVER;
        } else if (arg == "--analytic") {
            ctx.oracle_mode = DcolOracleMode::ANALYTIC_SUPPORT;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n"
                      << "Usage: " << argv[0] << " [--analytic|--inner-minisolver]\n";
            return 1;
        }
    }

    SolverConfig config;
    config.max_iters = 80;
    config.default_dt = ctx.dt;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.hessian_approximation = HessianApproximation::EXACT;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.tol_con = 5.0e-3;
    config.tol_dual = 2.0e-2;
    config.tol_mu = 1.0e-5;

    DcolSolver solver(kOuterHorizon, Backend::CPU_SERIAL, config);
    solver.set_dt(ctx.dt);
    seed_joint_avoidance_guess(solver, ctx);
    if (ctx.oracle_mode == DcolOracleMode::INNER_MINISOLVER) {
        initialize_inner_solvers(ctx, solver.get_horizon());
    }

    ApiStatus callback_status = refresh_dcol_quadratics(solver, ctx, false);
    if (callback_status != ApiStatus::OK) {
        std::cerr << "Initial DCOL refresh failed: " << api_status_to_string(callback_status)
                  << "\n";
        return 1;
    }
    const double initial_alpha = minimum_alpha(ctx, solver.get_horizon());
    solver.set_model_update_callback(update_dcol_quadratics, &ctx);

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();
    callback_status = refresh_dcol_quadratics(solver, ctx, false);
    if (callback_status != ApiStatus::OK) {
        std::cerr << "Final DCOL refresh failed: " << api_status_to_string(callback_status) << "\n";
        return 1;
    }
    const double final_alpha = minimum_alpha(ctx, solver.get_horizon());

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "DCol rectangle-constraint joint two-car callback demo\n";
    std::cout << "oracle="
              << (ctx.oracle_mode == DcolOracleMode::ANALYTIC_SUPPORT ? "analytic_support"
                                                                      : "inner_minisolver")
              << "\n";
    std::cout << "status=" << status_to_string(status) << " iterations=" << info.iterations
              << " callback_calls=" << ctx.calls << "\n";
    if (ctx.oracle_mode == DcolOracleMode::ANALYTIC_SUPPORT) {
        std::cout << "analytic_evals=" << ctx.analytic_evals << "\n";
    } else {
        const double inner_avg_iterations = ctx.inner_solves > 0
            ? static_cast<double>(ctx.inner_total_iterations)
                / static_cast<double>(ctx.inner_solves)
            : 0.0;
        std::cout << "inner_solves=" << ctx.inner_solves << " inner_failures=" << ctx.inner_failures
                  << " inner_total_iterations=" << ctx.inner_total_iterations
                  << " inner_avg_iterations=" << inner_avg_iterations
                  << " inner_max_iterations=" << ctx.inner_max_iterations << "\n";
    }
    std::cout << "primal=" << info.primal_inf << " dual=" << info.dual_inf
              << " complementarity=" << info.complementarity_inf << "\n";
    std::cout << "min_alpha_initial=" << initial_alpha << " min_alpha_final=" << final_alpha
              << " max_abs_y=" << max_abs_y(solver) << "\n";
    std::cout << "alpha > 1 means the two rectangles are at least 1m apart; "
                 "alpha <= 1 means the 1m clearance is violated.\n";
    std::cout << "sample trajectory: k x1 y1 theta1 x2 y2 theta2 alpha\n";
    for (int k = 0; k <= solver.get_horizon(); k += 5) {
        const Pose2 car1 = car1_pose(solver, k);
        const Pose2 car2 = car2_pose(solver, k);
        std::cout << k << " " << car1.x << " " << car1.y << " " << car1.theta << " " << car2.x
                  << " " << car2.y << " " << car2.theta << " "
                  << ctx.last_alpha[static_cast<size_t>(k)] << "\n";
    }

    return 0;
}
