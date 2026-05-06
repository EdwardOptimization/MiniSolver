#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>

#include "minisolver/algorithms/linear_solver.h"
#include "minisolver/algorithms/model_evaluation.h"
#include "minisolver/core/logger.h" // [NEW] Needed for MLOG_DEBUG
#include "minisolver/core/solver_options.h"
#include "minisolver/core/trajectory.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/line_search_utils.h" // [NEW] Needed for fraction_to_boundary_rule

namespace minisolver {

template <typename Model, int MAX_N> class LineSearchStrategy {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;

    using TrajectoryType = Trajectory<KnotPoint<double, NX, NU, NC, NP>, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    virtual ~LineSearchStrategy() = default;

    virtual LineSearchResult search(TrajectoryType& trajectory,
        LinearSolver<TrajArray>& linear_solver, const std::array<double, MAX_N>& dt_traj, double mu,
        double reg, const SolverConfig& config)
        = 0;

    virtual void reset() { }

    // IPOPT §3.1: when μ decreases, filter / merit history built under the old μ
    // is no longer comparable (φ contains -μΣlog(s)). The solver calls this hook
    // at every μ-decrease — default is a no-op so strategies that don't care
    // (e.g. NoLineSearch) need no override.
    virtual void on_barrier_update() { }
};

// --- No Line Search (Full Step / Fraction-to-Boundary Only) ---
// This is a common real-time NMPC setting: avoid backtracking and accept the step directly.
template <typename Model, int MAX_N> class NoLineSearch : public LineSearchStrategy<Model, MAX_N> {
    using Base = LineSearchStrategy<Model, MAX_N>;
    using typename Base::TrajArray;
    using typename Base::TrajectoryType;

public:
    LineSearchResult search(TrajectoryType& trajectory, LinearSolver<TrajArray>& /*linear_solver*/,
        const std::array<double, MAX_N>& dt_traj, double /*mu*/, double /*reg*/,
        const SolverConfig& config) override
    {
        const int N = trajectory.N;
        const auto& active = trajectory.active();

        // Keep s/lam (and soft vars for L1) inside the interior.
        const double alpha = fraction_to_boundary_rule<TrajArray, Model>(
            active, N, config.line_search_tau, config);
        if (alpha <= 1e-8) {
            return LineSearchResult(0.0);
        }

        trajectory.prepare_candidate();
        auto& candidate = trajectory.candidate();

        if (!config.enable_line_search_rollout) {
            // Multiple-shooting style trial point: x/u moved by linear step.
            for (int k = 0; k <= N; ++k) {
                candidate[k].x = active[k].x + active[k].dx * alpha;
                if (k < N) {
                    candidate[k].u = active[k].u + active[k].du * alpha;
                } else {
                    candidate[k].u.setZero();
                }
                candidate[k].s = active[k].s + active[k].ds * alpha;
                candidate[k].lam = active[k].lam + active[k].dlam * alpha;
                candidate[k].soft_s = active[k].soft_s + active[k].dsoft_s * alpha;
                candidate[k].p = active[k].p;
            }
        } else {
            // Dynamics-projection heuristic, not the canonical multiple-shooting
            // line-search point z + alpha*dz:
            // - keep x0 fixed;
            // - apply the multiple-shooting step to u/s/lam/soft_s;
            // - re-integrate states to reduce dynamics defects.
            // This is also not an iLQR/DDP rollout: it does not apply
            // u + alpha*k + K*(x_rollout - x_nominal).
            candidate[0].x = active[0].x;
            for (int k = 0; k <= N; ++k) {
                if (k < N) {
                    candidate[k].u = active[k].u + active[k].du * alpha;
                } else {
                    candidate[k].u.setZero();
                }
                candidate[k].s = active[k].s + active[k].ds * alpha;
                candidate[k].lam = active[k].lam + active[k].dlam * alpha;
                candidate[k].soft_s = active[k].soft_s + active[k].dsoft_s * alpha;
                candidate[k].p = active[k].p;

                const double current_dt = (k < N) ? dt_traj[static_cast<size_t>(k)] : 0.0;
                detail::evaluate_model_stage<Model>(candidate[k], config, current_dt, k == N);
                if (k < N) {
                    candidate[k + 1].x = candidate[k].f_resid;
                }
            }
        }

        // Important: keep the swapped-in iterate internally consistent.
        // prepare_candidate() only copies KnotState (including cached cost/g_val/f_resid from the
        // old iterate). Refresh them once at the accepted point so logging/heuristics and defect
        // metrics reflect the new (x,u,...) rather than stale values.
        if (!config.enable_line_search_rollout) {
            for (int k = 0; k <= N; ++k) {
                const double current_dt = (k < N) ? dt_traj[static_cast<size_t>(k)] : 0.0;
                detail::evaluate_model_stage<Model>(candidate[k], config, current_dt, k == N);
            }
        }

        trajectory.swap();
        return LineSearchResult(alpha);
    }
};

// --- Merit Function Strategy ---
template <typename Model, int MAX_N>
class MeritLineSearch : public LineSearchStrategy<Model, MAX_N> {
    using Base = LineSearchStrategy<Model, MAX_N>;
    using typename Base::TrajArray;
    using typename Base::TrajectoryType;

    double merit_nu = 1000.0;

    static double abs_directional_derivative(double value, double direction)
    {
        if (value > 0.0) {
            return direction;
        }
        if (value < 0.0) {
            return -direction;
        }
        return 0.0;
    }

    double constraint_direction(const typename TrajArray::value_type& kp, int i) const
    {
        double direction = kp.ds(i);
        for (int j = 0; j < Model::NX; ++j) {
            direction += kp.C(i, j) * kp.dx(j);
        }
        for (int j = 0; j < Model::NU; ++j) {
            direction += kp.D(i, j) * kp.du(j);
        }
        return direction;
    }

    double compute_merit_directional_derivative(
        const TrajArray& t, int N, double mu, const SolverConfig& config) const
    {
        double dphi = 0.0;
        const int NC = Model::NC;
        const int NX = Model::NX;
        const int NU = Model::NU;

        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];

            for (int j = 0; j < NX; ++j) {
                dphi += kp.q(j) * kp.dx(j);
            }
            if (k < N) {
                for (int j = 0; j < NU; ++j) {
                    dphi += kp.r(j) * kp.du(j);
                }
            }

            for (int i = 0; i < NC; ++i) {
                if (kp.s(i) > config.min_barrier_slack) {
                    dphi -= mu * kp.ds(i) / kp.s(i);
                }

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                const double g_dir = constraint_direction(kp, i);
                const double g_true = detail::true_constraint_value<Model>(kp, i);

                if (detail::is_l1_soft_constraint(type, w, config)) {
                    if (kp.soft_s(i) > config.min_barrier_slack) {
                        dphi -= mu * kp.dsoft_s(i) / kp.soft_s(i);
                    }
                    if (w - kp.lam(i) > detail::l1_soft_dual_floor(w, config)) {
                        dphi += mu * kp.dlam(i) / (w - kp.lam(i));
                    }
                    dphi += w * kp.dsoft_s(i);

                    const double residual = g_true + kp.s(i) - kp.soft_s(i);
                    const double residual_dir = g_dir - kp.dsoft_s(i);
                    dphi += merit_nu * abs_directional_derivative(residual, residual_dir);
                } else if (detail::is_l2_soft_constraint(type, w)) {
                    const double penalty_residual = g_true + kp.s(i);
                    dphi += w * penalty_residual * g_dir;

                    const double residual = g_true + kp.s(i) - kp.lam(i) / w;
                    const double residual_dir = g_dir - kp.dlam(i) / w;
                    dphi += merit_nu * abs_directional_derivative(residual, residual_dir);
                } else {
                    const double residual = g_true + kp.s(i);
                    dphi += merit_nu * abs_directional_derivative(residual, g_dir);
                }
            }

            if (k < N) {
                for (int row = 0; row < NX; ++row) {
                    double defect = t[k + 1].x(row) - kp.f_resid(row);
                    double defect_dir = t[k + 1].dx(row);
                    for (int col = 0; col < NX; ++col) {
                        defect_dir -= kp.A(row, col) * kp.dx(col);
                    }
                    for (int col = 0; col < NU; ++col) {
                        defect_dir -= kp.B(row, col) * kp.du(col);
                    }
                    dphi += merit_nu * abs_directional_derivative(defect, defect_dir);
                }
            }
        }

        return dphi;
    }

    double compute_merit(const TrajArray& t, int N, double mu, const SolverConfig& config)
    {
        double total_merit = 0.0;
        const int NC = Model::NC;
        const int NX = Model::NX;

        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];
            total_merit += kp.cost;

            // Barrier & Soft Constraint Penalty Calculation
            for (int i = 0; i < NC; ++i) {
                if (kp.s(i) > config.min_barrier_slack) {
                    total_merit -= mu * std::log(kp.s(i));
                } else {
                    total_merit += config.barrier_inf_cost;
                }

                // L1 Soft Constraint: Dual Barrier
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                // L1 Soft Constraint: Dual Barrier + Linear Penalty
                if (detail::is_l1_soft_constraint(type, w, config)) {
                    // 1. Barrier terms
                    if (kp.soft_s(i) > config.min_barrier_slack) {
                        total_merit -= mu * std::log(kp.soft_s(i));
                    } else {
                        total_merit += config.barrier_inf_cost;
                    }

                    if (w - kp.lam(i) > detail::l1_soft_dual_floor(w, config)) {
                        total_merit -= mu * std::log(w - kp.lam(i));
                    } else {
                        total_merit += config.barrier_inf_cost;
                    }

                    // 2. L1 Linear Penalty
                    total_merit += w * kp.soft_s(i);
                }

                // L2 Soft Constraint: Quadratic Penalty
                else if (detail::is_l2_soft_constraint(type, w)) {
                    // L2 Quadratic Penalty: 0.5 * w * (g + s)^2
                    double viol = detail::true_constraint_value<Model>(kp, i) + kp.s(i);
                    total_merit += 0.5 * w * viol * viol;
                }
            }

            // Inequality Violation
            for (int i = 0; i < NC; ++i) {
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                if (detail::is_l1_soft_constraint(type, w, config)) {
                    total_merit += merit_nu
                        * std::abs(
                            detail::true_constraint_value<Model>(kp, i) + kp.s(i) - kp.soft_s(i));
                } else if (detail::is_l2_soft_constraint(type, w)) {
                    total_merit += merit_nu
                        * std::abs(
                            detail::true_constraint_value<Model>(kp, i) + kp.s(i) - kp.lam(i) / w);
                } else {
                    total_merit += merit_nu
                        * std::abs(detail::true_constraint_value<Model>(kp, i) + kp.s(i));
                }
            }

            // Dynamic Defect Violation (Multiple Shooting)
            if (k < N) {
                MSVec<double, NX> defect = t[k + 1].x - kp.f_resid;
                // L1 Norm of defect
                for (int j = 0; j < NX; ++j) {
                    total_merit += merit_nu * std::abs(defect(j));
                }
            }
        }
        return total_merit;
    }

    void build_trial(TrajArray& candidate, const TrajArray& active,
        const std::array<double, MAX_N>& dt_traj, int N, double alpha, const SolverConfig& config)
    {
        if (!config.enable_line_search_rollout) {
            // Multiple-shooting style trial point: x/u moved by linear step.
            for (int k = 0; k <= N; ++k) {
                candidate[k].x = active[k].x + alpha * active[k].dx;
                if (k < N) {
                    candidate[k].u = active[k].u + alpha * active[k].du;
                } else {
                    candidate[k].u.setZero();
                }
                candidate[k].s = active[k].s + alpha * active[k].ds;
                candidate[k].lam = active[k].lam + alpha * active[k].dlam;
                candidate[k].soft_s = active[k].soft_s + alpha * active[k].dsoft_s;
                candidate[k].p = active[k].p;

                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                detail::evaluate_model_stage<Model>(candidate[k], config, current_dt, k == N);
            }
        } else {
            // Dynamics-projection heuristic, not the canonical multiple-shooting
            // line-search point z + alpha*dz:
            // - keep x0 fixed;
            // - apply the multiple-shooting step to u/s/lam/soft_s;
            // - re-integrate states to reduce dynamics defects.
            // This is also not an iLQR/DDP rollout: it does not apply
            // u + alpha*k + K*(x_rollout - x_nominal).
            candidate[0].x = active[0].x;
            for (int k = 0; k <= N; ++k) {
                if (k < N) {
                    candidate[k].u = active[k].u + alpha * active[k].du;
                } else {
                    candidate[k].u.setZero();
                }
                candidate[k].s = active[k].s + alpha * active[k].ds;
                candidate[k].lam = active[k].lam + alpha * active[k].dlam;
                candidate[k].soft_s = active[k].soft_s + alpha * active[k].dsoft_s;
                candidate[k].p = active[k].p;

                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                detail::evaluate_model_stage<Model>(candidate[k], config, current_dt, k == N);
                if (k < N) {
                    candidate[k + 1].x = candidate[k].f_resid;
                }
            }
        }
    }

public:
    void reset() override { merit_nu = 1000.0; }

    // After μ↓ the ratcheted merit_nu was calibrated against the old μ's dual
    // magnitudes and would bias future searches toward over-penalising
    // constraint violation. Reset to baseline so the next search() re-derives
    // merit_nu from the current duals.
    void on_barrier_update() override { merit_nu = 1000.0; }

    // Test / diagnostic accessor.
    double get_merit_nu() const { return merit_nu; }

    LineSearchResult search(TrajectoryType& trajectory, LinearSolver<TrajArray>& /*linear_solver*/,
        const std::array<double, MAX_N>& dt_traj, double mu, double /*reg*/,
        const SolverConfig& config) override
    {
        int N = trajectory.N;
        auto& active = trajectory.active();

        // 1. Update Nu
        double max_dual = 0.0;
        for (int k = 0; k <= N; ++k) {
            double local_max = MatOps::norm_inf(active[k].lam);
            if (local_max > max_dual) {
                max_dual = local_max;
            }
        }
        double required_nu = max_dual * 1.1 + 1.0;
        if (required_nu > merit_nu) {
            merit_nu = required_nu;
        }

        // 2. Initial Merit
        double phi_0 = compute_merit(active, N, mu, config);

        // 3. Calc max alpha
        double alpha = fraction_to_boundary_rule<TrajArray, Model>(
            active, N, config.line_search_tau, config);

        trajectory.prepare_candidate();
        auto& candidate = trajectory.candidate();

        bool accepted = false;
        int ls_iter = 0;

        double dphi = 0.0;
        if (config.armijo_c1 > 0.0 && alpha > 0.0) {
            if (!config.enable_line_search_rollout) {
                dphi = compute_merit_directional_derivative(active, N, mu, config);
            } else {
                const double eps_alpha = std::min(1.0e-6, std::max(1.0e-10, alpha * 1.0e-6));
                build_trial(candidate, active, dt_traj, N, eps_alpha, config);
                const double phi_eps = compute_merit(candidate, N, mu, config);
                dphi = (phi_eps - phi_0) / eps_alpha;
            }
            if (!std::isfinite(dphi)) {
                dphi = 0.0;
            }
        }

        while (ls_iter < config.line_search_max_iters) {
            build_trial(candidate, active, dt_traj, N, alpha, config);

            double phi_alpha = compute_merit(candidate, N, mu, config);

            // Standard Armijo sufficient decrease:
            //   phi(alpha) <= phi(0) + c1 * alpha * dphi
            // dphi is estimated once by a tiny finite-difference step along
            // the same trial-point construction. If the direction is not a
            // merit descent direction, fall back to strict decrease rather
            // than accepting a non-descent "Armijo" step.
            if (config.armijo_c1 > 0.0) {
                const double threshold = phi_0 + config.armijo_c1 * alpha * dphi;
                if (dphi < 0.0 && phi_alpha <= threshold) {
                    accepted = true;
                } else if (dphi >= 0.0 && phi_alpha < phi_0) {
                    accepted = true;
                }
            } else {
                // Simple decrease (legacy, no Armijo).
                if (phi_alpha < phi_0) {
                    accepted = true;
                }
            }

            if (accepted) {
                break;
            }
            alpha *= config.line_search_backtrack_factor;
            ls_iter++;
        }

        if (accepted) {
            trajectory.swap();
        } else {
            LineSearchResult fail(0.0);
            fail.backtracks = ls_iter;
            return fail;
        }

        LineSearchResult ok(alpha);
        ok.backtracks = ls_iter;
        return ok;
    }
};

// --- Filter Strategy ---
template <typename Model, int MAX_N>
class FilterLineSearch : public LineSearchStrategy<Model, MAX_N> {
    using Base = LineSearchStrategy<Model, MAX_N>;
    using typename Base::TrajArray;
    using typename Base::TrajectoryType;

    static constexpr size_t FILTER_CAPACITY = 1024;
    static constexpr double FILTER_THETA_MIN_FACTOR = 1.0e-4;
    static constexpr double FILTER_SWITCHING_DELTA = 1.0;
    static constexpr double FILTER_SWITCHING_S_THETA = 1.1;
    static constexpr double FILTER_SWITCHING_S_PHI = 2.3;

    enum class AcceptedStepType { HType, FType };

    struct FilterAcceptance {
        bool accepted = false;
        AcceptedStepType type = AcceptedStepType::HType;
    };

    std::array<std::pair<double, double>, FILTER_CAPACITY> filter {};
    size_t filter_size_ = 0;
    size_t filter_next_ = 0;
    TrajArray soc_scratch_ {};
    bool filter_bounds_initialized_ = false;
    double theta_min_ = 0.0;
    double theta_max_ = 0.0;

    std::pair<double, double> compute_metrics(
        const TrajArray& t, int N, double mu, const SolverConfig& config)
    {
        double theta = 0.0;
        double phi = 0.0;
        const int NC = Model::NC;
        const int NX = Model::NX;

        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];

            // Objective (Phi) Calculation
            phi += kp.cost;
            for (int i = 0; i < NC; ++i) {
                if (kp.s(i) > config.min_barrier_slack) {
                    phi -= mu * std::log(kp.s(i));
                } else {
                    phi += config.barrier_inf_cost;
                }

                // L1 Soft Constraint: Dual Barrier
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                // L1 Soft Constraint
                if (detail::is_l1_soft_constraint(type, w, config)) {
                    // Barrier terms
                    if (kp.soft_s(i) > config.min_barrier_slack) {
                        phi -= mu * std::log(kp.soft_s(i));
                    } else {
                        phi += config.barrier_inf_cost;
                    }

                    if (w - kp.lam(i) > detail::l1_soft_dual_floor(w, config)) {
                        phi -= mu * std::log(w - kp.lam(i));
                    } else {
                        phi += config.barrier_inf_cost;
                    }

                    // L1 Linear Penalty
                    phi += w * kp.soft_s(i);
                }

                // L2 Soft Constraint
                else if (detail::is_l2_soft_constraint(type, w)) {
                    // L2 Quadratic Penalty: 0.5 * w * (g + s)^2
                    double viol = detail::true_constraint_value<Model>(kp, i) + kp.s(i);
                    phi += 0.5 * w * viol * viol;
                }
            }

            // Infeasibility (Theta)
            for (int i = 0; i < NC; ++i) {
                // Correct residual for L1/L2
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                if (detail::is_l1_soft_constraint(type, w, config)) {
                    // L1: Check extended system residual
                    theta += std::abs(
                        detail::true_constraint_value<Model>(kp, i) + kp.s(i) - kp.soft_s(i));
                } else if (detail::is_l2_soft_constraint(type, w)) {
                    // L2 soft constraints use the primal-dual residual
                    // g_true + s - lam/w = 0. Keep filter theta consistent with
                    // compute_max_violation() on the true nonlinear constraint.
                    theta += std::abs(
                        detail::true_constraint_value<Model>(kp, i) + kp.s(i) - kp.lam(i) / w);
                } else {
                    // Hard
                    theta += std::abs(detail::true_constraint_value<Model>(kp, i) + kp.s(i));
                }
            }

            // Dynamic Defect
            if (k < N) {
                MSVec<double, NX> defect = t[k + 1].x - kp.f_resid;
                for (int j = 0; j < NX; ++j) {
                    theta += std::abs(defect(j));
                }
            }
        }
        return { theta, phi };
    }

    double compute_phi_directional_derivative(
        const TrajArray& t, int N, double mu, const SolverConfig& config) const
    {
        double dphi = 0.0;
        constexpr int NC = Model::NC;
        constexpr int NX = Model::NX;
        constexpr int NU = Model::NU;

        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];
            for (int j = 0; j < NX; ++j) {
                dphi += kp.q(j) * kp.dx(j);
            }
            if (k < N) {
                for (int j = 0; j < NU; ++j) {
                    dphi += kp.r(j) * kp.du(j);
                }
            }

            for (int i = 0; i < NC; ++i) {
                if (kp.s(i) > config.min_barrier_slack) {
                    dphi -= mu * kp.ds(i) / kp.s(i);
                }

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                double g_dir = kp.ds(i);
                for (int j = 0; j < NX; ++j) {
                    g_dir += kp.C(i, j) * kp.dx(j);
                }
                if (k < N) {
                    for (int j = 0; j < NU; ++j) {
                        g_dir += kp.D(i, j) * kp.du(j);
                    }
                }

                if (detail::is_l1_soft_constraint(type, w, config)) {
                    if (kp.soft_s(i) > config.min_barrier_slack) {
                        dphi -= mu * kp.dsoft_s(i) / kp.soft_s(i);
                    }
                    const double soft_dual = w - kp.lam(i);
                    if (soft_dual > detail::l1_soft_dual_floor(w, config)) {
                        dphi += mu * kp.dlam(i) / soft_dual;
                    }
                    dphi += w * kp.dsoft_s(i);
                } else if (detail::is_l2_soft_constraint(type, w)) {
                    const double residual = detail::true_constraint_value<Model>(kp, i) + kp.s(i);
                    dphi += w * residual * g_dir;
                }
            }
        }
        return dphi;
    }

    void initialize_filter_bounds(double theta_0, const SolverConfig& config)
    {
        if (filter_bounds_initialized_) {
            return;
        }
        const double reference_theta = std::max(1.0, theta_0);
        theta_min_ = FILTER_THETA_MIN_FACTOR * reference_theta;
        theta_max_ = config.filter_theta_max_factor * reference_theta;
        filter_bounds_initialized_ = true;
    }

    bool is_f_type(double theta_0, double alpha, double dphi) const
    {
        if (theta_0 > theta_min_ || !(dphi < 0.0) || alpha <= 0.0) {
            return false;
        }

        const double lhs = alpha * std::pow(-dphi, FILTER_SWITCHING_S_PHI);
        const double rhs
            = FILTER_SWITCHING_DELTA * std::pow(std::max(0.0, theta_0), FILTER_SWITCHING_S_THETA);
        return std::isfinite(lhs) && std::isfinite(rhs) && lhs > rhs;
    }

    bool is_h_type_acceptable(
        double theta, double phi, double theta_0, double phi_0, const SolverConfig& config)
    {
        if (theta > theta_max_) {
            return false;
        }

        // Check against current point (Sufficient Decrease)
        // Condition: theta <= (1-gamma)*theta_0 OR phi <= phi_0 - gamma*theta_0
        bool sufficient_decrease = (theta <= (1.0 - config.filter_gamma_theta) * theta_0)
            || (phi <= phi_0 - config.filter_gamma_phi * theta_0);

        if (!sufficient_decrease) {
            return false;
        }

        // Check against filter
        for (size_t idx = 0; idx < filter_size_; ++idx) {
            const auto& entry = filter[idx];
            double theta_j = entry.first;
            double phi_j = entry.second;
            bool sufficient_wrt_filter = (theta <= (1.0 - config.filter_gamma_theta) * theta_j)
                || (phi <= phi_j - config.filter_gamma_phi * theta_j);
            if (!sufficient_wrt_filter) {
                return false;
            }
        }
        return true;
    }

    FilterAcceptance check_acceptance(double theta, double phi, double theta_0, double phi_0,
        double alpha, double dphi, const SolverConfig& config)
    {
        FilterAcceptance result;
        if (theta > theta_max_) {
            return result;
        }

        if (is_f_type(theta_0, alpha, dphi)) {
            result.type = AcceptedStepType::FType;
            result.accepted = (phi <= phi_0 + config.eta_suff_descent * alpha * dphi);
            return result;
        }

        result.type = AcceptedStepType::HType;
        result.accepted = is_h_type_acceptable(theta, phi, theta_0, phi_0, config);
        return result;
    }

    bool try_soc_correction(const TrajArray& active, TrajArray& candidate,
        LinearSolver<TrajArray>& linear_solver, const std::array<double, MAX_N>& dt_traj, int N,
        double mu, double reg, double theta_0, double phi_0, const SolverConfig& config)
    {
        if (config.enable_line_search_rollout) {
            return false;
        }

        if (config.print_level >= PrintLevel::DEBUG) {
            MLOG_DEBUG("Step rejected. Attempting SOC.");
        }

        // Reuse preallocated scratch storage: SOC needs a full knot copy so the Riccati
        // solve can reuse the active linearization matrices without putting a trajectory
        // sized object on the call stack.
        TrajArray& soc_data = soc_scratch_;
        for (int k = 0; k <= N; ++k) {
            soc_data[k] = active[k];
        }

        // SOC correction is applied to the trial candidate, so the primal-dual variables used in
        // the correction equations must also be the candidate's. Keep A/B/C/D/Q/R/H from the
        // active linearization for Riccati reuse.
        for (int k = 0; k <= N; ++k) {
            soc_data[k].s = candidate[k].s;
            soc_data[k].lam = candidate[k].lam;
            soc_data[k].soft_s = candidate[k].soft_s;
        }
        for (int k = 0; k <= N; ++k) {
            detail::evaluate_soc_constraints<Model>(active[k], candidate[k], config);
        }

        // solve_soc uses the candidate as nonlinear residual source and writes the correction step
        // into soc_data.
        const LinearSolveResult soc_result = linear_solver.solve_soc(
            soc_data, candidate, N, mu, reg, config.inertia_strategy, config);
        if (!soc_result.ok) {
            return false;
        }

        const double beta_soc = fraction_to_boundary_rule<TrajArray, Model>(
            soc_data, N, config.line_search_tau, config);
        if (beta_soc <= 1e-8) {
            return false;
        }

        for (int k = 0; k <= N; ++k) {
            candidate[k].x += beta_soc * soc_data[k].dx;
            if (k < N) {
                candidate[k].u += beta_soc * soc_data[k].du;
            } else {
                candidate[k].u.setZero();
            }
            candidate[k].s += beta_soc * soc_data[k].ds;
            candidate[k].lam += beta_soc * soc_data[k].dlam;
            candidate[k].soft_s += beta_soc * soc_data[k].dsoft_s;

            const double current_dt = (k < N) ? dt_traj[k] : 0.0;
            detail::evaluate_model_stage<Model>(candidate[k], config, current_dt, k == N);
        }

        const auto m_soc = compute_metrics(candidate, N, mu, config);
        const bool accepted
            = is_h_type_acceptable(m_soc.first, m_soc.second, theta_0, phi_0, config);
        if (accepted && config.print_level >= PrintLevel::DEBUG) {
            MLOG_DEBUG("SOC Accepted.");
        }
        return accepted;
    }

public:
    FilterLineSearch() = default;

    void reset() override
    {
        filter_size_ = 0;
        filter_next_ = 0;
        filter_bounds_initialized_ = false;
        theta_min_ = 0.0;
        theta_max_ = 0.0;
    }

    // IPOPT §3.1: φ = cost − μ·Σ log(s) — filter entries recorded under the
    // old μ contain stale φ values that are not comparable at the new μ. Clear
    // the filter so the next search() builds a fresh history under the new μ.
    void on_barrier_update() override { reset(); }

    // Test / diagnostic accessor.
    size_t filter_size() const { return filter_size_; }

    LineSearchResult search(TrajectoryType& trajectory, LinearSolver<TrajArray>& linear_solver,
        const std::array<double, MAX_N>& dt_traj, double mu, double reg,
        const SolverConfig& config) override
    {
        int N = trajectory.N;
        auto& active = trajectory.active();

        auto m_0 = compute_metrics(active, N, mu, config);
        double theta_0 = m_0.first;
        double phi_0 = m_0.second;
        initialize_filter_bounds(theta_0, config);
        const double dphi = compute_phi_directional_derivative(active, N, mu, config);

        // Fraction to Boundary
        double alpha = fraction_to_boundary_rule<TrajArray, Model>(
            active, N, config.line_search_tau, config);

        trajectory.prepare_candidate();
        auto& candidate = trajectory.candidate();

        bool accepted = false;
        int ls_iter = 0;
        bool soc_attempted = false;
        LineSearchResult result;
        AcceptedStepType accepted_type = AcceptedStepType::HType;

        while (ls_iter < config.line_search_max_iters) {
            if (!config.enable_line_search_rollout) {
                for (int k = 0; k <= N; ++k) {
                    candidate[k].x = active[k].x + alpha * active[k].dx;
                    if (k < N) {
                        candidate[k].u = active[k].u + alpha * active[k].du;
                    } else {
                        candidate[k].u.setZero();
                    }
                    candidate[k].s = active[k].s + alpha * active[k].ds;
                    candidate[k].lam = active[k].lam + alpha * active[k].dlam;
                    candidate[k].soft_s = active[k].soft_s + alpha * active[k].dsoft_s;
                    candidate[k].p = active[k].p;

                    double current_dt = (k < N) ? dt_traj[k] : 0.0;

                    detail::evaluate_model_stage<Model>(candidate[k], config, current_dt, k == N);
                }
            } else {
                // Dynamics-projection heuristic, not the canonical
                // multiple-shooting line-search point z + alpha*dz. This
                // re-integrates states after applying the multiple-shooting
                // control/slack/dual step; it is not an iLQR/DDP rollout with
                // u + alpha*k + K*(x_rollout - x_nominal).
                candidate[0].x = active[0].x;
                for (int k = 0; k <= N; ++k) {
                    if (k < N) {
                        candidate[k].u = active[k].u + alpha * active[k].du;
                    } else {
                        candidate[k].u.setZero();
                    }
                    candidate[k].s = active[k].s + alpha * active[k].ds;
                    candidate[k].lam = active[k].lam + alpha * active[k].dlam;
                    candidate[k].soft_s = active[k].soft_s + alpha * active[k].dsoft_s;
                    candidate[k].p = active[k].p;

                    double current_dt = (k < N) ? dt_traj[k] : 0.0;

                    detail::evaluate_model_stage<Model>(candidate[k], config, current_dt, k == N);
                    if (k < N) {
                        candidate[k + 1].x = candidate[k].f_resid;
                    }
                }
            }

            auto m_alpha = compute_metrics(candidate, N, mu, config);
            const FilterAcceptance acceptance = check_acceptance(
                m_alpha.first, m_alpha.second, theta_0, phi_0, alpha, dphi, config);
            if (acceptance.accepted) {
                accepted = true;
                accepted_type = acceptance.type;
            }

            // SOC Logic
            // Current SOC is defined for multiple-shooting candidate updates. In rollout mode,
            // state corrections are not applied directly, so skip SOC until a control-space SOC
            // variant is implemented.
            if (!accepted && config.enable_soc && !soc_attempted && ls_iter == 0
                && alpha > config.soc_trigger_alpha && !config.enable_line_search_rollout) {
                const bool soc_accepted = try_soc_correction(
                    active, candidate, linear_solver, dt_traj, N, mu, reg, theta_0, phi_0, config);
                soc_attempted = true;
                accepted = soc_accepted;
                result.soc_attempted = true;
                result.soc_accepted = soc_accepted;
                result.soc_rejected = !soc_accepted;
                if (accepted) {
                    accepted_type = AcceptedStepType::HType;
                    break;
                }
            }

            if (accepted) {
                break;
            }
            alpha *= config.line_search_backtrack_factor;
            ls_iter++;
        }

        if (accepted) {
            trajectory.swap();
            if (accepted_type == AcceptedStepType::HType) {
                if (filter_size_ < FILTER_CAPACITY) {
                    filter[filter_size_] = { theta_0, phi_0 };
                    ++filter_size_;
                } else {
                    filter[filter_next_] = { theta_0, phi_0 };
                    filter_next_ = (filter_next_ + 1) % FILTER_CAPACITY;
                }
            }
        } else {
            result.alpha = 0.0;
            result.backtracks = ls_iter;
            return result; // Fail
        }

        result.alpha = alpha;
        result.backtracks = ls_iter;
        return result;
    }
};

}
