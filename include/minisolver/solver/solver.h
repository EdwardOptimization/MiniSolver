#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "minisolver/core/logger.h"
#include "minisolver/core/model_traits.h"
#include "minisolver/core/solver_context.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/trajectory.h"
#include "minisolver/core/types.h"

#include "minisolver/algorithms/line_search.h"
#include "minisolver/algorithms/linear_solver.h"
#include "minisolver/algorithms/model_evaluation.h"
#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/integrator/implicit_integrator.h"

#include "minisolver/backend/backend_interface.h"

namespace minisolver {

template <typename Model, int MAX_N> class SolverSerializer;

// Test-only friend hook. Defined in tests/; forward-declared here so the
// private `apply_slack_reset_` helper can be exercised in isolation without
// growing the public API.
namespace test {
    template <typename, int> struct SolverInternalAccess;
}

class SolverTimer {
public:
    using Clock = std::chrono::steady_clock;
    std::map<std::string, double> times;
    std::vector<std::pair<std::string, std::chrono::time_point<Clock>>> stack;
    bool enabled = false; // Default disabled, use for test memory allocation

    void start(const std::string& name)
    {
        if (!enabled) {
            return;
        }
        stack.push_back({ name, Clock::now() });
    }

    void stop()
    {
        if (!enabled) {
            return;
        }
        if (stack.empty()) {
            return;
        }
        auto end_time = Clock::now();
        auto& entry = stack.back();
        std::chrono::duration<double, std::milli> ms = end_time - entry.second;
        times[entry.first] += ms.count();
        stack.pop_back();
    }

    void reset()
    {
        times.clear();
        stack.clear();
    }

    void print()
    {
        if (!enabled) {
            return;
        }
        std::stringstream ss;
        ss << "\n--- Solver Profiling (ms) ---\n";
        for (auto const& [name, time] : times) {
            ss << std::left << std::setw(20) << name << ": " << time << "\n";
        }
        ss << "-----------------------------";
        MLOG_INFO(ss.str());
    }
};

template <typename Model, int _MAX_N> class MiniSolver {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;
    static constexpr int MAX_N = _MAX_N;

    using Knot = KnotPoint<double, NX, NU, NC, NP>;
    using TrajectoryType = Trajectory<Knot, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    friend class SolverSerializer<Model, MAX_N>;
    template <typename, int> friend struct ::minisolver::test::SolverInternalAccess;

    MiniSolver(int initial_N, Backend b, SolverConfig conf = SolverConfig())
        : trajectory(std::max(0, std::min(initial_N, MAX_N)))
        , N(std::max(0, std::min(initial_N, MAX_N)))
        , config(conf)
    {

        if (initial_N < 0 || initial_N > MAX_N) {
            std::cerr << "Error: N (" << initial_N << ") outside [0, " << MAX_N << "]. Clamping.\n";
        }

        // Constructor has an explicit backend argument; keep it as the source of truth.
        config.backend = b;
        context_.reset_algorithmic(config.mu_init, config.reg_init);

        // Fused Riccati kernel is CSE'd against a specific integrator's A/B
        // sparsity at code-gen time. Running with a different runtime
        // integrator may produce wrong fused-kernel results. Warn at
        // construction; the Riccati dispatch already skips the fused kernel
        // when the integrator doesn't match, so a hard throw here would block
        // legitimate non-fused usage.
        if constexpr (detail::has_generated_integrator_v<Model>) {
            if (Model::generated_integrator != config.integrator) {
                std::cerr << "MiniSolver: Model was generated for "
                          << static_cast<int>(Model::generated_integrator)
                          << " but config.integrator is " << static_cast<int>(config.integrator)
                          << ". Fused Riccati kernel will be skipped.\n";
            }
        }

        dt_traj.fill(conf.default_dt);
        rebuild_solver_components();

        // Pre-size the diagnostic line-search trace so solve()'s push_back
        // stays pointer-bump only (zero-malloc hot path). If the user later
        // raises config.max_iters via set_config(), solve() will reserve
        // again once on the first call at the new size.
        if (conf.max_iters > 0) {
            alpha_log_.reserve(static_cast<size_t>(conf.max_iters));
        }

        // Initialize Maps
        for (int i = 0; i < NX; ++i) {
            state_map[Model::state_names[i]] = i;
        }
        for (int i = 0; i < NU; ++i) {
            control_map[Model::control_names[i]] = i;
        }
        for (int i = 0; i < NP; ++i) {
            param_map[Model::param_names[i]] = i;
        }
    }

    // Reset Function
    void reset(ResetOption option = ResetOption::ALG_STATE)
    {
        // 1. Reset Algorithmic Scalars
        context_.reset_algorithmic(config.mu_init, config.reg_init);
        // 2. Reset Components
        if (line_search) {
            line_search->reset();
        }
        timer.reset();

        // 3. Reset Trajectory Data (Optional)
        if (option == ResetOption::FULL) {
            trajectory.reset();

            // Reset Time Steps to default configuration
            dt_traj.fill(config.default_dt);

            // Note: This clears parameters (p) too.
            // You will need to call set_parameter() and set_dt() again after a FULL reset.
        }
    }

    int get_state_idx(const std::string& name) const
    {
        auto it = state_map.find(name);
        if (it != state_map.end()) {
            return it->second;
        }
        return -1;
    }

    int get_control_idx(const std::string& name) const
    {
        auto it = control_map.find(name);
        if (it != control_map.end()) {
            return it->second;
        }
        return -1;
    }

    int get_param_idx(const std::string& name) const
    {
        auto it = param_map.find(name);
        if (it != param_map.end()) {
            return it->second;
        }
        return -1;
    }

    void resize_horizon(int new_n)
    {
        if (new_n < 0 || new_n > MAX_N) {
            std::cerr << "Error: new_n outside valid range [0, MAX_N]\n";
            return;
        }
        int old_n = N;
        N = new_n;
        trajectory.resize(N);
        if (new_n > old_n) {
            // Initialize new time steps for newly added intervals.
            for (int k = old_n; k < new_n; ++k) {
                dt_traj[k] = config.default_dt;
            }
        }
    }

    int get_horizon() const { return N; }

    void set_config(const SolverConfig& conf)
    {
        bool line_search_changed = conf.line_search_type != config.line_search_type;
        // Backend is fixed at construction time (see ctor note: "explicit
        // backend argument; keep it as the source of truth"). Preserve it
        // across set_config so a caller passing a default-constructed conf
        // doesn't silently switch backends.
        const Backend preserved_backend = config.backend;
        config = conf;
        config.backend = preserved_backend;
        if (config.max_iters > 0 && static_cast<int>(alpha_log_.capacity()) < config.max_iters) {
            alpha_log_.reserve(static_cast<size_t>(config.max_iters));
        }
        if (line_search_changed) {
            components_dirty = true;
        }
    }

    const SolverConfig& get_config() const { return config; }

    int get_iteration_count() const { return context_.solve.current_iter; }

    double get_profile_time_ms(const std::string& name) const
    {
        auto it = timer.times.find(name);
        return it != timer.times.end() ? it->second : 0.0;
    }

    // --- High-Level API ---

    // 1. Initial State
    void set_initial_state(const std::vector<double>& x0)
    {
        if (x0.size() != NX) {
            return;
        }
        auto& kp = trajectory[0];
        for (int i = 0; i < NX; ++i) {
            kp.x(i) = x0[i];
        }
    }

    void set_initial_state(const std::string& name, double value)
    {
        int idx = get_state_idx(name);
        if (idx != -1) {
            trajectory[0].x(idx) = value;
        } else {
            std::cerr << "Warning: Unknown state " << name << "\n";
        }
    }

    // 2. Parameters
    void set_parameter(int stage, int idx, double value)
    {
        if (stage > N || stage < 0) {
            return;
        }
        if (idx >= NP || idx < 0) {
            return;
        }
        trajectory[stage].p(idx) = value;
    }

    void set_parameter(int stage, const std::string& name, double value)
    {
        int idx = get_param_idx(name);
        if (idx != -1) {
            set_parameter(stage, idx, value);
        } else {
            std::cerr << "Warning: Unknown param " << name << "\n";
        }
    }

    void set_global_parameter(int idx, double value)
    {
        if (idx >= NP || idx < 0) {
            return;
        }
        for (int k = 0; k <= N; ++k) {
            trajectory[k].p(idx) = value;
        }
    }

    void set_global_parameter(const std::string& name, double value)
    {
        int idx = get_param_idx(name);
        if (idx != -1) {
            set_global_parameter(idx, value);
        } else {
            std::cerr << "Warning: Unknown param " << name << "\n";
        }
    }

    double get_parameter(int stage, int idx) const
    {
        if (stage > N || stage < 0 || idx >= NP || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].p(idx);
    }

    double get_parameter(int stage, const std::string& name) const
    {
        int idx = get_param_idx(name);
        if (idx != -1) {
            return get_parameter(stage, idx);
        }
        return 0.0;
    }

    std::vector<double> get_parameters(int stage) const
    {
        if (stage > N || stage < 0) {
            return {};
        }
        const auto& kp = trajectory[stage];
        std::vector<double> res(NP);
        for (int i = 0; i < NP; ++i) {
            res[i] = kp.p(i);
        }
        return res;
    }

    // 3. State Access
    void set_state_guess(int stage, int idx, double value)
    {
        if (stage > N || stage < 0 || idx >= NX || idx < 0) {
            return;
        }
        trajectory[stage].x(idx) = value;
    }

    void set_state_guess(int stage, const std::string& name, double value)
    {
        int idx = get_state_idx(name);
        if (idx != -1) {
            set_state_guess(stage, idx, value);
        }
    }

    void set_state_guess_traj(const std::string& name, const std::vector<double>& values)
    {
        int idx = get_state_idx(name);
        if (idx == -1) {
            return;
        }
        int count = std::min((int)values.size(), N + 1);
        for (int k = 0; k < count; ++k) {
            trajectory[k].x(idx) = values[k];
        }
    }

    std::vector<double> get_state_traj(const std::string& name) const
    {
        int idx = get_state_idx(name);
        if (idx == -1) {
            return {};
        }
        std::vector<double> res;
        res.reserve(N + 1);
        for (int k = 0; k <= N; ++k) {
            res.push_back(trajectory[k].x(idx));
        }
        return res;
    }

    std::vector<double> get_state(int stage) const
    {
        if (stage > N || stage < 0) {
            return {};
        }
        const auto& kp = trajectory[stage];
        std::vector<double> res(NX);
        for (int i = 0; i < NX; ++i) {
            res[i] = kp.x(i);
        }
        return res;
    }

    double get_state(int stage, int idx) const
    {
        if (stage > N || stage < 0 || idx >= NX || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].x(idx);
    }

    // 4. Control Access
    void set_control_guess(int stage, int idx, double value)
    {
        if (stage >= N || stage < 0 || idx >= NU || idx < 0) {
            return;
        }
        trajectory[stage].u(idx) = value;
    }

    void set_control_guess(int stage, const std::string& name, double value)
    {
        int idx = get_control_idx(name);
        if (idx != -1) {
            set_control_guess(stage, idx, value);
        }
    }

    void set_control_guess_traj(const std::string& name, const std::vector<double>& values)
    {
        int idx = get_control_idx(name);
        if (idx == -1) {
            return;
        }
        int count = std::min((int)values.size(), N);
        for (int k = 0; k < count; ++k) {
            trajectory[k].u(idx) = values[k];
        }
    }

    std::vector<double> get_control_traj(const std::string& name) const
    {
        int idx = get_control_idx(name);
        if (idx == -1) {
            return {};
        }
        std::vector<double> res;
        res.reserve(N);
        for (int k = 0; k < N; ++k) {
            res.push_back(trajectory[k].u(idx));
        }
        return res;
    }

    std::vector<double> get_control(int stage) const
    {
        if (stage >= N || stage < 0) {
            return {};
        }
        const auto& kp = trajectory[stage];
        std::vector<double> res(NU);
        for (int i = 0; i < NU; ++i) {
            res[i] = kp.u(i);
        }
        return res;
    }

    double get_control(int stage, int idx) const
    {
        if (stage >= N || stage < 0 || idx >= NU || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].u(idx);
    }

    // 5. Slack / Dual Access
    void set_slack_guess(int stage, int idx, double value)
    {
        if (stage > N || stage < 0 || idx >= NC || idx < 0) {
            return;
        }
        trajectory[stage].s(idx) = value;
    }

    void set_dual_guess(int stage, int idx, double value)
    {
        if (stage > N || stage < 0 || idx >= NC || idx < 0) {
            return;
        }
        trajectory[stage].lam(idx) = value;
    }

    std::vector<double> get_slack(int stage) const
    {
        if (stage > N || stage < 0) {
            return {};
        }
        const auto& kp = trajectory[stage];
        std::vector<double> res(NC);
        for (int i = 0; i < NC; ++i) {
            res[i] = kp.s(i);
        }
        return res;
    }

    std::vector<double> get_dual(int stage) const
    {
        if (stage > N || stage < 0) {
            return {};
        }
        const auto& kp = trajectory[stage];
        std::vector<double> res(NC);
        for (int i = 0; i < NC; ++i) {
            res[i] = kp.lam(i);
        }
        return res;
    }

    double get_slack(int stage, int idx) const
    {
        if (stage > N || stage < 0 || idx >= NC || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].s(idx);
    }

    double get_dual(int stage, int idx) const
    {
        if (stage > N || stage < 0 || idx >= NC || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].lam(idx);
    }

    // 6. Cost Access
    double get_stage_cost(int stage) const
    {
        if (stage > N || stage < 0) {
            return 0.0;
        }
        return trajectory[stage].cost;
    }

    // Per-solve line-search α trace. Appended once per accepted/rejected
    // line-search attempt inside solve(); cleared at solve() entry. Empty
    // outside of a solve. Purely diagnostic.
    const std::vector<double>& get_alpha_log() const { return alpha_log_; }

    // Helper to get constraint value
    double get_constraint_val(int stage, int idx) const
    {
        if (stage > N || stage < 0 || idx >= NC || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].g_val(idx);
    }

    void set_dt(const std::vector<double>& dts)
    {
        if (dts.size() > MAX_N) {
            std::cerr << "Warning: DT vector too large.\n";
        }
        int count = std::min((int)dts.size(), N);
        for (int i = 0; i < count; ++i) {
            dt_traj[i] = dts[i];
        }

        // [FIX] Initialize remaining steps to avoid garbage values
        double fill_val = (count > 0) ? dts[count - 1] : config.default_dt;
        for (int i = count; i < MAX_N; ++i) {
            dt_traj[i] = fill_val;
        }
    }

    void set_dt(double dt) { dt_traj.fill(dt); }

    void rollout_dynamics()
    {
        auto& traj = trajectory.active();
        for (int k = 0; k < N; ++k) {
            double current_dt = dt_traj[k];
            traj[k + 1].x = detail::dispatch_integrate<Model>(traj[k].x, traj[k].u, traj[k].p,
                current_dt, config.integrator, config.newton_config);
        }
    }

private:
    bool check_convergence(double max_viol, double max_dual, double max_kkt_error)
    {
        // must satisfy all of the following conditions:
        // 1. barrier parameter mu is small enough (target precision)
        // 2. primal constraints are satisfied (Feasible)
        // 3. dual gradient is satisfied (Stationary)
        // 4. complementarity is satisfied (Complementarity): s*lam is close to mu

        bool mu_converged = (context_.solve.mu <= config.mu_final);
        bool primal_ok = (max_viol <= config.tol_con);
        bool dual_ok = (max_dual <= config.tol_dual);

        // complementarity error tolerance is usually set to kappa * mu or a slightly relaxed fixed
        // value here we require it to converge to tol_cost or tol_mu level
        bool kkt_ok = (max_kkt_error <= std::max(config.tol_mu, 10.0 * context_.solve.mu));

        return mu_converged && primal_ok && dual_ok && kkt_ok;
    }

    double compute_objective_cost_(const TrajArray& traj) const
    {
        double total_cost = 0.0;
        for (int k = 0; k <= N; ++k) {
            total_cost += traj[k].cost;
        }
        return total_cost;
    }

    bool should_stop_for_cost_stagnation_(
        double current_cost, double last_cost, double last_mu) const
    {
        // 只有在满足一定可行性时，Cost 停滞才有意义。
        const double feasible_bound = config.tol_con * config.feasible_tol_scale;
        if (context_.metrics.last_prim_inf > feasible_bound) {
            return false;
        }

        const double cost_diff = std::abs(current_cost - last_cost);
        if (cost_diff >= config.tol_cost) {
            return false;
        }

        const bool mu_decreased = (context_.solve.mu < last_mu);
        const bool mu_small = (context_.solve.mu <= config.mu_final);
        return mu_small || !mu_decreased;
    }

    StepResidualSummary evaluate_step_model_(TrajArray& traj)
    {
        StepResidualSummary summary;
        double total_gap = 0.0;
        int total_gap_dim = 0;

        for (int k = 0; k <= N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;

            detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N);

            for (int i = 0; i < NC; ++i) {
                const double s = traj[k].s(i);
                const double lam = traj[k].lam(i);
                double comp = std::abs(s * lam - context_.solve.mu);
                if (comp > summary.max_kkt_error) {
                    summary.max_kkt_error = comp;
                }

                total_gap += s * lam;
                total_gap_dim += 1;

                // L1 soft constraint has an additional complementarity pair:
                //   soft_s * (w - lam) = mu, with implicit soft-dual (w - lam) > 0.
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }
                if (type == 1 && w > 1e-6) {
                    const double soft_s = traj[k].soft_s(i);
                    const double soft_dual = (w - lam);
                    double comp_soft = std::abs(soft_s * soft_dual - context_.solve.mu);
                    if (comp_soft > summary.max_kkt_error) {
                        summary.max_kkt_error = comp_soft;
                    }

                    total_gap += soft_s * soft_dual;
                    total_gap_dim += 1;
                }
            }
        }

        summary.max_prim_inf = compute_max_violation(traj);
        summary.avg_gap = (total_gap_dim > 0) ? (total_gap / total_gap_dim) : 0.0;
        return summary;
    }

    StepResidualSummary evaluate_derivatives_phase_(TrajArray& traj)
    {
        timer.start("Derivatives");
        StepResidualSummary residuals = evaluate_step_model_(traj);
        timer.stop();
        return residuals;
    }

    void update_barrier_for_step_(const StepResidualSummary& residuals)
    {
        // In Mehrotra mode, mu is updated dynamically inside the step via predictor-corrector
        // logic. We only use update_barrier for Monotone/Adaptive strategies or as a fallback.
        if (config.barrier_strategy != BarrierStrategy::MEHROTRA) {
            update_barrier(residuals.max_kkt_error, residuals.avg_gap);
        }
    }

    double compute_dual_infeasibility_(const TrajArray& traj) const
    {
        double max_dual_inf = 0.0;
        for (int k = 0; k <= N; ++k) {
            double r_norm = MatOps::norm_inf(traj[k].r_bar);
            if (r_norm > max_dual_inf) {
                max_dual_inf = r_norm;
            }
        }
        return max_dual_inf;
    }

    void increase_regularization_after_failed_solve_(bool clamp_to_min)
    {
        if (clamp_to_min && context_.solve.reg < config.reg_min) {
            context_.solve.reg = config.reg_min;
        }
        context_.solve.reg *= config.reg_scale_up;
        if (context_.solve.reg > config.reg_max) {
            context_.solve.reg = config.reg_max;
        }
    }

    bool solve_linear_system_with_retries_(
        TrajArray& traj, double target_mu, const TrajArray* affine_traj, bool clamp_reg_to_min)
    {
        bool success = false;
        for (int try_count = 0; try_count < config.inertia_max_retries; ++try_count) {
            success = linear_solver->solve(
                traj, N, target_mu, context_.solve.reg, config.inertia_strategy, config,
                affine_traj);
            if (success) {
                break;
            }
            increase_regularization_after_failed_solve_(clamp_reg_to_min);
        }
        return success;
    }

    void prepare_direction_workspace_()
    {
        // Candidate buffer preparation (shared by Mehrotra predictor and IR backup).
        // Both features need a full copy of the current linearized system (A, B, Q, R).
        // - Mehrotra: uses it as the affine solve workspace
        // - IR: uses it to access original A, B matrices after Riccati overwrites active
        // Note: The Mehrotra affine solve only modifies solver workspace (Q_bar, R_bar, K, dx...)
        //        but NOT model derivatives (A, B, C, D, Q, R, f_resid, x), so the IR backup
        //        remains valid even after the affine solve.
        bool need_candidate_backup = (config.barrier_strategy == BarrierStrategy::MEHROTRA)
            || config.enable_iterative_refinement;
        if (need_candidate_backup) {
            trajectory.prepare_candidate_full();
        }
    }

    double compute_fraction_to_boundary_(const TrajArray& direction_traj) const
    {
        double alpha = 1.0;
        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NC; ++i) {
                double s = direction_traj[k].s(i);
                double ds = direction_traj[k].ds(i);
                double lam = direction_traj[k].lam(i);
                double dlam = direction_traj[k].dlam(i);

                if (ds < 0) {
                    double a = -s / ds;
                    if (a < alpha) {
                        alpha = a;
                    }
                }
                if (dlam < 0) {
                    double a = -lam / dlam;
                    if (a < alpha) {
                        alpha = a;
                    }
                }

                // L1 soft: soft_s > 0 and (w - lam) > 0.
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }
                if (type == 1 && w > 1e-6) {
                    double soft_s = direction_traj[k].soft_s(i);
                    double dsoft_s = direction_traj[k].dsoft_s(i);
                    if (dsoft_s < 0) {
                        double a = -soft_s / dsoft_s;
                        if (a < alpha) {
                            alpha = a;
                        }
                    }
                    // soft dual: (w - lam) with direction -dlam.
                    double soft_dual = w - lam;
                    double dsoft_dual = -dlam; // d(w-lam)/dalpha = -dlam
                    if (dsoft_dual < 0) {
                        double a = -soft_dual / dsoft_dual;
                        if (a < alpha) {
                            alpha = a;
                        }
                    }
                }
            }
        }
        return alpha;
    }

    double compute_affine_barrier_mu_(
        const TrajArray& base_traj, const TrajArray& affine_traj, double alpha_aff) const
    {
        double total_comp = 0.0;
        int total_dim = 0;

        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NC; ++i) {
                double s_new = base_traj[k].s(i) + alpha_aff * affine_traj[k].ds(i);
                double lam_new = base_traj[k].lam(i) + alpha_aff * affine_traj[k].dlam(i);
                if (s_new < 0) {
                    s_new = 1e-8; // Should not happen with fraction_to_boundary.
                }
                if (lam_new < 0) {
                    lam_new = 1e-8;
                }
                total_comp += s_new * lam_new;
                total_dim++;

                // L1 soft pair: soft_s * (w - lam).
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }
                if (type == 1 && w > 1e-6) {
                    double soft_s_new
                        = base_traj[k].soft_s(i) + alpha_aff * affine_traj[k].dsoft_s(i);
                    double soft_dual_new = w - lam_new;
                    if (soft_s_new < 0) {
                        soft_s_new = 1e-8;
                    }
                    if (soft_dual_new < 0) {
                        soft_dual_new = 1e-8;
                    }
                    total_comp += soft_s_new * soft_dual_new;
                    total_dim++;
                }
            }
        }

        return total_comp / std::max(1, total_dim);
    }

    void maybe_decay_regularization_after_solve_(bool solve_success)
    {
        // A successful Riccati factorization only says the linearized KKT was factorizable at
        // the current reg; it says nothing about whether the resulting step is admissible.
        // The last accepted line-search α is a composite proxy for step quality.
        if (solve_success && context_.solve.reg > config.reg_min
            && context_.metrics.last_alpha > 0.5) {
            context_.solve.reg
                = std::max(config.reg_min, context_.solve.reg / config.reg_scale_down);
        }
    }

    void refine_direction_if_enabled_(TrajArray& traj, bool solve_success)
    {
        if (!solve_success || !config.enable_iterative_refinement) {
            return;
        }
        // Pass 'traj' (which contains solution dx, du) and 'candidate' (which contains original
        // system).
        linear_solver->refine(
            traj, trajectory.candidate(), N, context_.solve.mu, context_.solve.reg, config);
    }

    bool should_stop_after_line_search_(
        const TrajArray& traj, double alpha, double max_prim_inf, double max_dual_inf) const
    {
        bool is_feasible = (max_prim_inf < config.tol_con);
        bool is_dual_feasible = (max_dual_inf < config.tol_dual);

        if (context_.solve.mu > config.mu_final || alpha <= 1e-5) {
            return false;
        }

        double max_dx = 0.0;
        for (int k = 0; k <= N; ++k) {
            // Use MatOps::norm_inf to support both Eigen and MiniMatrix.
            double dx_norm = MatOps::norm_inf(traj[k].dx);
            if (dx_norm > max_dx) {
                max_dx = dx_norm;
            }
        }

        // Use unscaled Newton step (max_dx) to check stationarity.
        // Using (alpha * max_dx) is dangerous because small alpha (blocked step)
        // can look like convergence.
        return max_dx < config.tol_grad && is_feasible && is_dual_feasible;
    }

    void update_metrics_after_globalization_(double max_dual_inf)
    {
        // For outer-loop heuristics (e.g. cost stagnation), store feasibility of the *current*
        // iterate, i.e. after line-search has potentially swapped buffers.
        context_.metrics.last_prim_inf = compute_max_violation(trajectory.active());
        context_.metrics.last_dual_inf = max_dual_inf;
    }

    SolverStatus classify_tiny_step_stagnation_(
        double max_prim_inf, double max_dual_inf) const
    {
        if (max_prim_inf > config.tol_con) {
            return SolverStatus::UNSOLVED;
        }
        if (max_dual_inf <= config.tol_dual) {
            return SolverStatus::OPTIMAL;
        }
        return SolverStatus::FEASIBLE;
    }

    bool attempt_tiny_step_recovery_(TrajArray& traj_after_ls, double alpha)
    {
        bool recovered = false;

        // Step 1: Slack Reset (With counter protection)
        // Only allow attempt when slack_reset_consecutive_count < 1.
        // This means if SlackReset was used in the last iteration but Alpha=0 still (dead
        // loop), this time we force skipping Step 1 and go directly to Step 2 (Restoration).
        if (config.enable_slack_reset && alpha < config.slack_reset_trigger
            && context_.solve.slack_reset_consecutive_count < 1) {
            if (config.print_level >= PrintLevel::DEBUG) {
                MLOG_DEBUG("Triggering Slack Reset (Attempt "
                    << context_.solve.slack_reset_consecutive_count + 1 << ").");
            }

            apply_slack_reset_(traj_after_ls);
            // Try one solve to see if a valid direction can be obtained
            recovered = linear_solver->solve(traj_after_ls, N, context_.solve.mu,
                context_.solve.reg, config.inertia_strategy, config);

            if (recovered) {
                // Mark: If this Reset failed to get us out of trouble, disallow it next time
                context_.solve.slack_reset_consecutive_count++;
            }
        } else if (
            config.enable_slack_reset && context_.solve.slack_reset_consecutive_count >= 1) {
            if (config.print_level >= PrintLevel::DEBUG) {
                MLOG_DEBUG("Skipping Slack Reset to prevent cycle. Forcing Restoration.");
            }
        }

        // Step 2: Feasibility Restoration (If Step 1 failed or was skipped)
        if (!recovered && config.enable_feasibility_restoration) {
            recovered = feasibility_restoration();
            // If restoration succeeded (state x moved), we can reset the counter, allowing
            // SlackReset to be used again in the future
            if (recovered) {
                context_.solve.slack_reset_consecutive_count = 0;
            }
        }

        return recovered;
    }

    SolverStatus globalize_step_(
        double mu_before_step, double max_prim_inf, double max_dual_inf, double& alpha)
    {
        // Notify the line search if μ decreased during this step so it can
        // discard barrier-dependent history (filter entries, ratcheted merit_nu).
        if (line_search && context_.solve.mu < mu_before_step) {
            line_search->on_barrier_update();
        }

        timer.start("Line Search");
        alpha = line_search->search(
            trajectory, *linear_solver, dt_traj, context_.solve.mu, context_.solve.reg, config);
        alpha_log_.push_back(alpha);
        context_.metrics.last_alpha = alpha;
        // IMPORTANT: line_search may swap trajectory buffers.
        // Do not use references to trajectory.active() taken before this call.
        auto& traj_after_ls = trajectory.active();

        // If step size is valid, reset the counter.
        if (alpha > 1e-8) {
            context_.solve.slack_reset_consecutive_count = 0;
        } else {
            SolverStatus stagnation_status
                = classify_tiny_step_stagnation_(max_prim_inf, max_dual_inf);
            if (stagnation_status == SolverStatus::OPTIMAL) {
                if (config.print_level >= PrintLevel::INFO) {
                    MLOG_INFO("Line search stagnated at optimal point (PrimInf: "
                        << max_prim_inf << ", DualInf: " << max_dual_inf
                        << "). Terminating as SOLVED.");
                }
                timer.stop();
                return SolverStatus::OPTIMAL;
            }
            if (stagnation_status == SolverStatus::FEASIBLE) {
                if (config.print_level >= PrintLevel::INFO) {
                    MLOG_INFO("Line search stagnated at feasible point (PrimInf: "
                        << max_prim_inf << "). Terminating as FEASIBLE.");
                }
                timer.stop();
                return SolverStatus::FEASIBLE;
            }

            bool recovered = attempt_tiny_step_recovery_(traj_after_ls, alpha);

            if (!recovered) {
                // Both Slack Reset and Feasibility Restoration failed (or were disabled).
                timer.stop();
                print_iteration_log(alpha);
                return SolverStatus::INFEASIBLE;
            }
        }

        timer.stop();

        print_iteration_log(alpha);
        return SolverStatus::UNSOLVED;
    }

    double compute_mehrotra_target_mu_(double mu_curr, double mu_aff, double alpha_aff) const
    {
        // Aggressive Update: Use sigma^k with k >= 1
        // Heuristic: If affine step is good (large alpha_aff), be aggressive.
        // If alpha_aff is small, be conservative.
        double sigma_base = std::pow(mu_aff / mu_curr, 3);
        double sigma = sigma_base;

        // Aggressive Strategy
        // If alpha_aff close to 1, we can reduce mu significantly.
        if (config.enable_aggressive_barrier) {
            if (alpha_aff > 0.9) {
                sigma = std::min(
                    sigma, 0.01); // [AGGRESSIVE] Force 100x reduction if direction is good.
            }
            // If we are far from solution (large gap), allow faster drop.
            if (mu_curr > 1.0) {
                sigma = std::min(sigma, 0.1);
            }
        } else {
            // Mehrotra Centering Parameter Heuristic
            // Limit sigma to avoid aggressive reduction when affine step is bad.
            if (alpha_aff < 0.1) {
                // If affine direction is blocked quickly, we are close to boundary.
                // Be conservative to allow centering.
                sigma = std::max(sigma, 0.5);
            } else if (alpha_aff > 0.9) {
                // If affine direction is good, we can reduce mu significantly.
                sigma = std::min(sigma, 0.1);
            }
        }

        if (sigma > 1.0) {
            sigma = 1.0;
        }
        if (sigma < 1e-4) {
            sigma = 1e-4; // Prevent too small sigma.
        }

        double mu_target = sigma * mu_curr;
        if (mu_target < config.mu_final) {
            mu_target = config.mu_final; // Enforce lower bound.
        }
        return mu_target;
    }

    bool compute_mehrotra_direction_(TrajArray& traj)
    {
        // 1. Affine Step (Predictor)
        // Candidate already prepared by prepare_direction_workspace_().
        auto& affine_traj = trajectory.candidate();

        bool aff_success = solve_linear_system_with_retries_(affine_traj, 0.0, nullptr, true);
        if (!aff_success) {
            return false;
        }

        // Calc max step for affine direction.
        // Must cover hard slack (s, lam) AND L1 soft variables (soft_s, w-lam).
        double alpha_aff = compute_fraction_to_boundary_(affine_traj);

        double mu_curr = context_.solve.mu;
        double mu_aff = compute_affine_barrier_mu_(traj, affine_traj, alpha_aff);
        context_.metrics.last_mu_aff = mu_aff;
        context_.metrics.last_alpha_aff = alpha_aff;

        if (config.print_level >= PrintLevel::DEBUG) {
            MLOG_INFO("Mehrotra Debug: mu_curr=" << mu_curr << ", mu_aff=" << mu_aff
                                                 << ", alpha_aff=" << alpha_aff);
        }

        double mu_target = compute_mehrotra_target_mu_(mu_curr, mu_aff, alpha_aff);

        // 2. Corrector Step
        // Solve with mu_target and affine correction term.
        bool solve_success = false;
        if (config.enable_corrector) {
            solve_success = solve_linear_system_with_retries_(traj, mu_target, &affine_traj, false);
        } else {
            // Predictor only: update mu target but do not add correction term.
            solve_success = solve_linear_system_with_retries_(traj, mu_target, nullptr, false);
        }

        if (solve_success) {
            context_.solve.mu = mu_target;
        }
        return solve_success;
    }

    bool compute_direction_linear_solve_(TrajArray& traj)
    {
        if (config.barrier_strategy == BarrierStrategy::MEHROTRA) {
            return compute_mehrotra_direction_(traj);
        }
        return solve_linear_system_with_retries_(traj, context_.solve.mu, nullptr, true);
    }

    SolverStatus validate_search_direction_(
        const TrajArray& traj, bool solve_success, double& max_dual_inf)
    {
        if (!solve_success) {
            return SolverStatus::NUMERICAL_ERROR;
        }

        // Direction check after Riccati: scan the whole valid horizon so a
        // later-stage slack/dual/state NaN cannot hide behind a finite dx0/du0.
        if (has_invalid_search_direction(traj)) {
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_ERROR("Numerical Error: invalid search direction detected.");
            }
            return SolverStatus::NUMERICAL_ERROR;
        }

        // Dual infeasibility metric: use Qu (stored in r_bar after the Riccati backward pass).
        // This is only valid after a successful linear solve (r_bar is stale right after
        // line-search swaps).
        max_dual_inf = compute_dual_infeasibility_(traj);
        return SolverStatus::UNSOLVED;
    }

    SolverStatus compute_search_direction_(TrajArray& traj, double& max_dual_inf)
    {
        timer.start("Linear Solve");
        prepare_direction_workspace_();

        bool solve_success = compute_direction_linear_solve_(traj);

        // Gate reg decay on step quality.
        // The line search's accepted α is a composite proxy — direction
        // quality, barrier fraction-to-boundary, and filter/merit acceptance
        // all feed into it. A small α is necessary (not sufficient) for
        // distrusting the current reg level; decaying on `solve_success`
        // alone drives reg through α-collapse regions and pins it to
        // reg_min. Threshold 0.5 is a small-step heuristic chosen
        // empirically (see .claude/debug/reg-decay-too-aggressive-on-alpha-collapse
        // for traces — case resolved two of three failure metrics with
        // this value; matches informal IPM/SQP tuning convention).
        // Interaction note: feasibility_restoration() does not touch
        // context_.metrics.last_alpha, so after a collapse-triggered restoration the gate
        // stays closed until the next accepted line-search step. That is
        // the conservative behavior we want post-recovery.
        maybe_decay_regularization_after_solve_(solve_success);

        refine_direction_if_enabled_(traj, solve_success);

        timer.stop();

        return validate_search_direction_(traj, solve_success, max_dual_inf);
    }

    void update_barrier(double max_kkt_error, double avg_gap)
    {
        switch (config.barrier_strategy) {
        case BarrierStrategy::MONOTONE:
            if (max_kkt_error < config.barrier_tolerance_factor * context_.solve.mu) {
                double next_mu = std::max(
                    config.mu_final, context_.solve.mu * config.mu_linear_decrease_factor);
                context_.solve.mu = next_mu;
            }
            break;
        case BarrierStrategy::ADAPTIVE: {
            double target = avg_gap * config.mu_safety_margin;
            // Removed forced decrease to allow mu to hold steady if needed for convergence
            context_.solve.mu = std::max(config.mu_final, std::min(context_.solve.mu, target));
            break;
        }
        case BarrierStrategy::MEHROTRA: {
            double ratio = avg_gap / context_.solve.mu;
            if (ratio > 1.0) {
                ratio = 1.0;
            }
            double sigma = std::pow(ratio, 3);
            if (sigma < 0.05) {
                sigma = 0.05;
            }
            if (sigma > 0.8) {
                sigma = 0.8;
            }
            double next_mu = std::max(config.mu_final, context_.solve.mu * sigma);
            context_.solve.mu = next_mu;
            break;
        }
        }
    }

    void print_iteration_log(double alpha, bool header = false)
    {
// Use MLOG_INFO instead of std::cout, respecting the log level.
// We can check MINISOLVER_LOG_LEVEL against MLOG_LEVEL_INFO to avoid formatting cost if disabled.
#if MINISOLVER_LOG_LEVEL < MLOG_LEVEL_INFO
        return;
#endif

        if (config.print_level < PrintLevel::ITER) {
            return;
        }

        std::stringstream ss;
        if (header) {
            ss << std::left << std::setw(5) << "Iter" << std::setw(12) << "Cost" << std::setw(10)
               << "Log(Mu)" << std::setw(10) << "Log(Reg)" << std::setw(10) << "PrimInf"
               << std::setw(10) << "DualInf" << std::setw(10) << "Alpha";

            if (config.print_level >= PrintLevel::DEBUG) {
                ss << std::setw(12) << "MinSlack";
            }
            MLOG_INFO(ss.str());
            MLOG_INFO(std::string(80, '-'));
            return;
        }

        double total_cost = 0.0;
        double max_dual_inf = 0.0;
        double min_slack = 1e9;
        auto& traj = trajectory.active();

        // Use helper for Primal Inf
        double max_prim_inf = compute_max_violation(traj);

        for (int k = 0; k <= N; ++k) {
            const auto& kp = traj[k];
            total_cost += kp.cost;
            for (int i = 0; i < NC; ++i) {
                if (kp.s(i) < min_slack) {
                    min_slack = kp.s(i);
                }
            }
            // q_bar contains Vx (cost-to-go gradient / dynamics multiplier).
            // It is NOT a residual and should not be zero.
            // Only r_bar (control gradient) should be zero.
            double g_norm = 0.0; // MatOps::norm_inf(kp.q_bar);
            double r_norm = MatOps::norm_inf(kp.r_bar);
            double dual = std::max(g_norm, r_norm);
            if (dual > max_dual_inf) {
                max_dual_inf = dual;
            }
        }

        ss << std::left << std::setw(5) << context_.solve.current_iter << std::scientific
           << std::setprecision(3) << std::setw(12) << total_cost << std::fixed
           << std::setprecision(2) << std::setw(10) << std::log10(context_.solve.mu)
           << std::setw(10) << std::log10(context_.solve.reg) << std::scientific
           << std::setprecision(2) << std::setw(10) << max_prim_inf << std::setw(10)
           << max_dual_inf << std::fixed << std::setprecision(3) << std::setw(10) << alpha;

        if (config.print_level >= PrintLevel::DEBUG) {
            ss << std::scientific << std::setprecision(2) << std::setw(12) << min_slack;
        }
        MLOG_INFO(ss.str());
    }

    bool has_nans(const typename TrajectoryType::TrajArray& t) const
    {
        // Bit-level NaN detection (MatOps::has_nan) — works even under -ffast-math.
        // Only checks fields that are sources of NaN: search directions, cost, Jacobians.
        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];
            if (MatOps::has_nan(kp.dx) || MatOps::has_nan(kp.du) || MatOps::has_nan(kp.ds)
                || MatOps::has_nan(kp.dlam) || MatOps::has_nan(kp.dsoft_s)) {
                return true;
            }
            if (!MatOps::is_finite_scalar(kp.cost)) {
                return true;
            }
            // Dynamics outputs — NaN here propagates into defect checks/Riccati.
            if (MatOps::has_nan(kp.f_resid) || MatOps::has_nan(kp.A) || MatOps::has_nan(kp.B)) {
                return true;
            }
        }
        return false;
    }

    bool has_invalid_search_direction(const typename TrajectoryType::TrajArray& t) const
    {
        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];
            for (int i = 0; i < NX; ++i) {
                if (!MatOps::is_finite_scalar(kp.dx(i))) {
                    return true;
                }
            }
            for (int i = 0; i < NU; ++i) {
                if (!MatOps::is_finite_scalar(kp.du(i))) {
                    return true;
                }
            }
            for (int i = 0; i < NC; ++i) {
                if (!MatOps::is_finite_scalar(kp.ds(i))
                    || !MatOps::is_finite_scalar(kp.dlam(i))
                    || !MatOps::is_finite_scalar(kp.dsoft_s(i))) {
                    return true;
                }
            }
        }
        return false;
    }

    bool has_valid_primal_dual_guess(const typename TrajectoryType::TrajArray& t) const
    {
        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];
            if (MatOps::has_nan(kp.x) || MatOps::has_nan(kp.u) || MatOps::has_nan(kp.p)) {
                return false;
            }
            if (MatOps::has_nan(kp.s) || MatOps::has_nan(kp.lam)) {
                return false;
            }

            for (int i = 0; i < NC; ++i) {
                if (kp.s(i) <= 0.0 || kp.lam(i) <= 0.0) {
                    return false;
                }

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                if (type == 1 && w > 1e-6) {
                    if (!MatOps::is_finite_scalar(kp.soft_s(i)) || kp.soft_s(i) <= 0.0) {
                        return false;
                    }
                    if (w - kp.lam(i) <= config.min_barrier_slack) {
                        return false;
                    }
                }
            }
        }
        return !has_nans(t);
    }

    bool feasibility_restoration()
    {
        if (config.print_level >= PrintLevel::DEBUG) {
            MLOG_DEBUG("Entering Feasibility Restoration Phase.");
        }
        double saved_mu = context_.solve.mu;
        double saved_reg = context_.solve.reg;
        context_.solve.mu = config.restoration_mu;
        context_.solve.reg = config.restoration_reg;

        auto& traj = trajectory.active();
        bool success = false;

        for (int r_iter = 0; r_iter < config.max_restoration_iters; ++r_iter) {
            for (int k = 0; k <= N; ++k) {
                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N);

                traj[k].Q.setIdentity();
                traj[k].q.setZero();
                traj[k].H.setZero();

                if (k < N) {
                    traj[k].R.setIdentity();
                    traj[k].r.setZero();
                } else {
                    traj[k].R.setZero();
                    traj[k].r.setZero();
                }
            }

            // Restoration linear solve
            // [ALADIN-Inspired] Augmented Lagrangian Restoration
            // Minimizing 0.5*||dx||^2 + 0.5*rho*||C*dx + D*du + g + s||^2
            // This pulls the solution towards the constraint manifold more aggressively than simple
            // min-norm.
            if (config.barrier_strategy != BarrierStrategy::MEHROTRA) {
                double rho = 1000.0; // Penalty weight from ALADIN concepts
                for (int k = 0; k <= N; ++k) {
                    auto& kp = traj[k];

                    // Q += rho * C^T * C
                    kp.Q.noalias() += rho * kp.C.transpose() * kp.C;

                    // R += rho * D^T * D
                    kp.R.noalias() += rho * kp.D.transpose() * kp.D;

                    // H += rho * D^T * C (Cross term)
                    kp.H.noalias() += rho * kp.D.transpose() * kp.C;

                    // q += rho * C^T * g_val
                    // Note: Restoration usually ignores 's' in the quadratic penalty approximation
                    // or treats it as fixed residuals g_val.
                    kp.q.noalias() += rho * kp.C.transpose() * kp.g_val;

                    // r += rho * D^T * g_val
                    kp.r.noalias() += rho * kp.D.transpose() * kp.g_val;
                }
            }

            if (!linear_solver->solve(traj, N, context_.solve.mu, context_.solve.reg,
                    config.inertia_strategy, config)) {
                break;
            }

            double alpha = 1.0;
            for (int k = 0; k <= N; ++k) {
                for (int i = 0; i < NC; ++i) {
                    double s = traj[k].s(i);
                    double ds = traj[k].ds(i);
                    double lam = traj[k].lam(i);
                    double dlam = traj[k].dlam(i);
                    if (ds < 0) {
                        alpha = std::min(alpha, -config.restoration_alpha * s / ds);
                    }
                    if (dlam < 0) {
                        alpha = std::min(alpha, -config.restoration_alpha * lam / dlam);
                    }

                    // For L1 soft constraints, maintain:
                    // - soft_s > 0
                    // - w - lam > 0  (implicit soft-dual)
                    double w = 0.0;
                    int type = 0;
                    if constexpr (NC > 0) {
                        if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                            type = Model::constraint_types[i];
                            w = Model::constraint_weights[i];
                        }
                    }
                    if (type == 1 && w > 1e-6) {
                        const double soft_s = traj[k].soft_s(i);
                        const double dsoft_s = traj[k].dsoft_s(i);
                        if (dsoft_s < 0) {
                            alpha = std::min(alpha, -config.restoration_alpha * soft_s / dsoft_s);
                        }

                        if (dlam > 0) {
                            const double gap = (w - lam) - config.min_barrier_slack;
                            if (gap <= 0.0) {
                                alpha = 0.0;
                            } else {
                                alpha = std::min(alpha, config.restoration_alpha * gap / dlam);
                            }
                        }
                    }
                }
            }

            if (alpha < 1e-4) {
                break;
            }

            for (int k = 0; k <= N; ++k) {
                traj[k].x += alpha * traj[k].dx;
                if (k < N) {
                    traj[k].u += alpha * traj[k].du;
                } else {
                    traj[k].u.setZero();
                }
                traj[k].s += alpha * traj[k].ds;
                traj[k].lam += alpha * traj[k].dlam;
                traj[k].soft_s += alpha * traj[k].dsoft_s;
            }
            success = true; // At least one restoration step was applied
        }

        // Reset Lagrange Multipliers for the original problem to avoid dual contamination
        // form the restoration phase (which solves a different problem).
        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NC; ++i) {
                // Ensure s is positive
                if (traj[k].s(i) < 1e-9) {
                    traj[k].s(i) = 1e-9;
                }

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                // Preserve dual info from restoration if valuable, but ensure complementarity lower
                // bound.
                double lam_min = saved_mu / traj[k].s(i);

                if (type == 1 && w > 1e-6) {
                    // Keep lam strictly inside the dual box so barrier terms remain well-defined.
                    double lam_max = w - config.min_barrier_slack;
                    if (lam_max < config.min_barrier_slack) {
                        lam_max = config.min_barrier_slack;
                    }
                    traj[k].lam(i) = std::min(traj[k].lam(i), lam_max);
                    traj[k].lam(i) = std::max(traj[k].lam(i), lam_min);

                    // Rebuild the soft slack on the central path for the restored mu.
                    traj[k].soft_s(i)
                        = std::max(config.min_barrier_slack, saved_mu / (w - traj[k].lam(i)));
                } else {
                    traj[k].lam(i) = std::max(traj[k].lam(i), lam_min);
                }
            }
        }

        context_.solve.mu = saved_mu;
        context_.solve.reg = saved_reg;

        return success;
    }

public:
    SolverStatus solve()
    {
        begin_solve_();
        SolverStatus loop_exit_status = run_solve_loop_();

        // 3. Postsolve: 扫尾、刷新导数、统一评级
        return postsolve(loop_exit_status);
    }

private:
    void begin_solve_()
    {
        ensure_solver_components_ready();
        // 1. Presolve: 数据准备、冷热启动处理、内存复位
        presolve();

        // Diagnostic line-search trace — fresh per solve. Reserve once at the
        // configured max iteration count so the hot-path push_back stays
        // pointer-bump only.
        alpha_log_.clear();
        context_.metrics.reset_solve();
    }

    bool should_exit_after_step_status_(
        SolverStatus step_stat, int iter, SolverStatus& loop_exit_status) const
    {
        // SQP-RTI 模式：做一步即走，状态由 postsolve 判定。
        // Preserve the existing priority: RTI exits before inspecting step_stat.
        if (config.enable_rti) {
            loop_exit_status = SolverStatus::UNSOLVED;
            return true;
        }

        // 完美收敛 -> 记录状态并跳出，交给 postsolve 复核。
        if (step_stat == SolverStatus::OPTIMAL) {
            loop_exit_status = SolverStatus::OPTIMAL;
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_INFO("Converged in " << iter + 1 << " iterations.");
            }
            return true;
        }

        if (step_stat == SolverStatus::FEASIBLE || step_stat == SolverStatus::INFEASIBLE) {
            loop_exit_status = step_stat;
            return true;
        }

        // 数值错误 -> 立即中止。
        if (step_stat == SolverStatus::NUMERICAL_ERROR) {
            loop_exit_status = SolverStatus::NUMERICAL_ERROR;
            return true;
        }

        return false;
    }

    bool should_exit_for_cost_stagnation_(
        double& last_cost, double& last_mu, SolverStatus& loop_exit_status)
    {
        // Objective cost is comparable across μ updates (barrier terms are not part of
        // KnotPoint::cost). However, if μ is actively decreasing we should not stop purely
        // on objective stagnation, since IPM may still be progressing by reducing μ.
        // The intended use is to catch "μ frozen above μ_final" style stalls.
        double current_cost = compute_objective_cost_(trajectory.active());

        if (should_stop_for_cost_stagnation_(current_cost, last_cost, last_mu)) {
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_INFO("Cost Stagnation detected. Stopping early.");
            }
            // 标记为 UNSOLVED，表示非自然收敛，交给 postsolve 判定是 Feasible 还是 Optimal。
            loop_exit_status = SolverStatus::UNSOLVED;
            return true;
        }

        last_cost = current_cost;
        last_mu = context_.solve.mu;
        return false;
    }

    SolverStatus run_solve_loop_()
    {
        SolverStatus loop_exit_status = SolverStatus::UNSOLVED;
        double last_cost = 1e30;
        double last_mu = context_.solve.mu;

        // 2. Solve Loop: 数值迭代
        for (int iter = 0; iter < config.max_iters; ++iter) {
            timer.start("Total Step");
            SolverStatus step_stat = execute_solve_iteration_();
            timer.stop();

            if (should_exit_after_step_status_(step_stat, iter, loop_exit_status)) {
                break;
            }

            if (should_exit_for_cost_stagnation_(last_cost, last_mu, loop_exit_status)) {
                break;
            }
        }

        return loop_exit_status;
    }

    SolverStatus execute_solve_iteration_()
    {
        context_.solve.current_iter++;
        // Snapshot μ at the top of the step so we can notify the line search
        // (IPOPT §3.1: filter/merit history is not comparable across μ changes).
        // Both the non-Mehrotra update_barrier() path and the Mehrotra
        // mu = mu_target path mutate mu between here and the line-search call
        // below; the single comparison covers both.
        const double mu_before_step = context_.solve.mu;

        auto& traj = trajectory.active();

        StepResidualSummary residuals = evaluate_derivatives_phase_(traj);
        double max_kkt_error = residuals.max_kkt_error;
        double max_prim_inf = residuals.max_prim_inf;
        double max_dual_inf = 0.0;
        update_barrier_for_step_(residuals);

        SolverStatus direction_status = compute_search_direction_(traj, max_dual_inf);
        if (direction_status != SolverStatus::UNSOLVED) {
            return direction_status;
        }

        // Convergence check (Primal + Dual + Mu) using the freshly computed dual residual.
        // The final convergence verdict is always made in postsolve() with fresh data.
        if (context_.solve.current_iter > 1
            && check_convergence(max_prim_inf, max_dual_inf, max_kkt_error)) {
            return SolverStatus::OPTIMAL;
        }

        double alpha = 1.0;
        SolverStatus globalization_status
            = globalize_step_(mu_before_step, max_prim_inf, max_dual_inf, alpha);
        if (globalization_status != SolverStatus::UNSOLVED) {
            return globalization_status;
        }

        // Final Convergence Check using Step Size and Residuals.
        // Avoids wasting a full derivative computation in next step if we are already done.
        if (should_stop_after_line_search_(trajectory.active(), alpha, max_prim_inf, max_dual_inf)) {
            return SolverStatus::OPTIMAL;
        }

        update_metrics_after_globalization_(max_dual_inf);
        return SolverStatus::UNSOLVED;
    }

    // ============================================================
    // [Phase 1] Presolve: Preparation
    // ============================================================
    void reset_solve_runtime_state_()
    {
        // [Enable/Disable Profiling]
        timer.enabled = config.enable_profiling;
        if (line_search) {
            line_search->reset();
        }

        context_.solve.reset_algorithmic(config.mu_init, config.reg_init);
    }

    bool should_initialize_primal_dual_() const
    {
        if (config.initialization != InitializationMode::REUSE_PRIMAL_DUAL) {
            return true;
        }
        return !has_valid_primal_dual_guess(trajectory.active());
    }

    void initialize_constraint_primal_dual_(Knot& kp, int i)
    {
        double g = kp.g_val(i);
        double w = 0.0;
        int type = 0;
        if constexpr (NC > 0) {
            if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                type = Model::constraint_types[i];
                w = Model::constraint_weights[i];
            }
        }

        if (type == 1 && w > 1e-6) { // L1 Soft Constraint
            // Central Path:
            // 1) g + s - soft_s = 0
            // 2) s * lam = mu
            // 3) soft_s * (w - lam) = mu
            // Reduce to quadratic in lam: g*lam^2 - (g*w - 2*mu)*lam - mu*w = 0
            double a = g;
            double b = -(g * w - 2 * context_.solve.mu);
            double c = -context_.solve.mu * w;

            double lam_val;
            if (std::abs(a) < 1e-9) {
                // Linear case (g ≈ 0): 2*mu*lam = mu*w → lam = w/2
                lam_val = w / 2.0;
            } else {
                double delta = b * b - 4 * a * c;
                if (delta < 0) {
                    delta = 0;
                }
                lam_val = (-b + std::sqrt(delta)) / (2 * a);
            }

            // Clamp for safety
            lam_val = std::max(1e-8, std::min(w - 1e-8, lam_val));

            kp.lam(i) = lam_val;
            kp.s(i) = context_.solve.mu / lam_val;
            kp.soft_s(i) = context_.solve.mu / (w - lam_val);
        } else if (type == 2 && w > 1e-6) { // L2 Soft Constraint
            // Central Path:
            // 1) g + s - lam/w = 0
            // 2) s * lam = mu
            // Reduce to quadratic in lam: lam^2 - g*w*lam - mu*w = 0
            double b = -g * w;
            double c = -context_.solve.mu * w;
            // lam = (-b + sqrt(b^2 - 4ac)) / 2a, here a=1
            // lam = (g*w + sqrt(g^2*w^2 + 4*mu*w)) / 2
            double delta = b * b - 4 * c; // b^2 + 4*mu*w > 0 always
            double lam_val = (-b + std::sqrt(delta)) / 2.0;

            kp.lam(i) = std::max(1e-8, lam_val);
            kp.s(i) = context_.solve.mu / kp.lam(i);
            // soft_s not used in L2
        } else { // Hard Constraint
            double s_val = std::max(1e-6, -g);
            kp.s(i) = s_val;
            kp.lam(i) = context_.solve.mu / s_val;
        }
    }

    void initialize_primal_dual_from_model_()
    {
        auto& traj = trajectory.active();
        for (int k = 0; k <= N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;
            detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N);

            for (int i = 0; i < NC; ++i) {
                initialize_constraint_primal_dual_(traj[k], i);
            }
        }
    }

    void presolve()
    {
        reset_solve_runtime_state_();

        if (should_initialize_primal_dual_()) {
            initialize_primal_dual_from_model_();
        }

        print_iteration_log(0.0, true);
        timer.reset();
    }

    // ============================================================
    // [Phase 3] Postsolve: Finalization & Verdict
    // ============================================================
    bool refresh_postsolve_residuals_(TrajArray& traj, double& max_kkt_error)
    {
        for (int k = 0; k <= N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;

            detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N);

            // 2. Check NaNs (bit-level, works under -ffast-math)
            for (int i = 0; i < NC; ++i) {
                if (MatOps::is_nan_scalar(traj[k].g_val(i))
                    || MatOps::is_nan_scalar(traj[k].s(i))) {
                    return false;
                }
            }

            // 3. KKT complementarity (including L1-soft secondary pair).
            for (int i = 0; i < NC; ++i) {
                double comp = std::abs(traj[k].s(i) * traj[k].lam(i) - context_.solve.mu);
                if (comp > max_kkt_error) {
                    max_kkt_error = comp;
                }

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                if (type == 1 && w > 1e-6) {
                    const double soft_s = traj[k].soft_s(i);
                    const double soft_dual = (w - traj[k].lam(i));
                    double comp_soft
                        = std::abs(soft_s * soft_dual - context_.solve.mu);
                    if (comp_soft > max_kkt_error) {
                        max_kkt_error = comp_soft;
                    }
                }
            }
        }

        return true;
    }

    bool evaluate_postsolve_dual_residual_(double& max_dual_inf)
    {
        // Dual infeasibility metric: require a fresh Riccati backward pass so Qu includes
        // the dynamic multipliers (B^T * pi_{k+1}). Use the inactive trajectory buffer as
        // scratch so postsolve never overwrites the active solution directions/gains.
        max_dual_inf = std::numeric_limits<double>::infinity();
        if (linear_solver) {
            trajectory.prepare_candidate_full();
            return linear_solver->evaluate_dual_residual(trajectory.candidate(), N,
                context_.solve.mu, context_.solve.reg, config.inertia_strategy, config,
                max_dual_inf);
        }
        return false;
    }

    SolverStatus classify_postsolve_result_(
        double max_viol, double max_dual_inf, double max_kkt_error, bool linear_ok)
    {
        // Level 1: SOLVED (Optimal)
        // 即使 Loop 是因为 Stagnation 退出的，如果此时恰好满足最优性，也给 SOLVED
        if (linear_ok && check_convergence(max_viol, max_dual_inf, max_kkt_error)) {
            return SolverStatus::OPTIMAL;
        }
        // Level 2: FEASIBLE (Acceptable)
        double feasible_bound = config.tol_con * config.feasible_tol_scale;
        if (max_viol <= feasible_bound) {
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_INFO(
                    "Result: FEASIBLE (Viol: " << max_viol << " <= " << feasible_bound << ")");
            }
            return SolverStatus::FEASIBLE;
        }
        // Level 3: INFEASIBLE (Failed)
        if (config.print_level >= PrintLevel::WARN) {
            MLOG_WARN("Result: INFEASIBLE (Viol: " << max_viol << " > " << feasible_bound << ")");
        }
        return SolverStatus::INFEASIBLE;
    }

    SolverStatus postsolve(SolverStatus loop_status)
    {
        if (loop_status == SolverStatus::NUMERICAL_ERROR) {
            return SolverStatus::NUMERICAL_ERROR;
        }
        if (config.print_level >= PrintLevel::INFO) {
            if (loop_status == SolverStatus::UNSOLVED) {
                MLOG_INFO("Max iterations or stagnation.");
            }
        }
        // [Fix 2 Logic] 强制刷新导数
        // 无论是因为收敛还是因为耗尽步数退出，我们都重新计算一次精确的残差，
        // 以便做出最公正的最终评判。
        auto& traj = trajectory.active();
        double max_kkt_error = 0.0;
        if (!refresh_postsolve_residuals_(traj, max_kkt_error)) {
            return SolverStatus::NUMERICAL_ERROR;
        }

        // Primal feasibility uses the recomputed constraints/dynamics defects.
        double max_viol = compute_max_violation(traj);

        double max_dual_inf = std::numeric_limits<double>::infinity();
        bool linear_ok = evaluate_postsolve_dual_residual_(max_dual_inf);

        // [最终评级]
        return classify_postsolve_result_(max_viol, max_dual_inf, max_kkt_error, linear_ok);
    }

private:
    // Slack reset kernel, extracted from step() so unit tests can exercise it
    // without running the full line-search / linear-solve pipeline.
    // For every knot/constraint with s(i) < |g_val(i)| + sqrt(mu), raise s to
    // that floor and pull the matching dual onto the central path. The L1
    // soft-constraint coupling (lam <= w and soft_s = mu/(w-lam)) is preserved.
    void apply_slack_reset_(TrajArray& traj)
    {
        for (int k = 0; k <= N; ++k) {
            auto& kp = traj[k];
            for (int i = 0; i < NC; ++i) {
                double min_s = std::abs(kp.g_val(i)) + std::sqrt(context_.solve.mu);
                if (kp.s(i) < min_s) {
                    kp.s(i) = min_s;
                    // Pull the dual onto the central path at the CURRENT barrier
                    // parameter mu. Using config.mu_init here would pump lam up
                    // to the initial barrier scale after mu has already decayed,
                    // breaking KKT complementarity (|s*lam - mu| ~ mu_init).
                    kp.lam(i) = std::max(kp.lam(i), context_.solve.mu / kp.s(i));

                    // For L1 soft constraints, also keep the implicit soft-dual feasible:
                    // (w - lam) > 0.
                    double w = 0.0;
                    int type = 0;
                    if constexpr (NC > 0) {
                        if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                            type = Model::constraint_types[i];
                            w = Model::constraint_weights[i];
                        }
                    }
                    if (type == 1 && w > 1e-6) {
                        double lam_max = w - config.min_barrier_slack;
                        if (lam_max < config.min_barrier_slack) {
                            lam_max = config.min_barrier_slack;
                        }
                        if (kp.lam(i) > lam_max) {
                            kp.lam(i) = lam_max;
                        }
                        // Keep soft slack on the central path so barrier terms remain
                        // well-defined.
                        kp.soft_s(i) = std::max(
                            config.min_barrier_slack, context_.solve.mu / (w - kp.lam(i)));
                    }
                }
            }
        }
    }

    // helper function: calculate the maximum constraint violation of the current trajectory
    // Includes both inequality constraint residuals AND dynamics defects (multiple shooting).
    double compute_max_violation(const TrajArray& traj) const
    {
        double max_viol = 0.0;
        for (int k = 0; k <= N; ++k) {
            const auto& kp = traj[k];

            // 1. Inequality constraint violation
            for (int i = 0; i < NC; ++i) {
                double viol = 0.0;

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }
                if (type == 1 && w > 1e-6) { // L1
                    viol = std::abs(kp.g_val(i) + kp.s(i) - kp.soft_s(i));
                } else if (type == 2 && w > 1e-6) { // L2
                    viol = std::abs(kp.g_val(i) + kp.s(i) - kp.lam(i) / w);
                } else { // Hard
                    viol = std::abs(kp.g_val(i) + kp.s(i));
                }
                if (viol > max_viol) {
                    max_viol = viol;
                }
            }

            // 2. Dynamics defect (multiple shooting): x_{k+1} - f(x_k, u_k)
            if (k < N) {
                for (int j = 0; j < NX; ++j) {
                    double defect = std::abs(traj[k + 1].x(j) - kp.f_resid(j));
                    if (defect > max_viol) {
                        max_viol = defect;
                    }
                }
            }
        }
        return max_viol;
    }

    void rebuild_solver_components()
    {
        linear_solver = std::make_unique<RiccatiSolver<TrajArray, Model>>();

        if (config.line_search_type == LineSearchType::NONE) {
            line_search = std::make_unique<NoLineSearch<Model, MAX_N>>();
        } else if (config.line_search_type == LineSearchType::MERIT) {
            line_search = std::make_unique<MeritLineSearch<Model, MAX_N>>();
        } else {
            line_search = std::make_unique<FilterLineSearch<Model, MAX_N>>();
        }
    }

    void ensure_solver_components_ready()
    {
        if (!components_dirty) {
            return;
        }
        rebuild_solver_components();
        components_dirty = false;
    }

    // Lookup Maps
    std::unordered_map<std::string, int> state_map;
    std::unordered_map<std::string, int> control_map;
    std::unordered_map<std::string, int> param_map;

    // Components
    // Pass Model type to RiccatiSolver for static constraint info access
    std::unique_ptr<RiccatiSolver<TrajArray, Model>> linear_solver;
    std::unique_ptr<LineSearchStrategy<Model, MAX_N>> line_search;

    // Per-solve line-search α trace (cleared at solve() entry, appended after
    // each step's line search). Purely diagnostic — not serialized.
    std::vector<double> alpha_log_;

    SolverContext context_;

    TrajectoryType trajectory;

    int N;
    std::array<double, MAX_N> dt_traj;

    SolverTimer timer;

    SolverConfig config;
    bool components_dirty = false;
};
}
