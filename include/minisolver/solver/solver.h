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

#include "minisolver/core/config_validation.h"
#include "minisolver/core/constraint_semantics.h"
#include "minisolver/core/logger.h"
#include "minisolver/core/model_traits.h"
#include "minisolver/core/solver_context.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/solver_plan.h"
#include "minisolver/core/trajectory.h"
#include "minisolver/core/types.h"

#include "minisolver/algorithms/barrier_update.h"
#include "minisolver/algorithms/initialization.h"
#include "minisolver/algorithms/line_search.h"
#include "minisolver/algorithms/linear_solver.h"
#include "minisolver/algorithms/model_evaluation.h"
#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/algorithms/termination.h"
#include "minisolver/integrator/implicit_integrator.h"

#include "minisolver/backend/backend_interface.h"

namespace minisolver {

template <typename Model, int MAX_N> class SolverSnapshotIO;

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

    friend class SolverSnapshotIO<Model, MAX_N>;
    template <typename, int> friend struct ::minisolver::test::SolverInternalAccess;

    MiniSolver(int initial_N, Backend b, SolverConfig conf = SolverConfig())
        : trajectory(validate_horizon_or_throw_(initial_N))
        , N(validate_horizon_or_throw_(initial_N))
        , config(conf)
    {
        // Constructor has an explicit backend argument; keep it as the source of truth.
        config.backend = b;
        if (detail::validate_solver_config(config) != ApiStatus::OK) {
            throw std::invalid_argument("MiniSolver constructed with invalid SolverConfig");
        }
        context_.reset_algorithmic(config.mu_init, config.reg_init);
        rti_lite_last_x0_.setZero();

        dt_traj.fill(conf.default_dt);
        rebuild_solver_components_if_dirty_();

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

    ApiStatus resize_horizon(int new_n)
    {
        if (new_n < 0 || new_n > MAX_N) {
            MLOG_ERROR("new_n outside valid range [0, MAX_N]");
            return ApiStatus::InvalidHorizon;
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
        return ApiStatus::OK;
    }

    int get_horizon() const { return N; }

    ApiStatus set_config(const SolverConfig& conf)
    {
        // Backend is fixed at construction time (see ctor note: "explicit
        // backend argument; keep it as the source of truth"). Preserve it
        // across set_config so a caller passing a default-constructed conf
        // doesn't silently switch backends.
        const Backend preserved_backend = config.backend;
        SolverConfig candidate = conf;
        candidate.backend = preserved_backend;
        const ApiStatus validation = detail::validate_solver_config(candidate);
        if (validation != ApiStatus::OK) {
            return validation;
        }
        config = candidate;
        if (config.max_iters > 0 && static_cast<int>(alpha_log_.capacity()) < config.max_iters) {
            alpha_log_.reserve(static_cast<size_t>(config.max_iters));
        }
        build_state_.dirty = true;
        // A new config can change strategies wholesale (line search, barrier,
        // tolerances), so the previous solve's primal-dual iterate is no
        // longer a safe RTI-lite seed. Drop the RTI-lite history.
        rti_lite_have_previous_solve_ = false;
        rti_lite_last_solve_acceptable_ = false;
        rti_lite_linearization_age_ = 0;
        return ApiStatus::OK;
    }

    const SolverConfig& get_config() const { return config; }

    int get_iteration_count() const { return context_.solve.current_iter; }

    const SolverInfo& get_info() const { return context_.info; }

    double get_profile_time_ms(const std::string& name) const
    {
        auto it = timer.times.find(name);
        return it != timer.times.end() ? it->second : 0.0;
    }

    // --- High-Level API ---

    // 1. Initial State
    ApiStatus set_initial_state(const std::vector<double>& x0)
    {
        if (x0.size() != NX) {
            return ApiStatus::SizeMismatch;
        }
        for (double value : x0) {
            if (!std::isfinite(value)) {
                return ApiStatus::NonFiniteValue;
            }
        }
        auto& kp = trajectory[0];
        for (int i = 0; i < NX; ++i) {
            kp.x(i) = x0[i];
        }
        return ApiStatus::OK;
    }

    ApiStatus set_initial_state(const std::string& name, double value)
    {
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        int idx = get_state_idx(name);
        if (idx != -1) {
            trajectory[0].x(idx) = value;
            return ApiStatus::OK;
        } else {
            MLOG_WARN("Unknown state " << name);
            return ApiStatus::UnknownName;
        }
    }

    // 2. Parameters
    ApiStatus set_parameter(int stage, int idx, double value)
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NP || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        trajectory[stage].p(idx) = value;
        return ApiStatus::OK;
    }

    ApiStatus set_parameter(int stage, const std::string& name, double value)
    {
        int idx = get_param_idx(name);
        if (idx != -1) {
            return set_parameter(stage, idx, value);
        } else {
            MLOG_WARN("Unknown param " << name);
            return ApiStatus::UnknownName;
        }
    }

    ApiStatus set_global_parameter(int idx, double value)
    {
        if (idx >= NP || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        for (int k = 0; k <= N; ++k) {
            trajectory[k].p(idx) = value;
        }
        return ApiStatus::OK;
    }

    ApiStatus set_global_parameter(const std::string& name, double value)
    {
        int idx = get_param_idx(name);
        if (idx != -1) {
            return set_global_parameter(idx, value);
        } else {
            MLOG_WARN("Unknown param " << name);
            return ApiStatus::UnknownName;
        }
    }

    double get_parameter(int stage, int idx) const
    {
        if (stage > N || stage < 0 || idx >= NP || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].p(idx);
    }

    ApiStatus get_parameter(int stage, int idx, double& out) const
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NP || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        out = trajectory[stage].p(idx);
        return ApiStatus::OK;
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
    ApiStatus set_state_guess(int stage, int idx, double value)
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NX || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        trajectory[stage].x(idx) = value;
        return ApiStatus::OK;
    }

    ApiStatus set_state_guess(int stage, const std::string& name, double value)
    {
        int idx = get_state_idx(name);
        if (idx != -1) {
            return set_state_guess(stage, idx, value);
        }
        MLOG_WARN("Unknown state " << name);
        return ApiStatus::UnknownName;
    }

    ApiStatus set_state_guess_traj(const std::string& name, const std::vector<double>& values)
    {
        int idx = get_state_idx(name);
        if (idx == -1) {
            MLOG_WARN("Unknown state " << name);
            return ApiStatus::UnknownName;
        }
        int count = std::min((int)values.size(), N + 1);
        for (int k = 0; k < count; ++k) {
            if (!std::isfinite(values[k])) {
                return ApiStatus::NonFiniteValue;
            }
        }
        for (int k = 0; k < count; ++k) {
            trajectory[k].x(idx) = values[k];
        }
        return ApiStatus::OK;
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

    ApiStatus get_state(int stage, int idx, double& out) const
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NX || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        out = trajectory[stage].x(idx);
        return ApiStatus::OK;
    }

    // 4. Control Access
    ApiStatus set_control_guess(int stage, int idx, double value)
    {
        if (stage == N) {
            return ApiStatus::TerminalControl;
        }
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NU || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        trajectory[stage].u(idx) = value;
        return ApiStatus::OK;
    }

    ApiStatus set_control_guess(int stage, const std::string& name, double value)
    {
        int idx = get_control_idx(name);
        if (idx != -1) {
            return set_control_guess(stage, idx, value);
        }
        MLOG_WARN("Unknown control " << name);
        return ApiStatus::UnknownName;
    }

    ApiStatus set_control_guess_traj(const std::string& name, const std::vector<double>& values)
    {
        int idx = get_control_idx(name);
        if (idx == -1) {
            MLOG_WARN("Unknown control " << name);
            return ApiStatus::UnknownName;
        }
        int count = std::min((int)values.size(), N);
        for (int k = 0; k < count; ++k) {
            if (!std::isfinite(values[k])) {
                return ApiStatus::NonFiniteValue;
            }
        }
        for (int k = 0; k < count; ++k) {
            trajectory[k].u(idx) = values[k];
        }
        return ApiStatus::OK;
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

    ApiStatus get_control(int stage, int idx, double& out) const
    {
        if (stage == N) {
            return ApiStatus::TerminalControl;
        }
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NU || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        out = trajectory[stage].u(idx);
        return ApiStatus::OK;
    }

    // 5. Slack / Dual Access
    ApiStatus set_slack_guess(int stage, int idx, double value)
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NC || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        trajectory[stage].s(idx) = value;
        return ApiStatus::OK;
    }

    ApiStatus set_dual_guess(int stage, int idx, double value)
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NC || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        trajectory[stage].lam(idx) = value;
        return ApiStatus::OK;
    }

    // Set the L1 soft-constraint slack (`soft_s`) at stage k for constraint idx.
    // Use this when warm-starting an L1 soft-constrained model where the user
    // wants to seed the soft slack alongside the hard slack/dual guesses. For L2
    // soft constraints and hard constraints this setter still validates inputs
    // but the solver does not consume `soft_s` for those cases.
    ApiStatus set_soft_slack_guess(int stage, int idx, double value)
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NC || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value)) {
            return ApiStatus::NonFiniteValue;
        }
        trajectory[stage].soft_s(idx) = value;
        return ApiStatus::OK;
    }

    // Warm-start aliases. These document intent at the call site for repeated
    // MPC solves that reuse the previous primal-dual iterate; the underlying
    // semantics are identical to `set_slack_guess` / `set_dual_guess` /
    // `set_soft_slack_guess`.
    ApiStatus set_warm_start_slack(int stage, int idx, double value)
    {
        return set_slack_guess(stage, idx, value);
    }

    ApiStatus set_warm_start_dual(int stage, int idx, double value)
    {
        return set_dual_guess(stage, idx, value);
    }

    ApiStatus set_warm_start_soft_slack(int stage, int idx, double value)
    {
        return set_soft_slack_guess(stage, idx, value);
    }

    // -----------------------------------------------------------------------
    // Coordinate-scaling hint API (Stage 5 minimal viable).
    //
    // The scale factors describe the *typical magnitude* of the corresponding
    // control coordinate in user units. They never rescale state/control/
    // parameter values returned by getters, never alter the search direction,
    // and never feed back into Riccati/SOC/restoration. They are consumed
    // exclusively by the dual-stationarity termination metric when
    // `config.coordinate_scaling == CoordinateScalingMethod::USER_SUPPLIED`.
    //
    // Why control-only: dual stationarity is reported via the inf-norm of the
    // Riccati-projected control residual `r_bar`. State stationarity is
    // eliminated by the Riccati substitution (it is implicitly satisfied by
    // the linearised KKT) and parameters are not optimisation variables, so
    // there is no `r_bar`-style residual on those axes. Earlier revisions
    // exposed `set_state_scale` / `set_parameter_scale` that stored values
    // but never affected `dual_inf`; they were a silent no-op while
    // `coordinate_scaling_active` reported `true`. They have been removed
    // until a state/parameter-aware termination metric exists.
    //
    // Validation: each scale must be finite and inside
    // `[config.coordinate_scale_min, config.coordinate_scale_max]`. The
    // defaults are 1.0 so callers that never touch this API keep the prior
    // numerical contract bit-for-bit.
    // -----------------------------------------------------------------------

    ApiStatus set_control_scale(int idx, double value)
    {
        if (idx < 0 || idx >= NU) {
            return ApiStatus::InvalidIndex;
        }
        if (!std::isfinite(value) || value < config.coordinate_scale_min
            || value > config.coordinate_scale_max) {
            return ApiStatus::InvalidArgument;
        }
        control_coord_scale_[idx] = value;
        return ApiStatus::OK;
    }

    ApiStatus set_control_scale(const std::string& name, double value)
    {
        const int idx = get_control_idx(name);
        if (idx < 0) {
            return ApiStatus::UnknownName;
        }
        return set_control_scale(idx, value);
    }

    void reset_coordinate_scaling() { control_coord_scale_.fill(1.0); }

    double get_control_scale(int idx) const
    {
        if (idx < 0 || idx >= NU) {
            return 1.0;
        }
        return control_coord_scale_[idx];
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

    ApiStatus get_slack(int stage, int idx, double& out) const
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NC || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        out = trajectory[stage].s(idx);
        return ApiStatus::OK;
    }

    double get_dual(int stage, int idx) const
    {
        if (stage > N || stage < 0 || idx >= NC || idx < 0) {
            return 0.0;
        }
        return trajectory[stage].lam(idx);
    }

    ApiStatus get_dual(int stage, int idx, double& out) const
    {
        if (stage > N || stage < 0) {
            return ApiStatus::InvalidStage;
        }
        if (idx >= NC || idx < 0) {
            return ApiStatus::InvalidIndex;
        }
        out = trajectory[stage].lam(idx);
        return ApiStatus::OK;
    }

    // 6. Cost Access
    double get_stage_cost(int stage) const
    {
        if (stage > N || stage < 0) {
            return 0.0;
        }
        return trajectory[stage].cost_unscaled;
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
        return detail::true_constraint_value<Model>(trajectory[stage], idx);
    }

    ApiStatus set_dt(const std::vector<double>& dts)
    {
        if (dts.size() > MAX_N) {
            MLOG_WARN("DT vector too large.");
            return ApiStatus::SizeMismatch;
        }
        for (double dt : dts) {
            if (!std::isfinite(dt)) {
                return ApiStatus::NonFiniteValue;
            }
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
        return ApiStatus::OK;
    }

    ApiStatus set_dt(double dt)
    {
        if (!std::isfinite(dt)) {
            return ApiStatus::NonFiniteValue;
        }
        dt_traj.fill(dt);
        return ApiStatus::OK;
    }

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
    ApiStatus restore_config_from_snapshot_(const SolverConfig& restored_config)
    {
        const ApiStatus validation = detail::validate_solver_config(restored_config);
        if (validation != ApiStatus::OK) {
            return validation;
        }
        config = restored_config;
        if (config.max_iters > 0 && static_cast<int>(alpha_log_.capacity()) < config.max_iters) {
            alpha_log_.reserve(static_cast<size_t>(config.max_iters));
        }
        build_state_.dirty = true;
        return ApiStatus::OK;
    }

    TerminationReason reason_for_loop_status_(SolverStatus status) const
    {
        switch (status) {
        case SolverStatus::OPTIMAL:
            return TerminationReason::CONVERGED;
        case SolverStatus::FEASIBLE:
            return TerminationReason::PRIMAL_FEASIBLE;
        case SolverStatus::MAX_ITER:
        case SolverStatus::UNSOLVED:
            return TerminationReason::MAX_ITERATIONS;
        case SolverStatus::STEP_TOO_SMALL:
            return TerminationReason::LINE_SEARCH_FAILED;
        case SolverStatus::INSUFFICIENT_PROGRESS:
            return TerminationReason::COST_STAGNATION;
        case SolverStatus::LINEAR_SOLVE_FAILED:
            return TerminationReason::LINEAR_SOLVE_FAILED;
        case SolverStatus::RESTORATION_FAILED:
            return TerminationReason::RESTORATION_FAILED;
        case SolverStatus::INVALID_INPUT:
            return TerminationReason::INVALID_INPUT;
        case SolverStatus::NUMERICAL_ERROR:
            return TerminationReason::NUMERICAL_ERROR;
        case SolverStatus::INFEASIBLE:
            return TerminationReason::POSTSOLVE_INFEASIBLE;
        default:
            return TerminationReason::NUMERICAL_ERROR;
        }
    }

    void record_iteration_info_(const StepResidualSummary& residuals, double max_dual)
    {
        context_.info.iterations = context_.solve.current_iter;
        context_.info.primal_inf = residuals.max_primal_inf;
        context_.info.unscaled_primal_inf = residuals.max_unscaled_primal_inf;
        context_.info.dual_inf = max_dual;
        context_.info.complementarity_inf = residuals.max_complementarity_gap;
        context_.info.barrier_centrality_inf = residuals.max_barrier_complementarity_residual;
        context_.info.mu = residuals.barrier_mu;
        context_.info.reg = context_.solve.reg;
        context_.info.alpha = context_.metrics.last_alpha;
        context_.info.linear_ok = true;
        context_.info.constraint_scaling_active = build_state_.plan.constraint_scaling_active;
        context_.info.objective_scaling_active = build_state_.plan.objective_scaling_active;
        context_.info.problem_scaling_active = build_state_.plan.problem_scaling_active;
        context_.info.coordinate_scaling_active = coordinate_scaling_has_nontrivial_factors_();
    }

    void record_postsolve_info_(SolverStatus final_status, SolverStatus loop_status,
        TerminationReason reason, const PostsolveResiduals& residuals)
    {
        context_.info.status = final_status;
        context_.info.loop_status = loop_status;
        context_.info.termination_reason = reason;
        context_.info.iterations = context_.solve.current_iter;
        context_.info.primal_inf = residuals.max_primal_inf;
        context_.info.unscaled_primal_inf = residuals.max_unscaled_primal_inf;
        context_.info.dual_inf = residuals.max_dual_inf;
        context_.info.complementarity_inf = residuals.max_complementarity_gap;
        context_.info.barrier_centrality_inf = residuals.max_barrier_complementarity_residual;
        context_.info.mu = residuals.barrier_mu;
        context_.info.reg = context_.solve.reg;
        context_.info.alpha = context_.metrics.last_alpha;
        context_.info.linear_ok = residuals.linear_ok;
        context_.info.constraint_scaling_active = build_state_.plan.constraint_scaling_active;
        context_.info.objective_scaling_active = build_state_.plan.objective_scaling_active;
        context_.info.problem_scaling_active = build_state_.plan.problem_scaling_active;
        context_.info.coordinate_scaling_active = coordinate_scaling_has_nontrivial_factors_();
    }

    void record_terminal_info_(SolverStatus final_status, SolverStatus loop_status)
    {
        context_.info.status = final_status;
        context_.info.loop_status = loop_status;
        context_.info.termination_reason = reason_for_loop_status_(loop_status);
        context_.info.iterations = context_.solve.current_iter;
        context_.info.mu = context_.solve.mu;
        context_.info.reg = context_.solve.reg;
        context_.info.alpha = context_.metrics.last_alpha;
        context_.info.linear_ok = false;
        context_.info.constraint_scaling_active = build_state_.plan.constraint_scaling_active;
        context_.info.objective_scaling_active = build_state_.plan.objective_scaling_active;
        context_.info.problem_scaling_active = build_state_.plan.problem_scaling_active;
        context_.info.coordinate_scaling_active = coordinate_scaling_has_nontrivial_factors_();
    }

    bool check_convergence(const StepResidualSummary& residuals, double max_dual)
    {
        detail::TerminationSnapshot snapshot;
        snapshot.linear_ok = true;
        snapshot.primal_inf = residuals.max_primal_inf;
        snapshot.dual_inf = max_dual;
        snapshot.complementarity_inf = residuals.max_complementarity_gap;
        snapshot.barrier_centrality_inf = residuals.max_barrier_complementarity_residual;
        snapshot.mu = residuals.barrier_mu;
        return detail::TerminationKernel::check_convergence(config, snapshot);
    }

    bool check_convergence(const PostsolveResiduals& residuals)
    {
        detail::TerminationSnapshot snapshot;
        snapshot.linear_ok = residuals.linear_ok;
        snapshot.primal_inf = residuals.max_primal_inf;
        snapshot.dual_inf = residuals.max_dual_inf;
        snapshot.complementarity_inf = residuals.max_complementarity_gap;
        snapshot.barrier_centrality_inf = residuals.max_barrier_complementarity_residual;
        snapshot.mu = residuals.barrier_mu;
        return detail::TerminationKernel::check_convergence(config, snapshot);
    }

    double compute_objective_cost_(const TrajArray& traj) const
    {
        double total_cost = 0.0;
        for (int k = 0; k <= N; ++k) {
            total_cost += traj[k].cost_unscaled;
        }
        return total_cost;
    }

    bool should_stop_for_cost_stagnation_(
        double current_cost, double last_cost, double last_mu) const
    {
        return detail::TerminationKernel::should_stop_for_cost_stagnation(config,
            context_.metrics.last_prim_inf, current_cost, last_cost, context_.solve.mu, last_mu);
    }

    StepResidualSummary evaluate_step_model_(TrajArray& traj)
    {
        StepResidualSummary summary;
        summary.barrier_mu = context_.solve.mu;
        const double mu_eval = summary.barrier_mu;
        double total_gap = 0.0;
        int total_gap_dim = 0;

        for (int k = 0; k <= N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;

            detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N);

            for (int i = 0; i < NC; ++i) {
                const double s = traj[k].s(i);
                const double lam = traj[k].lam(i);
                const double gap = s * lam;
                double comp = std::abs(s * lam - mu_eval);
                if (comp > summary.max_barrier_complementarity_residual) {
                    summary.max_barrier_complementarity_residual = comp;
                }
                if (std::abs(gap) > summary.max_complementarity_gap) {
                    summary.max_complementarity_gap = std::abs(gap);
                }

                total_gap += gap;
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
                if (detail::is_l1_soft_constraint(type, w, config)) {
                    const double soft_s = traj[k].soft_s(i);
                    const double soft_dual = (w - lam);
                    const double soft_gap = soft_s * soft_dual;
                    double comp_soft = std::abs(soft_s * soft_dual - mu_eval);
                    if (comp_soft > summary.max_barrier_complementarity_residual) {
                        summary.max_barrier_complementarity_residual = comp_soft;
                    }
                    if (std::abs(soft_gap) > summary.max_complementarity_gap) {
                        summary.max_complementarity_gap = std::abs(soft_gap);
                    }

                    total_gap += soft_gap;
                    total_gap_dim += 1;
                }
            }
        }

        summary.max_primal_inf = compute_max_violation(traj);
        summary.max_unscaled_primal_inf = compute_unscaled_max_violation(traj);
        summary.avg_complementarity_gap = (total_gap_dim > 0) ? (total_gap / total_gap_dim) : 0.0;
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
            update_barrier(
                residuals.max_barrier_complementarity_residual, residuals.avg_complementarity_gap);
        }
    }

    double compute_dual_infeasibility_(const TrajArray& traj) const
    {
        // Default path: unweighted inf-norm of the control-stationarity vector.
        // This is the legacy contract and is preserved bit-for-bit when
        // CoordinateScalingMethod::NONE is selected.
        if (config.coordinate_scaling != CoordinateScalingMethod::USER_SUPPLIED) {
            double max_dual_inf = 0.0;
            for (int k = 0; k <= N; ++k) {
                const double r_norm = MatOps::norm_inf(traj[k].r_bar);
                if (r_norm > max_dual_inf) {
                    max_dual_inf = r_norm;
                }
            }
            return max_dual_inf;
        }

        // USER_SUPPLIED: weight each control-stationarity component by its
        // user-declared coordinate scale before taking the maximum. This
        // never rescales the search direction; it only normalises the
        // termination metric so coordinates with naturally large gradients
        // do not mask convergence on coordinates with small gradients.
        double max_dual_inf = 0.0;
        for (int k = 0; k <= N; ++k) {
            const auto& kp = traj[k];
            for (int i = 0; i < NU; ++i) {
                const double weighted = std::abs(kp.r_bar(i)) * control_coord_scale_[i];
                if (weighted > max_dual_inf) {
                    max_dual_inf = weighted;
                }
            }
        }
        return max_dual_inf;
    }

    bool coordinate_scaling_has_nontrivial_factors_() const
    {
        if (config.coordinate_scaling != CoordinateScalingMethod::USER_SUPPLIED) {
            return false;
        }
        for (int i = 0; i < NU; ++i) {
            if (control_coord_scale_[i] != 1.0) {
                return true;
            }
        }
        return false;
    }

    void increase_regularization_after_failed_solve_(bool clamp_to_min)
    {
        context_.info.regularization_escalation_count++;
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
        for (int attempt = 0; attempt < config.linear_solve_max_attempts; ++attempt) {
            const LinearSolveResult linear_result = linear_solver->solve(traj, N, target_mu,
                context_.solve.reg, config.inertia_strategy, config, affine_traj);
            record_linear_solver_diagnostics_(linear_result);
            if (linear_result.ok) {
                return true;
            }
            if (attempt + 1 < config.linear_solve_max_attempts) {
                increase_regularization_after_failed_solve_(clamp_reg_to_min);
            }
        }
        return false;
    }

    void record_linear_solver_diagnostics_(const LinearSolveResult& result)
    {
        if (result.degraded_step) {
            context_.info.degraded_step = true;
            context_.info.degraded_riccati_freeze_count += result.degraded_riccati_freeze_count;
        }
        // Riccati inertia-correction diagnostics. Always accumulate the
        // counters so any caller can detect numerically suspicious solves.
        // RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS additionally promotes a
        // non-zero correction count to a degraded_step flag for monitoring.
        context_.info.riccati_indefinite_blocks += result.riccati_indefinite_blocks;
        if (result.riccati_max_diagonal_perturbation
            > context_.info.riccati_max_diagonal_perturbation) {
            context_.info.riccati_max_diagonal_perturbation
                = result.riccati_max_diagonal_perturbation;
        }
        if (config.riccati_robust_mode == RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS
            && result.riccati_indefinite_blocks > 0) {
            context_.info.degraded_step = true;
        }
    }

    void record_line_search_diagnostics_(const LineSearchResult& result)
    {
        if (result.soc_attempted) {
            context_.info.soc_attempt_count++;
        }
        if (result.soc_accepted) {
            context_.info.soc_accept_count++;
        }
        if (result.soc_rejected) {
            context_.info.soc_reject_count++;
        }
        context_.info.line_search_backtracking_count += result.backtracks;
        // Pareto-frontier filter diagnostics. MeritLineSearch leaves these
        // at zero so the cumulative counters only ever reflect the filter
        // line-search path that actually maintains a history.
        context_.info.filter_entries_pruned_total += result.filter_entries_pruned;
        context_.info.filter_redundant_inserts_total += result.filter_redundant_inserts;
        if (result.filter_size_after > context_.info.filter_max_history_size) {
            context_.info.filter_max_history_size = result.filter_size_after;
        }
    }

    void prepare_direction_workspace_()
    {
        // Candidate buffer preparation (shared by Mehrotra predictor and direction-refinement
        // backup). Both features need a full copy of the current linearized system (A, B, Q, R).
        // - Mehrotra: uses it as the affine solve workspace
        // - Direction refinement: uses it to access original A, B matrices after Riccati overwrites
        //   active.
        // Note: The Mehrotra affine solve only modifies solver workspace (Q_bar, R_bar, K, dx...)
        // but NOT model derivatives (A, B, C, D, Q, R, f_resid, x), so the backup remains valid
        // even after the affine solve.
        bool need_candidate_backup = (config.barrier_strategy == BarrierStrategy::MEHROTRA)
            || config.direction_refinement != DirectionRefinementMode::NONE;
        if (need_candidate_backup) {
            trajectory.prepare_candidate_full();
        }
    }

    struct FractionToBoundaryResult {
        double primal = 1.0;
        double dual = 1.0;

        double combined() const { return std::min(primal, dual); }
    };

    FractionToBoundaryResult compute_fraction_to_boundary_(const TrajArray& direction_traj) const
    {
        FractionToBoundaryResult alpha;
        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NC; ++i) {
                double s = direction_traj[k].s(i);
                double ds = direction_traj[k].ds(i);
                double lam = direction_traj[k].lam(i);
                double dlam = direction_traj[k].dlam(i);

                if (ds < 0) {
                    double a = -s / ds;
                    if (a < alpha.primal) {
                        alpha.primal = a;
                    }
                }
                if (dlam < 0) {
                    double a = -lam / dlam;
                    if (a < alpha.dual) {
                        alpha.dual = a;
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
                if (detail::is_l1_soft_constraint(type, w, config)) {
                    double soft_s = direction_traj[k].soft_s(i);
                    double dsoft_s = direction_traj[k].dsoft_s(i);
                    if (dsoft_s < 0) {
                        double a = -soft_s / dsoft_s;
                        if (a < alpha.primal) {
                            alpha.primal = a;
                        }
                    }
                    // soft dual: (w - lam) with direction -dlam.
                    double soft_dual = w - lam;
                    double dsoft_dual = -dlam; // d(w-lam)/dalpha = -dlam
                    if (dsoft_dual < 0) {
                        double a = -soft_dual / dsoft_dual;
                        if (a < alpha.dual) {
                            alpha.dual = a;
                        }
                    }
                }
            }
        }
        return alpha;
    }

    double compute_affine_barrier_mu_(const TrajArray& base_traj, const TrajArray& affine_traj,
        double alpha_primal_aff, double alpha_dual_aff) const
    {
        double total_comp = 0.0;
        int total_dim = 0;

        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NC; ++i) {
                double s_new = base_traj[k].s(i) + alpha_primal_aff * affine_traj[k].ds(i);
                double lam_new = base_traj[k].lam(i) + alpha_dual_aff * affine_traj[k].dlam(i);
                if (s_new < 0) {
                    s_new = detail::barrier_floor(config);
                }
                if (lam_new < 0) {
                    lam_new = detail::barrier_floor(config);
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
                if (detail::is_l1_soft_constraint(type, w, config)) {
                    double soft_s_new
                        = base_traj[k].soft_s(i) + alpha_primal_aff * affine_traj[k].dsoft_s(i);
                    double soft_dual_new = w - lam_new;
                    if (soft_s_new < 0) {
                        soft_s_new = detail::barrier_floor(config);
                    }
                    if (soft_dual_new < detail::l1_soft_dual_floor(w, config)) {
                        soft_dual_new = detail::l1_soft_dual_floor(w, config);
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

    void apply_direction_refinement_if_enabled_(TrajArray& traj, bool solve_success)
    {
        if (!solve_success || config.direction_refinement == DirectionRefinementMode::NONE) {
            return;
        }
        // Pass 'traj' (which contains solution dx, du) and 'candidate' (which contains original
        // system).
        linear_solver->refine_direction(
            traj, trajectory.candidate(), N, context_.solve.mu, context_.solve.reg, config);
        context_.info.direction_refinement_passes += linear_solver->last_refine_passes_consumed();
        context_.info.direction_refinement_last_defect = linear_solver->last_refine_defect();
    }

    void update_metrics_after_globalization_(double max_dual_inf)
    {
        // For outer-loop heuristics (e.g. cost stagnation), store feasibility of the *current*
        // iterate, i.e. after line-search has potentially swapped buffers.
        context_.metrics.last_prim_inf = compute_max_violation(trajectory.active());
        context_.metrics.last_dual_inf = max_dual_inf;
    }

    SolverStatus classify_tiny_step_stagnation_(double max_prim_inf, double max_dual_inf) const
    {
        return detail::TerminationKernel::classify_tiny_step_stagnation(
            config, max_prim_inf, max_dual_inf);
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
            const LinearSolveResult linear_result = linear_solver->solve(traj_after_ls, N,
                context_.solve.mu, context_.solve.reg, config.inertia_strategy, config);
            record_linear_solver_diagnostics_(linear_result);
            recovered = linear_result.ok;

            if (recovered) {
                // Mark: If this Reset failed to get us out of trouble, disallow it next time
                context_.solve.slack_reset_consecutive_count++;
            }
        } else if (config.enable_slack_reset && context_.solve.slack_reset_consecutive_count >= 1) {
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

    GlobalizationResult globalize_step_(
        double barrier_mu_at_residual_eval, double max_prim_inf, double max_dual_inf)
    {
        GlobalizationResult result;

        // Notify the line search if μ decreased during this step so it can
        // discard barrier-dependent history (filter entries, ratcheted merit_nu).
        if (line_search && context_.solve.mu < barrier_mu_at_residual_eval) {
            line_search->on_barrier_update();
        }

        timer.start("Line Search");
        const LineSearchResult line_search_result = line_search->search(
            trajectory, *linear_solver, dt_traj, context_.solve.mu, context_.solve.reg, config);
        record_line_search_diagnostics_(line_search_result);
        result.alpha = line_search_result.alpha;
        alpha_log_.push_back(result.alpha);
        context_.metrics.last_alpha = result.alpha;
        // IMPORTANT: line_search may swap trajectory buffers.
        // Do not use references to trajectory.active() taken before this call.
        auto& traj_after_ls = trajectory.active();

        // If step size is valid, reset the counter.
        if (result.alpha > 1e-8) {
            context_.solve.slack_reset_consecutive_count = 0;
        } else {
            SolverStatus stagnation_status
                = classify_tiny_step_stagnation_(max_prim_inf, max_dual_inf);
            if (stagnation_status == SolverStatus::OPTIMAL) {
                if (config.print_level >= PrintLevel::INFO) {
                    MLOG_INFO("Line search stagnated at optimal point (PrimInf: "
                        << max_prim_inf << ", DualInf: " << max_dual_inf
                        << "). Terminating as OPTIMAL.");
                }
                timer.stop();
                result.status = SolverStatus::OPTIMAL;
                return result;
            }
            if (stagnation_status == SolverStatus::FEASIBLE) {
                if (config.print_level >= PrintLevel::INFO) {
                    MLOG_INFO("Line search stagnated at feasible point (PrimInf: "
                        << max_prim_inf << "). Terminating as FEASIBLE.");
                }
                timer.stop();
                result.status = SolverStatus::FEASIBLE;
                return result;
            }

            result.recovered = attempt_tiny_step_recovery_(traj_after_ls, result.alpha);

            if (!result.recovered) {
                // Both Slack Reset and Feasibility Restoration failed (or were disabled).
                context_.info.line_search_failed = true;
                timer.stop();
                print_iteration_log(result.alpha);
                result.status = config.enable_feasibility_restoration
                    ? SolverStatus::RESTORATION_FAILED
                    : SolverStatus::STEP_TOO_SMALL;
                return result;
            }
        }

        timer.stop();

        context_.globalization.accepted_alpha = result.alpha;
        context_.globalization.recovered = result.recovered;
        print_iteration_log(result.alpha);
        return result;
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
        FractionToBoundaryResult alpha_aff = compute_fraction_to_boundary_(affine_traj);

        double mu_curr = context_.solve.mu;
        double mu_aff
            = compute_affine_barrier_mu_(traj, affine_traj, alpha_aff.primal, alpha_aff.dual);
        context_.metrics.last_mu_aff = mu_aff;
        context_.metrics.last_alpha_aff = alpha_aff.combined();

        if (config.print_level >= PrintLevel::DEBUG) {
            MLOG_INFO("Mehrotra Debug: mu_curr=" << mu_curr << ", mu_aff=" << mu_aff
                                                 << ", alpha_aff=" << alpha_aff.combined());
        }

        double mu_target = detail::BarrierUpdateKernel::mehrotra_target_mu(
            config, mu_curr, mu_aff, alpha_aff.combined());

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

    bool is_unsupported_backend_requested_() const
    {
        return config.backend == Backend::GPU_MPX || config.backend == Backend::GPU_PCR;
    }

    DirectionResult validate_search_direction_(const TrajArray& traj, bool solve_success)
    {
        DirectionResult result;
        result.solve_success = solve_success;

        if (!solve_success) {
            result.status = SolverStatus::LINEAR_SOLVE_FAILED;
            return result;
        }

        // Direction check after Riccati: scan the whole valid horizon so a
        // later-stage slack/dual/state NaN cannot hide behind a finite dx0/du0.
        if (has_invalid_search_direction(traj)) {
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_ERROR("Numerical Error: invalid search direction detected.");
            }
            result.status = SolverStatus::NUMERICAL_ERROR;
            return result;
        }

        // Dual infeasibility metric: use Qu (stored in r_bar after the Riccati backward pass).
        // This is only valid after a successful linear solve (r_bar is stale right after
        // line-search swaps).
        result.max_dual_inf = compute_dual_infeasibility_(traj);
        return result;
    }

    DirectionResult compute_search_direction_(TrajArray& traj)
    {
        if (is_unsupported_backend_requested_()) {
#ifdef USE_CUDA
            MLOG_ERROR("GPU backend is not implemented yet.");
#else
            MLOG_ERROR("CUDA not enabled; GPU backend is unsupported.");
#endif
            DirectionResult result;
            result.status = SolverStatus::INVALID_INPUT;
            return result;
        }

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

        apply_direction_refinement_if_enabled_(traj, solve_success);

        timer.stop();

        DirectionResult result = validate_search_direction_(traj, solve_success);
        context_.direction.solve_success = result.solve_success;
        context_.direction.max_dual_inf = result.max_dual_inf;
        return result;
    }

    void update_barrier(double max_barrier_complementarity_residual, double avg_complementarity_gap)
    {
        context_.solve.mu = detail::BarrierUpdateKernel::update_mu(config, context_.solve.mu,
            max_barrier_complementarity_residual, avg_complementarity_gap);
    }

    // Allocation contract:
    //   * PrintLevel::NONE (default): this function early-returns BEFORE
    //     constructing std::stringstream, so it does not perturb the zero-
    //     malloc invariant of solve(). test_memory's
    //     IterationLogPrintLevelNoneStaysAllocationFreeAndSilent locks this in.
    //   * PrintLevel::ITER and higher: this function intentionally constructs
    //     std::stringstream for header/row formatting and routes through
    //     MLOG_INFO. This is opt-in for diagnostics and is NOT zero-malloc.
    //     test_memory's IterationLogPrintLevelIterAllocatesAndRoutesByDesign
    //     reverse-anchors that boundary so a future refactor cannot silently
    //     claim PrintLevel::ITER is allocation-free.
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

            if (config.barrier_strategy == BarrierStrategy::MEHROTRA) {
                ss << std::setw(10) << "AlphaAff" << std::setw(12) << "MuAff";
            }
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
            total_cost += kp.cost_unscaled;
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
           << std::setprecision(2) << std::setw(10) << max_prim_inf << std::setw(10) << max_dual_inf
           << std::fixed << std::setprecision(3) << std::setw(10) << alpha;

        if (config.barrier_strategy == BarrierStrategy::MEHROTRA) {
            ss << std::fixed << std::setprecision(3) << std::setw(10)
               << context_.metrics.last_alpha_aff << std::scientific << std::setprecision(2)
               << std::setw(12) << context_.metrics.last_mu_aff;
        }
        if (config.print_level >= PrintLevel::DEBUG) {
            ss << std::scientific << std::setprecision(2) << std::setw(12) << min_slack;
        }
        MLOG_INFO(ss.str());
    }

    bool has_nans(const typename TrajectoryType::TrajArray& t) const
    {
        // Historical name: this is the invalid-number guard for solve hot-path data.
        // Use bit-level finite checks so Inf is rejected even under -ffast-math.
        for (int k = 0; k <= N; ++k) {
            const auto& kp = t[k];
            if (!MatOps::all_finite(kp.x) || !MatOps::all_finite(kp.u) || !MatOps::all_finite(kp.p)
                || !MatOps::all_finite(kp.s) || !MatOps::all_finite(kp.lam)
                || !MatOps::all_finite(kp.soft_s)) {
                return true;
            }
            if (!MatOps::all_finite(kp.dx) || !MatOps::all_finite(kp.du)
                || !MatOps::all_finite(kp.ds) || !MatOps::all_finite(kp.dlam)
                || !MatOps::all_finite(kp.dsoft_s)) {
                return true;
            }
            if (!MatOps::is_finite_scalar(kp.cost)) {
                return true;
            }
            if (!MatOps::is_finite_scalar(kp.cost_unscaled)
                || !MatOps::is_finite_scalar(kp.objective_scale)) {
                return true;
            }
            // Model outputs — invalid numbers here propagate into barrier/Riccati.
            if (!MatOps::all_finite(kp.g_val) || !MatOps::all_finite(kp.g_true)
                || !MatOps::all_finite(kp.f_resid) || !MatOps::all_finite(kp.A)
                || !MatOps::all_finite(kp.B) || !MatOps::all_finite(kp.C)
                || !MatOps::all_finite(kp.D) || !MatOps::all_finite(kp.Q)
                || !MatOps::all_finite(kp.R) || !MatOps::all_finite(kp.H)
                || !MatOps::all_finite(kp.q) || !MatOps::all_finite(kp.r)) {
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
                if (!MatOps::is_finite_scalar(kp.ds(i)) || !MatOps::is_finite_scalar(kp.dlam(i))
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
            if (!MatOps::all_finite(kp.x) || !MatOps::all_finite(kp.u)
                || !MatOps::all_finite(kp.p)) {
                return false;
            }
            if (!MatOps::all_finite(kp.s) || !MatOps::all_finite(kp.lam)) {
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

                if (detail::is_l1_soft_constraint(type, w, config)) {
                    if (!MatOps::is_finite_scalar(kp.soft_s(i)) || kp.soft_s(i) <= 0.0) {
                        return false;
                    }
                    if (w - kp.lam(i) <= detail::l1_soft_dual_floor(w, config)) {
                        return false;
                    }
                }
            }
        }
        return !has_nans(t);
    }

    bool feasibility_restoration()
    {
        context_.info.restoration_used = true;
        context_.info.restoration_attempt_count++;
        if (config.print_level >= PrintLevel::DEBUG) {
            MLOG_DEBUG("Entering Feasibility Restoration Phase.");
        }
        double saved_mu = context_.solve.mu;
        double saved_reg = context_.solve.reg;
        context_.solve.mu = config.restoration_mu;
        context_.solve.reg = config.restoration_reg;

        auto& traj = trajectory.active();
        bool success = false;
        auto refresh_trajectory_model = [this, &traj]() {
            for (int k = 0; k <= N; ++k) {
                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N);
            }
        };

        refresh_trajectory_model();
        const double violation_before = compute_max_violation(traj);

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

            // Restoration linear solve.
            // Quadratic-penalty feasibility restoration:
            // Minimizing 0.5*||dx||^2 + 0.5*rho*||C*dx + D*du + g + s||^2
            // This pulls the solution towards the constraint manifold more aggressively than simple
            // min-norm. This is a restoration heuristic, not a full ALADIN or augmented-Lagrangian
            // outer loop.
            //
            // rho selection:
            //   FIXED: rho = restoration_rho_init (legacy 1000.0 default).
            //   VIOLATION_ADAPTIVE: rho = clamp(rho_init / max(theta, floor),
            //                                  rho_min, rho_max).
            // Adaptive mode keeps the augmented Hessian well-conditioned when
            // theta is large and pulls aggressively to feasibility once
            // theta drops, without retuning restoration_mu / restoration_reg.
            double rho = config.restoration_rho_init;
            if (config.restoration_penalty_mode
                == SolverConfig::RestorationPenaltyMode::VIOLATION_ADAPTIVE) {
                double theta_inf = config.restoration_rho_violation_floor;
                for (int k = 0; k <= N; ++k) {
                    const auto& kp = traj[k];
                    for (int i = 0; i < NC; ++i) {
                        const double val = std::abs(kp.g_val(i) + kp.s(i));
                        if (val > theta_inf) {
                            theta_inf = val;
                        }
                    }
                }
                rho = config.restoration_rho_init / theta_inf;
                if (rho < config.restoration_rho_min) {
                    rho = config.restoration_rho_min;
                }
                if (rho > config.restoration_rho_max) {
                    rho = config.restoration_rho_max;
                }
                context_.info.restoration_rho_adaptive_steps++;
            }
            if (context_.info.restoration_rho_min_used == 0.0
                || rho < context_.info.restoration_rho_min_used) {
                context_.info.restoration_rho_min_used = rho;
            }
            if (rho > context_.info.restoration_rho_max_used) {
                context_.info.restoration_rho_max_used = rho;
            }
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

            const LinearSolveResult linear_result = linear_solver->solve(
                traj, N, context_.solve.mu, context_.solve.reg, config.inertia_strategy, config);
            record_linear_solver_diagnostics_(linear_result);
            if (!linear_result.ok) {
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
                    if (detail::is_l1_soft_constraint(type, w, config)) {
                        const double soft_s = traj[k].soft_s(i);
                        const double dsoft_s = traj[k].dsoft_s(i);
                        if (dsoft_s < 0) {
                            alpha = std::min(alpha, -config.restoration_alpha * soft_s / dsoft_s);
                        }

                        if (dlam > 0) {
                            const double gap = (w - lam) - detail::l1_soft_dual_floor(w, config);
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
                if (traj[k].s(i) < detail::barrier_floor(config)) {
                    traj[k].s(i) = detail::barrier_floor(config);
                }

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                if (detail::is_l1_soft_constraint(type, w, config)) {
                    (void)project_l1_soft_pair_to_central_path_(traj[k], i, saved_mu, w);
                } else {
                    // Preserve dual info from restoration if valuable, but ensure complementarity
                    // lower bound.
                    const double lam_min = saved_mu / traj[k].s(i);
                    traj[k].lam(i) = std::max(traj[k].lam(i), lam_min);
                }
            }
        }

        context_.solve.mu = saved_mu;
        context_.solve.reg = saved_reg;

        refresh_trajectory_model();
        const double violation_after = compute_max_violation(traj);
        const double feasible_bound = config.tol_con * config.feasible_tol_scale;
        const double required_violation
            = config.restoration_sufficient_decrease_factor * violation_before;

        const bool restored = success && MatOps::is_finite_scalar(violation_after)
            && (violation_after <= feasible_bound || violation_after <= required_violation);
        if (restored) {
            context_.info.restoration_success_count++;
        }
        return restored;
    }

    bool project_l1_soft_pair_to_central_path_(Knot& kp, int i, double mu, double w) const
    {
        const double barrier_floor
            = std::max(config.min_barrier_slack, std::numeric_limits<double>::epsilon());
        const double soft_dual_floor = detail::l1_soft_dual_floor(w, config);
        const double lam_max = w - soft_dual_floor;
        if (!MatOps::is_finite_scalar(lam_max) || lam_max <= barrier_floor) {
            return false;
        }

        const double min_s_for_box = mu / lam_max;
        if (kp.s(i) < min_s_for_box) {
            kp.s(i) = min_s_for_box;
        }

        const double lam_min = mu / kp.s(i);
        kp.lam(i) = std::clamp(kp.lam(i), lam_min, lam_max);

        const double soft_dual = std::max(soft_dual_floor, w - kp.lam(i));
        kp.soft_s(i) = std::max(config.min_barrier_slack, mu / soft_dual);

        return MatOps::is_finite_scalar(kp.s(i)) && MatOps::is_finite_scalar(kp.lam(i))
            && MatOps::is_finite_scalar(kp.soft_s(i)) && kp.s(i) > 0.0 && kp.lam(i) > 0.0
            && (w - kp.lam(i)) > soft_dual_floor;
    }

public:
    SolverStatus solve()
    {
        // RTI-lite preflight: under user opt-in, decide whether the upcoming
        // solve may reuse the previous primal-dual iterate. The reuse gates
        // are intentionally conservative (state delta + age budget + previous
        // status) so a fall-back to the full solve happens whenever any
        // invariant is unclear. The solver-strategy fields are temporarily
        // overridden via an RAII guard so the caller's SolverConfig is never
        // mutated.
        const SolverConfig saved_config = config;
        const bool rti_lite_reuse = rti_lite_should_reuse_();
        if (rti_lite_reuse) {
            apply_rti_lite_overrides_();
        } else {
            rti_lite_linearization_age_ = 0;
        }

        if (!begin_solve_()) {
            config = saved_config;
            context_.info.rti_lite_reused_linearization = false;
            context_.info.rti_lite_linearization_age = 0;
            return SolverStatus::INVALID_INPUT;
        }
        SolverStatus loop_exit_status = run_solve_loop_();

        // 3. Postsolve: refresh residuals and produce the final verdict.
        const SolverStatus final_status = postsolve(loop_exit_status);

        config = saved_config;

        rti_lite_record_solve_outcome_(final_status, rti_lite_reuse);
        return final_status;
    }

private:
    // RTI-lite gates: every reuse condition must hold or we fall back to a
    // full solve. The state-delta gate uses an L2 norm so the threshold has
    // physical units rather than per-coordinate semantics; coordinate-level
    // scaling will refine this once Tier 3.1 lands.
    bool rti_lite_should_reuse_() const
    {
        if (!config.enable_rti_lite) {
            return false;
        }
        if (!rti_lite_have_previous_solve_ || !rti_lite_last_solve_acceptable_) {
            return false;
        }
        if (rti_lite_linearization_age_ >= config.rti_lite_max_linearization_age) {
            return false;
        }
        const auto& x_now = trajectory[0].x;
        double sq = 0.0;
        for (int i = 0; i < Model::NX; ++i) {
            const double d = x_now(i) - rti_lite_last_x0_(i);
            sq += d * d;
        }
        const double delta = std::sqrt(sq);
        return delta < config.rti_lite_max_state_delta;
    }

    void apply_rti_lite_overrides_()
    {
        // Reuse path: cap iterations at the linearization-age budget and
        // accept ACCEPTABLE_NMPC quality. The user's max_iters is preserved
        // as an upper bound so a smaller user budget still wins.
        const int reuse_budget = std::max(1, config.rti_lite_max_linearization_age);
        config.max_iters = std::min(config.max_iters, reuse_budget);
        config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    }

    void rti_lite_record_solve_outcome_(SolverStatus final_status, bool reused)
    {
        const bool acceptable
            = (final_status == SolverStatus::OPTIMAL || final_status == SolverStatus::FEASIBLE);

        rti_lite_last_solve_acceptable_ = acceptable;
        rti_lite_have_previous_solve_ = true;
        for (int i = 0; i < Model::NX; ++i) {
            rti_lite_last_x0_(i) = trajectory[0].x(i);
        }

        if (acceptable && reused) {
            ++rti_lite_linearization_age_;
        } else if (!acceptable) {
            rti_lite_linearization_age_ = 0;
        } else {
            // Acceptable but not reused -- start a fresh age counter.
            rti_lite_linearization_age_ = 0;
        }
        context_.info.rti_lite_reused_linearization = reused;
        context_.info.rti_lite_linearization_age = rti_lite_linearization_age_;
    }

    bool begin_solve_()
    {
        rebuild_solver_components_if_dirty_();
        // Diagnostic line-search trace — fresh per solve. Reserve once at the
        // configured max iteration count so the hot-path push_back stays
        // pointer-bump only.
        alpha_log_.clear();
        context_.metrics.reset_solve();

        if (!build_state_.plan.constraint_scaling_plan_valid
            || !build_state_.plan.objective_scaling_plan_valid
            || !build_state_.plan.problem_scaling_plan_valid) {
            context_.reset_solve();
            record_terminal_info_(SolverStatus::INVALID_INPUT, SolverStatus::INVALID_INPUT);
            return false;
        }
        // 1. Presolve: prepare data, handle initialization, and reset runtime state.
        presolve();
        return true;
    }

    LoopExitDecision should_exit_after_step_status_(SolverStatus step_stat, int iter) const
    {
        LoopExitDecision decision;

        // Terminal failures stop immediately.
        if (step_stat == SolverStatus::NUMERICAL_ERROR
            || step_stat == SolverStatus::LINEAR_SOLVE_FAILED
            || step_stat == SolverStatus::STEP_TOO_SMALL
            || step_stat == SolverStatus::INSUFFICIENT_PROGRESS
            || step_stat == SolverStatus::RESTORATION_FAILED
            || step_stat == SolverStatus::INVALID_INPUT) {
            decision.should_exit = true;
            decision.status = step_stat;
            decision.reason = reason_for_loop_status_(step_stat);
            return decision;
        }

        // SQP-RTI mode exits after one non-fatal step; postsolve assigns the final quality.
        // Fatal direction/search failures above must not be masked as fixed-iteration exits.
        if (detail::TerminationKernel::uses_fixed_iteration_profile(config)) {
            decision.should_exit = true;
            decision.reason = TerminationReason::FIXED_ITERATION;
            return decision;
        }

        // Candidate convergence: exit the loop and let postsolve verify with fresh residuals.
        if (step_stat == SolverStatus::OPTIMAL) {
            decision.should_exit = true;
            decision.status = SolverStatus::OPTIMAL;
            decision.reason = TerminationReason::CONVERGED;
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_INFO("Converged in " << iter + 1 << " iterations.");
            }
            return decision;
        }

        if (step_stat == SolverStatus::FEASIBLE || step_stat == SolverStatus::INFEASIBLE) {
            decision.should_exit = true;
            decision.status = step_stat;
            decision.reason = reason_for_loop_status_(step_stat);
            return decision;
        }

        return decision;
    }

    LoopExitDecision should_exit_for_cost_stagnation_(double& last_cost, double& last_mu)
    {
        LoopExitDecision decision;

        // Objective cost is comparable across μ updates (barrier terms are not part of
        // KnotPoint::cost). However, if μ is actively decreasing we should not stop purely
        // on objective stagnation, since IPM may still be progressing by reducing μ.
        // The intended use is to catch "μ frozen above μ_final" style stalls.
        double current_cost = compute_objective_cost_(trajectory.active());

        if (should_stop_for_cost_stagnation_(current_cost, last_cost, last_mu)) {
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_INFO("Cost Stagnation detected. Stopping early.");
            }
            // Preserve the termination reason; postsolve can still upgrade to OPTIMAL/FEASIBLE
            // if fresh residuals justify it.
            decision.should_exit = true;
            decision.status = SolverStatus::INSUFFICIENT_PROGRESS;
            decision.reason = TerminationReason::COST_STAGNATION;
            decision.cost_stagnated = true;
            return decision;
        }

        last_cost = current_cost;
        last_mu = context_.solve.mu;
        return decision;
    }

    SolverStatus run_solve_loop_()
    {
        SolverStatus loop_exit_status = SolverStatus::MAX_ITER;
        TerminationReason loop_exit_reason = TerminationReason::MAX_ITERATIONS;
        double last_cost = 1e30;
        double last_mu = context_.solve.mu;

        // 2. Solve loop: numerical iterations.
        for (int iter = 0; iter < config.max_iters; ++iter) {
            timer.start("Total Step");
            SolverStatus step_stat = execute_solve_iteration_();
            timer.stop();

            LoopExitDecision step_exit = should_exit_after_step_status_(step_stat, iter);
            if (step_exit.should_exit) {
                loop_exit_status = step_exit.status;
                loop_exit_reason = step_exit.reason;
                context_.termination.loop_exit_status = step_exit.status;
                break;
            }

            LoopExitDecision stagnation_exit = should_exit_for_cost_stagnation_(last_cost, last_mu);
            if (stagnation_exit.should_exit) {
                loop_exit_status = stagnation_exit.status;
                loop_exit_reason = stagnation_exit.reason;
                context_.termination.loop_exit_status = stagnation_exit.status;
                context_.termination.cost_stagnated = stagnation_exit.cost_stagnated;
                break;
            }
        }

        context_.termination.loop_exit_status = loop_exit_status;
        context_.info.loop_status = loop_exit_status;
        context_.info.termination_reason = loop_exit_reason;
        context_.info.iterations = context_.solve.current_iter;
        return loop_exit_status;
    }

    SolverStatus execute_solve_iteration_()
    {
        context_.solve.current_iter++;
        auto& traj = trajectory.active();

        StepResidualSummary residuals = evaluate_derivatives_phase_(traj);
        update_barrier_for_step_(residuals);

        DirectionResult direction = compute_search_direction_(traj);
        record_iteration_info_(residuals, direction.max_dual_inf);
        if (direction.status != SolverStatus::UNSOLVED) {
            return direction.status;
        }
        double max_dual_inf = direction.max_dual_inf;

        // Convergence check (Primal + Dual + Mu) using the freshly computed dual residual.
        // The final convergence verdict is always made in postsolve() with fresh data.
        if (context_.solve.current_iter > 1 && check_convergence(residuals, max_dual_inf)) {
            return SolverStatus::OPTIMAL;
        }

        GlobalizationResult globalization
            = globalize_step_(residuals.barrier_mu, residuals.max_primal_inf, max_dual_inf);
        if (globalization.status != SolverStatus::UNSOLVED) {
            return globalization.status;
        }

        // Do not certify convergence immediately after line search: the accepted
        // primal trajectory and pre-line-search dual residual belong to different
        // snapshots. The next iteration or postsolve() refreshes both on one iterate.
        update_metrics_after_globalization_(max_dual_inf);
        return SolverStatus::UNSOLVED;
    }

    // ============================================================
    // [Phase 1] Presolve: Preparation
    // ============================================================
    void reset_solve_runtime_state_(bool can_reuse_primal_dual)
    {
        const double previous_mu = context_.solve.mu;
        const double previous_reg = context_.solve.reg;

        // [Enable/Disable Profiling]
        timer.enabled = config.enable_profiling;
        if (line_search) {
            line_search->reset();
        }

        const double next_mu = detail::WarmStartKernel::select_barrier_mu<Model>(
            config, trajectory.active(), N, previous_mu, can_reuse_primal_dual);
        const double next_reg
            = detail::WarmStartKernel::select_regularization(config, previous_reg);
        context_.reset_algorithmic(next_mu, next_reg);
    }

    bool should_initialize_primal_dual_() const
    {
        return detail::InitializationKernel::should_initialize_primal_dual(
            config, has_valid_primal_dual_guess(trajectory.active()));
    }

    void initialize_primal_dual_from_model_()
    {
        auto& traj = trajectory.active();
        for (int k = 0; k <= N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;
            detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N, true);

            for (int i = 0; i < NC; ++i) {
                detail::InitializationKernel::initialize_constraint_primal_dual<Model>(
                    traj[k], i, context_.solve.mu, config);
            }
        }
    }

    void presolve()
    {
        const bool initialize_primal_dual = should_initialize_primal_dual_();

        reset_solve_runtime_state_(!initialize_primal_dual);

        if (initialize_primal_dual) {
            initialize_primal_dual_from_model_();
        }

        print_iteration_log(0.0, true);
        timer.reset();
    }

    // ============================================================
    // [Phase 3] Postsolve: Finalization & Verdict
    // ============================================================
    PostsolveResiduals refresh_postsolve_residuals_(TrajArray& traj)
    {
        PostsolveResiduals residuals;
        residuals.barrier_mu = context_.solve.mu;
        const double mu_eval = residuals.barrier_mu;
        residuals.max_dual_inf = std::numeric_limits<double>::infinity();

        for (int k = 0; k <= N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;

            detail::evaluate_model_stage<Model>(traj[k], config, current_dt, k == N);

            // 2. Check NaNs (bit-level, works under -ffast-math)
            for (int i = 0; i < NC; ++i) {
                if (MatOps::is_nan_scalar(traj[k].g_val(i))
                    || MatOps::is_nan_scalar(detail::true_constraint_value<Model>(traj[k], i))
                    || MatOps::is_nan_scalar(traj[k].s(i))) {
                    residuals.residuals_ok = false;
                    return residuals;
                }
            }

            // 3. Barrier complementarity (including L1-soft secondary pair).
            for (int i = 0; i < NC; ++i) {
                double comp = std::abs(traj[k].s(i) * traj[k].lam(i) - mu_eval);
                const double gap = traj[k].s(i) * traj[k].lam(i);
                if (comp > residuals.max_barrier_complementarity_residual) {
                    residuals.max_barrier_complementarity_residual = comp;
                }
                if (std::abs(gap) > residuals.max_complementarity_gap) {
                    residuals.max_complementarity_gap = std::abs(gap);
                }

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                if (detail::is_l1_soft_constraint(type, w, config)) {
                    const double soft_s = traj[k].soft_s(i);
                    const double soft_dual = (w - traj[k].lam(i));
                    const double soft_gap = soft_s * soft_dual;
                    double comp_soft = std::abs(soft_s * soft_dual - mu_eval);
                    if (comp_soft > residuals.max_barrier_complementarity_residual) {
                        residuals.max_barrier_complementarity_residual = comp_soft;
                    }
                    if (std::abs(soft_gap) > residuals.max_complementarity_gap) {
                        residuals.max_complementarity_gap = std::abs(soft_gap);
                    }
                }
            }
        }

        residuals.max_primal_inf = compute_max_violation(traj);
        residuals.max_unscaled_primal_inf = compute_unscaled_max_violation(traj);
        return residuals;
    }

    void evaluate_postsolve_dual_residual_(PostsolveResiduals& residuals)
    {
        // Dual infeasibility metric: require a fresh Riccati backward pass so Qu includes
        // the dynamic multipliers (B^T * pi_{k+1}). Use the inactive trajectory buffer as
        // scratch so postsolve never overwrites the active solution directions/gains.
        residuals.max_dual_inf = std::numeric_limits<double>::infinity();
        residuals.linear_ok = false;
        if (linear_solver) {
            trajectory.prepare_candidate_full();
            residuals.linear_ok = linear_solver->evaluate_dual_residual(trajectory.candidate(), N,
                residuals.barrier_mu, context_.solve.reg, config.inertia_strategy, config,
                residuals.max_dual_inf);
            // Coordinate-scaling hint: re-evaluate the dual-stationarity inf-norm
            // in scale-weighted units when the postsolve scratch trajectory is
            // available. The base solver call above used the unweighted norm by
            // contract; here we read the freshly populated r_bar back out of the
            // candidate buffer and apply the user-supplied weights.
            if (residuals.linear_ok
                && config.coordinate_scaling == CoordinateScalingMethod::USER_SUPPLIED
                && coordinate_scaling_has_nontrivial_factors_()) {
                const TrajArray& scratch = trajectory.candidate();
                double weighted = 0.0;
                for (int k = 0; k <= N; ++k) {
                    const auto& kp = scratch[k];
                    for (int i = 0; i < NU; ++i) {
                        const double w = std::abs(kp.r_bar(i)) * control_coord_scale_[i];
                        if (w > weighted) {
                            weighted = w;
                        }
                    }
                }
                residuals.max_dual_inf = weighted;
            }
        }
    }

    SolverStatus classify_postsolve_solution_quality_(const PostsolveResiduals& residuals)
    {
        detail::TerminationSnapshot snapshot;
        snapshot.linear_ok = residuals.linear_ok;
        snapshot.primal_inf = residuals.max_primal_inf;
        snapshot.dual_inf = residuals.max_dual_inf;
        snapshot.complementarity_inf = residuals.max_complementarity_gap;
        snapshot.barrier_centrality_inf = residuals.max_barrier_complementarity_residual;
        snapshot.mu = residuals.barrier_mu;

        const SolverStatus quality_status
            = detail::TerminationKernel::classify_solution_quality(config, snapshot);
        if (quality_status == SolverStatus::OPTIMAL) {
            return SolverStatus::OPTIMAL;
        }
        if (quality_status == SolverStatus::FEASIBLE) {
            if (config.print_level >= PrintLevel::INFO) {
                const double feasible_bound = config.tol_con * config.feasible_tol_scale;
                MLOG_INFO("Result: FEASIBLE (Viol: " << residuals.max_primal_inf
                                                     << " <= " << feasible_bound << ")");
            }
            return SolverStatus::FEASIBLE;
        }
        return SolverStatus::UNSOLVED;
    }

    SolverStatus classify_postsolve_failure_(SolverStatus loop_status) const
    {
        switch (loop_status) {
        case SolverStatus::MAX_ITER:
        case SolverStatus::STEP_TOO_SMALL:
        case SolverStatus::INSUFFICIENT_PROGRESS:
        case SolverStatus::RESTORATION_FAILED:
        case SolverStatus::INVALID_INPUT:
            return loop_status;
        case SolverStatus::UNSOLVED:
            return SolverStatus::MAX_ITER;
        case SolverStatus::OPTIMAL:
        case SolverStatus::FEASIBLE:
            // A success-like loop verdict contradicted by fresh primal residuals means the
            // returned iterate is infeasible. This is not used for plain budget exhaustion.
            return SolverStatus::INFEASIBLE;
        case SolverStatus::INFEASIBLE:
        case SolverStatus::LINEAR_SOLVE_FAILED:
        case SolverStatus::NUMERICAL_ERROR:
            return loop_status;
        default:
            return SolverStatus::NUMERICAL_ERROR;
        }
    }

    SolverStatus postsolve(SolverStatus loop_status)
    {
        if (loop_status == SolverStatus::NUMERICAL_ERROR
            || loop_status == SolverStatus::LINEAR_SOLVE_FAILED
            || loop_status == SolverStatus::INVALID_INPUT) {
            record_terminal_info_(loop_status, loop_status);
            return loop_status;
        }
        if (config.print_level >= PrintLevel::INFO) {
            if (loop_status == SolverStatus::MAX_ITER || loop_status == SolverStatus::UNSOLVED) {
                MLOG_INFO("Max iterations reached.");
            } else if (loop_status == SolverStatus::INSUFFICIENT_PROGRESS) {
                MLOG_INFO("Insufficient progress.");
            } else if (loop_status == SolverStatus::RESTORATION_FAILED) {
                MLOG_INFO("Feasibility restoration failed.");
            } else if (loop_status == SolverStatus::STEP_TOO_SMALL) {
                MLOG_INFO("Step size became too small.");
            }
        }
        // Always refresh residuals before the final verdict. This keeps loop-level shortcuts,
        // stagnation exits, and max-iteration exits from returning stale status.
        auto& traj = trajectory.active();
        PostsolveResiduals residuals = refresh_postsolve_residuals_(traj);
        if (!residuals.residuals_ok) {
            record_postsolve_info_(SolverStatus::NUMERICAL_ERROR, loop_status,
                TerminationReason::NUMERICAL_ERROR, residuals);
            return SolverStatus::NUMERICAL_ERROR;
        }

        evaluate_postsolve_dual_residual_(residuals);

        SolverStatus quality_status = classify_postsolve_solution_quality_(residuals);
        if (quality_status == SolverStatus::OPTIMAL || quality_status == SolverStatus::FEASIBLE) {
            TerminationReason reason = context_.info.termination_reason;
            if (reason == TerminationReason::NONE) {
                reason = (quality_status == SolverStatus::OPTIMAL)
                    ? TerminationReason::CONVERGED
                    : TerminationReason::PRIMAL_FEASIBLE;
            }
            record_postsolve_info_(quality_status, loop_status, reason, residuals);
            return quality_status;
        }

        const SolverStatus final_status = classify_postsolve_failure_(loop_status);
        TerminationReason final_reason = context_.info.termination_reason;
        if (final_status == SolverStatus::INFEASIBLE) {
            final_reason = TerminationReason::POSTSOLVE_INFEASIBLE;
        } else if (final_reason == TerminationReason::NONE) {
            final_reason = reason_for_loop_status_(final_status);
        }
        record_postsolve_info_(final_status, loop_status, final_reason, residuals);
        if (config.print_level >= PrintLevel::WARN) {
            const double feasible_bound = config.tol_con * config.feasible_tol_scale;
            MLOG_WARN("Result: " << status_to_string(final_status) << " (Viol: "
                                 << residuals.max_primal_inf << " > " << feasible_bound << ")");
        }
        return final_status;
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
                    if (detail::is_l1_soft_constraint(type, w, config)) {
                        (void)project_l1_soft_pair_to_central_path_(kp, i, context_.solve.mu, w);
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
                const double g_true = detail::true_constraint_value<Model>(kp, i);
                if (detail::is_l1_soft_constraint(type, w, config)) { // L1
                    viol = std::abs(g_true + kp.s(i) - kp.soft_s(i));
                } else if (detail::is_l2_soft_constraint(type, w)) { // L2
                    viol = std::abs(g_true + kp.s(i) - kp.lam(i) / w);
                } else { // Hard
                    viol = std::abs(g_true + kp.s(i));
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

    double compute_unscaled_max_violation(const TrajArray& traj) const
    {
        if (!detail::constraint_row_scaling_active(config)) {
            return compute_max_violation(traj);
        }

        double max_viol = 0.0;
        for (int k = 0; k <= N; ++k) {
            const auto& kp = traj[k];

            for (int i = 0; i < NC; ++i) {
                const double scale = detail::active_constraint_row_scale(kp, config, i);
                const double inv_scale = (scale != 0.0) ? (1.0 / scale) : 1.0;

                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                const double g_raw = detail::unscaled_true_constraint_value<Model>(kp, i, config);
                double viol = 0.0;
                if (detail::is_l1_soft_constraint(type, w, config)) {
                    viol = std::abs(g_raw + kp.s(i) * inv_scale - kp.soft_s(i) * inv_scale);
                } else if (detail::is_l2_soft_constraint(type, w)) {
                    viol = std::abs(g_raw + kp.s(i) * inv_scale - kp.lam(i) / w);
                } else {
                    viol = std::abs(g_raw + kp.s(i) * inv_scale);
                }
                if (viol > max_viol) {
                    max_viol = viol;
                }
            }

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

    SolverPlanInfo make_solver_plan_info_() const
    {
        SolverPlanInfo info;
        info.backend = config.backend;
        info.line_search_type = config.line_search_type;
        info.integrator = config.integrator;
        info.constraint_scaling = config.constraint_scaling;
        info.objective_scaling = config.objective_scaling;
        info.problem_scaling = config.problem_scaling;
        info.fused_riccati_integrator_compatible
            = detail::generated_integrator_matches<Model>(config.integrator);
        info.constraint_scaling_plan_valid = detail::constraint_scaling_plan_valid(config);
        info.constraint_scaling_active
            = info.constraint_scaling_plan_valid && detail::constraint_row_scaling_active(config);
        info.objective_scaling_plan_valid = detail::objective_scaling_plan_valid(config);
        info.objective_scaling_active
            = info.objective_scaling_plan_valid && detail::objective_scaling_active(config);
        info.problem_scaling_plan_valid = detail::problem_scaling_plan_valid(config);
        info.problem_scaling_active
            = info.problem_scaling_plan_valid && detail::problem_scaling_active(config);
        info.linear_solver_ready = static_cast<bool>(linear_solver);
        info.line_search_ready = static_cast<bool>(line_search);
        return info;
    }

    void warn_if_solver_plan_degraded_(const SolverPlanInfo& info) const
    {
        if (!info.constraint_scaling_plan_valid) {
            MLOG_WARN("Constraint row scaling requires finite positive scale bounds.");
        }
        if (!info.objective_scaling_plan_valid) {
            MLOG_WARN("Objective scaling requires a supported method and finite positive bounds.");
        }
        if (!info.problem_scaling_plan_valid) {
            MLOG_WARN(
                "Problem-level scaling requires a supported method and finite positive bounds.");
        }
        if constexpr (detail::has_generated_integrator_v<Model>) {
            if (!info.fused_riccati_integrator_compatible) {
                MLOG_WARN("Model was generated for "
                    << static_cast<int>(Model::generated_integrator) << " but config.integrator is "
                    << static_cast<int>(config.integrator)
                    << ". Fused Riccati kernel will be skipped.");
            }
        }
    }

    void rebuild_solver_components_if_dirty_()
    {
        if (!build_state_.dirty && linear_solver && line_search) {
            return;
        }

        if (!linear_solver) {
            linear_solver = std::make_unique<RiccatiSolver<TrajArray, Model>>();
        }

        if (!no_line_search_) {
            no_line_search_ = std::make_unique<NoLineSearch<Model, MAX_N>>();
        }
        if (!merit_line_search_) {
            merit_line_search_ = std::make_unique<MeritLineSearch<Model, MAX_N>>();
        }
        if (!filter_line_search_) {
            filter_line_search_ = std::make_unique<FilterLineSearch<Model, MAX_N>>();
        }

        const bool line_search_changed
            = !line_search || build_state_.plan.line_search_type != config.line_search_type;
        if (config.line_search_type == LineSearchType::NONE) {
            line_search = no_line_search_.get();
        } else if (config.line_search_type == LineSearchType::MERIT) {
            line_search = merit_line_search_.get();
        } else {
            line_search = filter_line_search_.get();
        }
        if (line_search_changed && line_search) {
            line_search->reset();
        }

        build_state_.plan = make_solver_plan_info_();
        warn_if_solver_plan_degraded_(build_state_.plan);
        build_state_.dirty = false;
    }

    static int validate_horizon_or_throw_(int horizon)
    {
        if (horizon < 0 || horizon > MAX_N) {
            throw std::invalid_argument("MiniSolver horizon outside [0, MAX_N]");
        }
        return horizon;
    }

    // Lookup Maps
    std::unordered_map<std::string, int> state_map;
    std::unordered_map<std::string, int> control_map;
    std::unordered_map<std::string, int> param_map;

    // Components
    // Pass Model type to RiccatiSolver for static constraint info access
    std::unique_ptr<RiccatiSolver<TrajArray, Model>> linear_solver;
    std::unique_ptr<NoLineSearch<Model, MAX_N>> no_line_search_;
    std::unique_ptr<MeritLineSearch<Model, MAX_N>> merit_line_search_;
    std::unique_ptr<FilterLineSearch<Model, MAX_N>> filter_line_search_;
    LineSearchStrategy<Model, MAX_N>* line_search = nullptr;

    // Per-solve line-search α trace (cleared at solve() entry, appended after
    // each step's line search). Purely diagnostic — not captured by snapshots.
    std::vector<double> alpha_log_;

    SolverContext context_;

    TrajectoryType trajectory;

    int N;
    std::array<double, MAX_N> dt_traj;

    SolverTimer timer;

    SolverConfig config;
    SolverBuildState build_state_;

    // --- RTI-lite state ---
    // last_x0 from the previous solve(); used for the state-delta safety gate.
    MSVec<double, Model::NX> rti_lite_last_x0_;
    bool rti_lite_have_previous_solve_ = false;
    bool rti_lite_last_solve_acceptable_ = false;
    int rti_lite_linearization_age_ = 0;

    // --- Coordinate scaling hint (Stage 5 minimal viable) ---
    // Per-coordinate scale factors in user units. The defaults are 1.0 so
    // unconfigured solvers keep byte-for-byte parity with previous behaviour.
    // Values are validated at API entry against config.coordinate_scale_min/max
    // and are only consumed by termination metrics when
    // config.coordinate_scaling == CoordinateScalingMethod::USER_SUPPLIED.
    // Only NU scales: state stationarity is eliminated by Riccati and
    // parameters are not optimisation variables (see set_control_scale doc).
    std::array<double, NU> control_coord_scale_ = ([] {
        std::array<double, NU> s {};
        s.fill(1.0);
        return s;
    })();
};
}
