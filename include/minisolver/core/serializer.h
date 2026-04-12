#pragma once
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace minisolver {

/**
 * @brief Serializer to save/load full solver state (Config, Trajectory, Params)
 * Use this to capture edge cases from production and replay them locally.
 */
template <typename Model, int MAX_N> class SolverSerializer {
public:
    using SolverType = MiniSolver<Model, MAX_N>;
    static constexpr const char* kCurrentFormatMagic = "MINISOLV_3";
    static constexpr const char* kLegacyFormatMagicV2 = "MINISOLV_2";
    static constexpr const char* kLegacyFormatMagicV1 = "MINISOLV_1";

    // =============================================================
    // [New] Memory Snapshot Structure
    // Used to buffer solver state in memory to avoid blocking control loop with I/O
    // =============================================================
    struct SolverState {
        SolverConfig config;
        int N;
        std::vector<double> dt_traj;

        // Metadata
        SolverStatus status = SolverStatus::UNSOLVED;
        int iterations = 0;
        double total_cost = 0.0;
        double mu = 0.0;
        double reg = 0.0;

        // Flattened trajectory data for serialization
        // Structure: [k=0...N]
        struct KnotData {
            std::array<double, Model::NX> x;
            std::array<double, Model::NU> u;
            std::array<double, Model::NP> p;
            std::array<double, Model::NC> s;
            std::array<double, Model::NC> soft_s;
            std::array<double, Model::NC> lam;
        };
        std::vector<KnotData> trajectory;
    };

private:
    template <typename T> static void write_pod(std::ofstream& out, const T& v)
    {
        out.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }

    template <typename T> static bool read_pod(std::ifstream& in, T& v)
    {
        in.read(reinterpret_cast<char*>(&v), sizeof(T));
        return in.good();
    }

    template <typename Enum> static void write_enum(std::ofstream& out, Enum e)
    {
        std::int32_t v = static_cast<std::int32_t>(e);
        write_pod(out, v);
    }

    template <typename Enum> static bool read_enum(std::ifstream& in, Enum& e)
    {
        std::int32_t v = 0;
        if (!read_pod(in, v))
            return false;
        e = static_cast<Enum>(v);
        return true;
    }

    static void write_bool(std::ofstream& out, bool v)
    {
        std::uint8_t b = v ? 1u : 0u;
        write_pod(out, b);
    }

    static bool read_bool(std::ifstream& in, bool& v)
    {
        std::uint8_t b = 0;
        if (!read_pod(in, b))
            return false;
        v = (b != 0);
        return true;
    }

    static void write_config(std::ofstream& out, const SolverConfig& cfg)
    {
        write_enum(out, cfg.backend);
        write_enum(out, cfg.initialization);

        write_enum(out, cfg.integrator);
        write_pod(out, cfg.default_dt);

        write_enum(out, cfg.barrier_strategy);
        write_pod(out, cfg.mu_init);
        write_pod(out, cfg.mu_final);
        write_pod(out, cfg.mu_linear_decrease_factor);
        write_pod(out, cfg.barrier_tolerance_factor);
        write_pod(out, cfg.mu_safety_margin);

        write_enum(out, cfg.inertia_strategy);
        write_pod(out, cfg.reg_init);
        write_pod(out, cfg.reg_min);
        write_pod(out, cfg.reg_max);
        write_pod(out, cfg.reg_scale_up);
        write_pod(out, cfg.reg_scale_down);
        write_pod(out, cfg.regularization_step);
        write_pod(out, cfg.singular_threshold);
        write_pod(out, cfg.huge_penalty);
        std::int32_t inertia_max_retries = static_cast<std::int32_t>(cfg.inertia_max_retries);
        write_pod(out, inertia_max_retries);

        write_pod(out, cfg.tol_grad);
        write_pod(out, cfg.tol_con);
        write_pod(out, cfg.tol_dual);
        write_pod(out, cfg.tol_mu);
        write_pod(out, cfg.tol_cost);
        write_pod(out, cfg.feasible_tol_scale);

        write_enum(out, cfg.line_search_type);
        std::int32_t line_search_max_iters = static_cast<std::int32_t>(cfg.line_search_max_iters);
        write_pod(out, line_search_max_iters);
        write_pod(out, cfg.line_search_tau);
        write_pod(out, cfg.line_search_backtrack_factor);
        write_pod(out, cfg.filter_gamma_theta);
        write_pod(out, cfg.filter_gamma_phi);

        write_pod(out, cfg.min_barrier_slack);
        write_pod(out, cfg.barrier_inf_cost);
        write_pod(out, cfg.slack_reset_trigger);
        write_pod(out, cfg.warm_start_slack_init);
        write_pod(out, cfg.soc_trigger_alpha);
        write_pod(out, cfg.merit_nu_init);
        write_pod(out, cfg.eta_suff_descent);

        std::int32_t max_restoration_iters = static_cast<std::int32_t>(cfg.max_restoration_iters);
        write_pod(out, max_restoration_iters);
        write_pod(out, cfg.restoration_mu);
        write_pod(out, cfg.restoration_reg);
        write_pod(out, cfg.restoration_alpha);

        std::int32_t max_iters = static_cast<std::int32_t>(cfg.max_iters);
        write_pod(out, max_iters);
        write_enum(out, cfg.print_level);
        write_bool(out, cfg.enable_profiling);

        write_enum(out, cfg.hessian_approximation);
        write_bool(out, cfg.enable_iterative_refinement);
        std::int32_t max_refinement_steps = static_cast<std::int32_t>(cfg.max_refinement_steps);
        write_pod(out, max_refinement_steps);
        write_bool(out, cfg.enable_rti);
        write_bool(out, cfg.enable_line_search_rollout);

        write_bool(out, cfg.enable_defect_correction);
        write_bool(out, cfg.enable_corrector);
        write_bool(out, cfg.enable_aggressive_barrier);

        write_bool(out, cfg.enable_slack_reset);
        write_bool(out, cfg.enable_feasibility_restoration);
        write_bool(out, cfg.enable_soc);
    }

    static bool read_config(std::ifstream& in, SolverConfig& cfg)
    {
        if (!read_enum(in, cfg.backend) || !read_enum(in, cfg.initialization)
            || !read_enum(in, cfg.integrator) || !read_pod(in, cfg.default_dt)
            || !read_enum(in, cfg.barrier_strategy) || !read_pod(in, cfg.mu_init)
            || !read_pod(in, cfg.mu_final) || !read_pod(in, cfg.mu_linear_decrease_factor)
            || !read_pod(in, cfg.barrier_tolerance_factor) || !read_pod(in, cfg.mu_safety_margin)
            || !read_enum(in, cfg.inertia_strategy) || !read_pod(in, cfg.reg_init)
            || !read_pod(in, cfg.reg_min) || !read_pod(in, cfg.reg_max)
            || !read_pod(in, cfg.reg_scale_up) || !read_pod(in, cfg.reg_scale_down)
            || !read_pod(in, cfg.regularization_step) || !read_pod(in, cfg.singular_threshold)
            || !read_pod(in, cfg.huge_penalty)) {
            return false;
        }

        std::int32_t inertia_max_retries = 0;
        if (!read_pod(in, inertia_max_retries))
            return false;
        cfg.inertia_max_retries = static_cast<int>(inertia_max_retries);

        if (!read_pod(in, cfg.tol_grad) || !read_pod(in, cfg.tol_con) || !read_pod(in, cfg.tol_dual)
            || !read_pod(in, cfg.tol_mu) || !read_pod(in, cfg.tol_cost)
            || !read_pod(in, cfg.feasible_tol_scale) || !read_enum(in, cfg.line_search_type)) {
            return false;
        }

        std::int32_t line_search_max_iters = 0;
        if (!read_pod(in, line_search_max_iters))
            return false;
        cfg.line_search_max_iters = static_cast<int>(line_search_max_iters);

        if (!read_pod(in, cfg.line_search_tau) || !read_pod(in, cfg.line_search_backtrack_factor)
            || !read_pod(in, cfg.filter_gamma_theta) || !read_pod(in, cfg.filter_gamma_phi)
            || !read_pod(in, cfg.min_barrier_slack) || !read_pod(in, cfg.barrier_inf_cost)
            || !read_pod(in, cfg.slack_reset_trigger) || !read_pod(in, cfg.warm_start_slack_init)
            || !read_pod(in, cfg.soc_trigger_alpha) || !read_pod(in, cfg.merit_nu_init)
            || !read_pod(in, cfg.eta_suff_descent)) {
            return false;
        }

        std::int32_t max_restoration_iters = 0;
        if (!read_pod(in, max_restoration_iters))
            return false;
        cfg.max_restoration_iters = static_cast<int>(max_restoration_iters);

        if (!read_pod(in, cfg.restoration_mu) || !read_pod(in, cfg.restoration_reg)
            || !read_pod(in, cfg.restoration_alpha)) {
            return false;
        }

        std::int32_t max_iters = 0;
        if (!read_pod(in, max_iters))
            return false;
        cfg.max_iters = static_cast<int>(max_iters);

        if (!read_enum(in, cfg.print_level) || !read_bool(in, cfg.enable_profiling)
            || !read_enum(in, cfg.hessian_approximation)
            || !read_bool(in, cfg.enable_iterative_refinement)) {
            return false;
        }

        std::int32_t max_refinement_steps = 0;
        if (!read_pod(in, max_refinement_steps))
            return false;
        cfg.max_refinement_steps = static_cast<int>(max_refinement_steps);

        if (!read_bool(in, cfg.enable_rti) || !read_bool(in, cfg.enable_line_search_rollout)
            || !read_bool(in, cfg.enable_defect_correction) || !read_bool(in, cfg.enable_corrector)
            || !read_bool(in, cfg.enable_aggressive_barrier)
            || !read_bool(in, cfg.enable_slack_reset)
            || !read_bool(in, cfg.enable_feasibility_restoration)
            || !read_bool(in, cfg.enable_soc)) {
            return false;
        }

        return true;
    }

public:
    /**
     * @brief [New] Core Interface 1: Capture current solver state into memory
     * This is a very fast operation (memory copy), safe to call in control loops.
     * @param solver The solver instance
     * @param status Optional status to record (default: UNSOLVED)
     */
    static SolverState capture_state(
        const SolverType& solver, SolverStatus status = SolverStatus::UNSOLVED)
    {
        SolverState state;
        state.config = solver.config;
        state.N = solver.N;
        state.status = status;
        state.iterations = solver.current_iter;
        state.mu = solver.mu;
        state.reg = solver.reg;

        // 1. Copy Time Steps
        state.dt_traj.resize(solver.N);
        for (int k = 0; k < solver.N; ++k) {
            state.dt_traj[k] = solver.dt_traj[k];
        }

        // 2. Copy Trajectory (Primal + Dual) & Compute Cost
        state.trajectory.resize(solver.N + 1);
        const auto& active_traj = solver.trajectory.active();

        state.total_cost = 0.0;

        for (int k = 0; k <= solver.N; ++k) {
            const auto& kp = active_traj[k];
            auto& data = state.trajectory[k];

            state.total_cost += kp.cost;

            // Manual copy to ensure compatibility between MSVec (Eigen/MiniMatrix) and std::array
            for (int i = 0; i < Model::NX; ++i)
                data.x[i] = kp.x(i);
            for (int i = 0; i < Model::NU; ++i)
                data.u[i] = kp.u(i);
            for (int i = 0; i < Model::NP; ++i)
                data.p[i] = kp.p(i);
            for (int i = 0; i < Model::NC; ++i)
                data.s[i] = kp.s(i);
            for (int i = 0; i < Model::NC; ++i)
                data.soft_s[i] = kp.soft_s(i);
            for (int i = 0; i < Model::NC; ++i)
                data.lam[i] = kp.lam(i);
        }

        return state;
    }

    /**
     * @brief [New] Core Interface 2: Save memory snapshot to disk
     * Can be called after solving or in a low-priority thread.
     */
    static bool save_state(const std::string& filename, const SolverState& state)
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "[Serializer] Failed to open " << filename << " for writing.\n";
            return false;
        }
        if (state.N < 0) {
            std::cerr << "[Serializer] Invalid negative horizon in snapshot.\n";
            return false;
        }
        if (static_cast<int>(state.dt_traj.size()) != state.N) {
            std::cerr << "[Serializer] Invalid dt_traj size in snapshot (expected " << state.N
                      << ", got " << state.dt_traj.size() << ").\n";
            return false;
        }
        if (static_cast<int>(state.trajectory.size()) != state.N + 1) {
            std::cerr << "[Serializer] Invalid trajectory size in snapshot (expected "
                      << (state.N + 1) << ", got " << state.trajectory.size() << ").\n";
            return false;
        }

        // 1. Header & Version
        // Since this is the first release version, we use a single magic identifier.
        out.write(kCurrentFormatMagic, 10);

        // 2. Dimensions & Horizon
        int NX = Model::NX;
        int NU = Model::NU;
        int NP = Model::NP;
        int NC = Model::NC;
        out.write((char*)&state.N, sizeof(state.N));
        out.write((char*)&NX, sizeof(NX));
        out.write((char*)&NU, sizeof(NU));
        out.write((char*)&NP, sizeof(NP));
        out.write((char*)&NC, sizeof(NC));

        // 3. Configuration
        write_config(out, state.config);

        // 4. Status & Stats
        int status_i = static_cast<int>(state.status);
        out.write((char*)&status_i, sizeof(status_i));
        out.write((char*)&state.iterations, sizeof(state.iterations));
        out.write((char*)&state.total_cost, sizeof(state.total_cost));
        out.write((char*)&state.mu, sizeof(state.mu));
        out.write((char*)&state.reg, sizeof(state.reg));

        // 5. Time Steps
        if (!state.dt_traj.empty()) {
            out.write((char*)state.dt_traj.data(), sizeof(double) * state.N);
        }

        // 6. Trajectory
        for (const auto& knot : state.trajectory) {
            out.write((char*)knot.x.data(), sizeof(double) * NX);
            out.write((char*)knot.u.data(), sizeof(double) * NU);
            out.write((char*)knot.p.data(), sizeof(double) * NP);
            out.write((char*)knot.s.data(), sizeof(double) * NC);
            out.write((char*)knot.soft_s.data(), sizeof(double) * NC);
            out.write((char*)knot.lam.data(), sizeof(double) * NC);
        }

        if (out.bad()) {
            std::cerr << "[Serializer] I/O Error during write.\n";
            return false;
        }

        std::cout << "[Serializer] Snapshot saved to " << filename << "\n";
        return true;
    }

    /**
     * @brief Backward compatible wrapper
     * Now combines capture + save
     */
    static bool save_case(const std::string& filename, const SolverType& solver)
    {
        SolverState state = capture_state(solver);
        return save_state(filename, state);
    }

    /**
     * @brief Load a case into the solver.
     */
    static bool load_case(const std::string& filename, SolverType& solver)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cerr << "[Serializer] Failed to open " << filename << " for reading.\n";
            return false;
        }

        auto read_exact = [&in](char* data, std::streamsize size) {
            in.read(data, size);
            return in.good();
        };

        char magic[11] = { 0 };
        if (!read_exact(magic, 10)) {
            std::cerr << "[Serializer] Failed to read file header.\n";
            return false;
        }
        std::string version(magic);

        if (version == kLegacyFormatMagicV2 || version == kLegacyFormatMagicV1) {
            std::cerr << "[Serializer] Unsupported snapshot format " << version
                      << ". Regenerate the snapshot with the current build.\n";
            return false;
        }

        // Only accept the current version
        if (version != kCurrentFormatMagic) {
            std::cerr << "[Serializer] Invalid file format or version: " << version << "\n";
            return false;
        }

        int N, NX, NU, NP, NC;
        if (!read_exact((char*)&N, sizeof(N)) || !read_exact((char*)&NX, sizeof(NX))
            || !read_exact((char*)&NU, sizeof(NU)) || !read_exact((char*)&NP, sizeof(NP))
            || !read_exact((char*)&NC, sizeof(NC))) {
            std::cerr << "[Serializer] Failed to read model dimensions.\n";
            return false;
        }

        if (NX != Model::NX || NU != Model::NU || NP != Model::NP || NC != Model::NC) {
            std::cerr << "[Serializer] Model dimension mismatch!\n";
            return false;
        }

        if (N < 0) {
            std::cerr << "[Serializer] Invalid negative horizon in snapshot.\n";
            return false;
        }

        if (N > MAX_N) {
            std::cerr << "[Serializer] Snapshot horizon N=" << N << " exceeds MAX_N=" << MAX_N
                      << ". Refuse to truncate; rebuild the replay binary with larger MAX_N.\n";
            return false;
        }

        SolverConfig cfg;
        if (!read_config(in, cfg)) {
            std::cerr << "[Serializer] Failed to read solver configuration.\n";
            return false;
        }

        // Read Stats
        int status_i;
        int iter;
        double cost;
        double mu_val, reg_val;
        if (!read_exact((char*)&status_i, sizeof(status_i))
            || !read_exact((char*)&iter, sizeof(iter)) || !read_exact((char*)&cost, sizeof(cost))
            || !read_exact((char*)&mu_val, sizeof(mu_val))
            || !read_exact((char*)&reg_val, sizeof(reg_val))) {
            std::cerr << "[Serializer] Failed to read solver metadata.\n";
            return false;
        }

        std::cout << "[Serializer] Info - Saved Status: "
                  << status_to_string((SolverStatus)status_i) << ", Iters: " << iter
                  << ", Cost: " << cost << "\n";

        // Read remaining data into local temporaries first to keep load atomic.
        std::vector<double> dt_local(static_cast<size_t>(N));
        if (N > 0 && !read_exact(reinterpret_cast<char*>(dt_local.data()), sizeof(double) * N)) {
            std::cerr << "[Serializer] Failed to read time-step vector.\n";
            return false;
        }

        std::vector<typename SolverState::KnotData> knots(static_cast<size_t>(N + 1));
        for (int k = 0; k <= N; ++k) {
            auto& knot = knots[static_cast<size_t>(k)];
            if (!read_exact(reinterpret_cast<char*>(knot.x.data()), sizeof(double) * NX)
                || !read_exact(reinterpret_cast<char*>(knot.u.data()), sizeof(double) * NU)
                || !read_exact(reinterpret_cast<char*>(knot.p.data()), sizeof(double) * NP)
                || !read_exact(reinterpret_cast<char*>(knot.s.data()), sizeof(double) * NC)
                || !read_exact(reinterpret_cast<char*>(knot.soft_s.data()), sizeof(double) * NC)
                || !read_exact(reinterpret_cast<char*>(knot.lam.data()), sizeof(double) * NC)) {
                std::cerr << "[Serializer] Failed to read trajectory data.\n";
                return false;
            }
        }

        bool has_trailing = (in.peek() != std::ifstream::traits_type::eof());

        // Commit to solver only after the full read succeeds.
        solver.resize_horizon(N);
        solver.config = cfg;
        solver.current_iter = iter;
        solver.mu = mu_val;
        solver.reg = reg_val;

        solver.rebuild_solver_components();
        solver.components_dirty = false;

        for (int k = 0; k < N; ++k) {
            solver.dt_traj[k] = dt_local[static_cast<size_t>(k)];
        }

        auto& traj = solver.trajectory.active();
        for (int k = 0; k <= N; ++k) {
            const auto& src = knots[static_cast<size_t>(k)];
            for (int i = 0; i < NX; ++i)
                traj[k].x(i) = src.x[static_cast<size_t>(i)];
            for (int i = 0; i < NU; ++i)
                traj[k].u(i) = src.u[static_cast<size_t>(i)];
            for (int i = 0; i < NP; ++i)
                traj[k].p(i) = src.p[static_cast<size_t>(i)];
            for (int i = 0; i < NC; ++i)
                traj[k].s(i) = src.s[static_cast<size_t>(i)];
            for (int i = 0; i < NC; ++i)
                traj[k].soft_s(i) = src.soft_s[static_cast<size_t>(i)];
            for (int i = 0; i < NC; ++i)
                traj[k].lam(i) = src.lam[static_cast<size_t>(i)];
        }

        // Keep candidate buffer consistent with the loaded active buffer.
        solver.trajectory.prepare_candidate();

        for (int k = 0; k <= N; ++k) {
            double current_dt = (k < N) ? solver.dt_traj[k] : 0.0;
            if (solver.config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                Model::compute_cost_gn(traj[k]);
                Model::compute_dynamics(traj[k], solver.config.integrator, current_dt);
                Model::compute_constraints(traj[k]);
            } else {
                Model::compute_cost_exact(traj[k]);
                Model::compute_dynamics(traj[k], solver.config.integrator, current_dt);
                Model::compute_constraints(traj[k]);
            }
        }

        if (has_trailing) {
            std::cerr << "[Serializer] Warning: trailing bytes detected in snapshot.\n";
        }

        std::cout << "[Serializer] Case loaded successfully.\n";
        return true;
    }
};

}
