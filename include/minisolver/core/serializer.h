#pragma once
	#include <fstream>
	#include <vector>
	#include <iostream>
	#include <string>
	#include <array>
	#include <algorithm>
	#include <memory>
	#include "minisolver/solver/solver.h"
	#include "minisolver/core/types.h"

namespace minisolver {

/**
 * @brief Serializer to save/load full solver state (Config, Trajectory, Params)
 * Use this to capture edge cases from production and replay them locally.
 */
template<typename Model, int MAX_N>
class SolverSerializer {
public:
    using SolverType = MiniSolver<Model, MAX_N>;
    static constexpr const char* kCurrentFormatMagic = "MINISOLV_2";
    static constexpr const char* kLegacyFormatMagic = "MINISOLV_1";

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
            std::array<double, Model::NC> lam;
        };
        std::vector<KnotData> trajectory;
    };

    /**
     * @brief [New] Core Interface 1: Capture current solver state into memory
     * This is a very fast operation (memory copy), safe to call in control loops.
     * @param solver The solver instance
     * @param status Optional status to record (default: UNSOLVED)
     */
    static SolverState capture_state(const SolverType& solver, SolverStatus status = SolverStatus::UNSOLVED) {
        SolverState state;
        state.config = solver.config;
        state.N = solver.N;
        state.status = status;
        state.iterations = solver.current_iter;
        state.mu = solver.mu;
        state.reg = solver.reg;
        
        // 1. Copy Time Steps
        state.dt_traj.resize(solver.N);
        for(int k=0; k<solver.N; ++k) {
            state.dt_traj[k] = solver.dt_traj[k];
        }

        // 2. Copy Trajectory (Primal + Dual) & Compute Cost
        state.trajectory.resize(solver.N + 1);
        const auto& active_traj = solver.trajectory.active();
        
        state.total_cost = 0.0;

        for(int k=0; k<=solver.N; ++k) {
            const auto& kp = active_traj[k];
            auto& data = state.trajectory[k];

            state.total_cost += kp.cost;

            // Manual copy to ensure compatibility between MSVec (Eigen/MiniMatrix) and std::array
            for(int i=0; i<Model::NX; ++i) data.x[i] = kp.x(i);
            for(int i=0; i<Model::NU; ++i) data.u[i] = kp.u(i);
            for(int i=0; i<Model::NP; ++i) data.p[i] = kp.p(i);
            for(int i=0; i<Model::NC; ++i) data.s[i] = kp.s(i);
            for(int i=0; i<Model::NC; ++i) data.lam[i] = kp.lam(i);
        }
        
        return state;
    }

    /**
     * @brief [New] Core Interface 2: Save memory snapshot to disk
     * Can be called after solving or in a low-priority thread.
     */
	    static bool save_state(const std::string& filename, const SolverState& state) {
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
	            std::cerr << "[Serializer] Invalid trajectory size in snapshot (expected " << (state.N + 1)
	                      << ", got " << state.trajectory.size() << ").\n";
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
        out.write((char*)&state.config, sizeof(SolverConfig));

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
        for(const auto& knot : state.trajectory) {
            out.write((char*)knot.x.data(), sizeof(double) * NX);
            out.write((char*)knot.u.data(), sizeof(double) * NU);
            out.write((char*)knot.p.data(), sizeof(double) * NP);
            out.write((char*)knot.s.data(), sizeof(double) * NC);
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
    static bool save_case(const std::string& filename, const SolverType& solver) {
        SolverState state = capture_state(solver);
        return save_state(filename, state);
    }

    /**
     * @brief Load a case into the solver.
     */
	    static bool load_case(const std::string& filename, SolverType& solver) {
	        std::ifstream in(filename, std::ios::binary);
	        if (!in) {
	            std::cerr << "[Serializer] Failed to open " << filename << " for reading.\n";
	            return false;
	        }

        auto read_exact = [&in](char* data, std::streamsize size) {
            in.read(data, size);
            return in.good();
        };

        char magic[11] = {0};
        if (!read_exact(magic, 10)) {
            std::cerr << "[Serializer] Failed to read file header.\n";
            return false;
        }
        std::string version(magic);
        
        if (version == kLegacyFormatMagic) {
            std::cerr << "[Serializer] Unsupported legacy format MINISOLV_1. "
                      << "SolverConfig layout changed; regenerate the snapshot with a newer build.\n";
            return false;
        }

        // Only accept the current version
        if (version != kCurrentFormatMagic) {
            std::cerr << "[Serializer] Invalid file format or version: " << version << "\n";
            return false;
        }
        
	        int N, NX, NU, NP, NC;
	        if (!read_exact((char*)&N, sizeof(N)) ||
	            !read_exact((char*)&NX, sizeof(NX)) ||
	            !read_exact((char*)&NU, sizeof(NU)) ||
	            !read_exact((char*)&NP, sizeof(NP)) ||
	            !read_exact((char*)&NC, sizeof(NC))) {
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
	        if (!read_exact((char*)&cfg, sizeof(SolverConfig))) {
	            std::cerr << "[Serializer] Failed to read solver configuration.\n";
	            return false;
	        }
	        
	        // Read Stats
	        int status_i;
	        int iter;
	        double cost;
	        double mu_val, reg_val;
	        if (!read_exact((char*)&status_i, sizeof(status_i)) ||
	            !read_exact((char*)&iter, sizeof(iter)) ||
	            !read_exact((char*)&cost, sizeof(cost)) ||
	            !read_exact((char*)&mu_val, sizeof(mu_val)) ||
	            !read_exact((char*)&reg_val, sizeof(reg_val))) {
	            std::cerr << "[Serializer] Failed to read solver metadata.\n";
	            return false;
	        }
	        
	        std::cout << "[Serializer] Info - Saved Status: " << status_to_string((SolverStatus)status_i)
	                  << ", Iters: " << iter << ", Cost: " << cost << "\n";

	        // Read remaining data into local temporaries first to keep load atomic.
	        std::vector<double> dt_local(static_cast<size_t>(N));
	        if (N > 0 && !read_exact(reinterpret_cast<char*>(dt_local.data()), sizeof(double) * N)) {
	            std::cerr << "[Serializer] Failed to read time-step vector.\n";
	            return false;
	        }

	        std::vector<typename SolverState::KnotData> knots(static_cast<size_t>(N + 1));
	        for(int k=0; k<=N; ++k) {
	            auto& knot = knots[static_cast<size_t>(k)];
	            if (!read_exact(reinterpret_cast<char*>(knot.x.data()), sizeof(double) * NX) ||
	                !read_exact(reinterpret_cast<char*>(knot.u.data()), sizeof(double) * NU) ||
	                !read_exact(reinterpret_cast<char*>(knot.p.data()), sizeof(double) * NP) ||
	                !read_exact(reinterpret_cast<char*>(knot.s.data()), sizeof(double) * NC) ||
	                !read_exact(reinterpret_cast<char*>(knot.lam.data()), sizeof(double) * NC)) {
	                std::cerr << "[Serializer] Failed to read trajectory data.\n";
	                return false;
	            }
	        }

	        bool has_trailing = (in.peek() != std::ifstream::traits_type::eof());

	        // Commit to solver only after the full read succeeds.
	        solver.resize_horizon(N);
	        solver.config = cfg;
	        solver.backend = cfg.backend;
	        solver.current_iter = iter;
	        solver.mu = mu_val;
	        solver.reg = reg_val;

	        solver.rebuild_solver_components();
	        solver.components_dirty = false;

	        for(int k=0; k<N; ++k) {
	            solver.dt_traj[k] = dt_local[static_cast<size_t>(k)];
	        }

	        auto& traj = solver.trajectory.active();
	        for(int k=0; k<=N; ++k) {
	            const auto& src = knots[static_cast<size_t>(k)];
	            std::copy_n(src.x.data(), NX, traj[k].x.data());
	            std::copy_n(src.u.data(), NU, traj[k].u.data());
	            std::copy_n(src.p.data(), NP, traj[k].p.data());
	            std::copy_n(src.s.data(), NC, traj[k].s.data());
	            std::copy_n(src.lam.data(), NC, traj[k].lam.data());
	        }

	        // Keep candidate buffer consistent with the loaded active buffer.
	        solver.trajectory.prepare_candidate();

	        for(int k=0; k<=N; ++k) {
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
