#pragma once
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <array>
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
        auto* active_state = solver.trajectory.get_active_state();
        
        state.total_cost = 0.0;

        for(int k=0; k<=solver.N; ++k) {
            const auto& kp = active_state[k];
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

        // 1. Header & Version
        // Since this is the first release version, we use a single magic identifier.
        const char* magic = "MINISOLV_1"; 
        out.write(magic, 10);
        
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

        char magic[11] = {0};
        in.read(magic, 10);
        std::string version(magic);
        
        // Only accept the current version
        if (version != "MINISOLV_1") {
            std::cerr << "[Serializer] Invalid file format or version: " << version << "\n";
            return false;
        }
        
        int N, NX, NU, NP, NC;
        in.read((char*)&N, sizeof(N));
        in.read((char*)&NX, sizeof(NX));
        in.read((char*)&NU, sizeof(NU));
        in.read((char*)&NP, sizeof(NP));
        in.read((char*)&NC, sizeof(NC));

        if (NX != Model::NX || NU != Model::NU || NP != Model::NP || NC != Model::NC) {
            std::cerr << "[Serializer] Model dimension mismatch!\n";
            return false;
        }

        if (N > MAX_N) N = MAX_N;
        solver.resize_horizon(N);

        SolverConfig cfg;
        in.read((char*)&cfg, sizeof(SolverConfig));
        solver.config = cfg;
        
        // Read Stats
        int status_i;
        int iter;
        double cost;
        in.read((char*)&status_i, sizeof(status_i));
        in.read((char*)&iter, sizeof(iter));
        in.read((char*)&cost, sizeof(cost));
        double mu_val, reg_val;
        in.read((char*)&mu_val, sizeof(mu_val));
        in.read((char*)&reg_val, sizeof(reg_val));
        
        std::cout << "[Serializer] Info - Saved Status: " << status_to_string((SolverStatus)status_i)
                  << ", Iters: " << iter << ", Cost: " << cost << "\n";

        solver.mu = mu_val;
        solver.reg = reg_val;
        
        // Update components based on config
        if (cfg.line_search_type == LineSearchType::MERIT) {
            solver.line_search = std::make_unique<MeritLineSearch<Model, MAX_N>>();
        } else {
            solver.line_search = std::make_unique<FilterLineSearch<Model, MAX_N>>();
        }

        in.read((char*)solver.dt_traj.data(), sizeof(double) * N);

        auto* state = solver.trajectory.get_active_state();
        for(int k=0; k<=N; ++k) {
            in.read((char*)state[k].x.data(), sizeof(double) * NX);
            in.read((char*)state[k].u.data(), sizeof(double) * NU);
            in.read((char*)state[k].p.data(), sizeof(double) * NP);
            in.read((char*)state[k].s.data(), sizeof(double) * NC);
            in.read((char*)state[k].lam.data(), sizeof(double) * NC);
        }
        
        std::cout << "[Serializer] Case loaded successfully.\n";
        return true;
    }
};

}
