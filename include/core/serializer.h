#pragma once
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include "solver/solver.h"

namespace minisolver {

/**
 * @brief Serializer to save/load full solver state (Config, Trajectory, Params)
 * Use this to capture edge cases from production and replay them locally.
 */
template<typename Model, int MAX_N>
class SolverSerializer {
public:
    using SolverType = MiniSolver<Model, MAX_N>;

    /**
     * @brief Save the current solver state to a binary file.
     * Captures: Config, N, Dimensions, dt, x, u, p, s, lam.
     */
    static bool save_case(const std::string& filename, const SolverType& solver) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "[Serializer] Failed to open " << filename << " for writing.\n";
            return false;
        }

        // 1. Header & Version
        const char* magic = "MS_CASE_V2"; // Updated version for Duals
        out.write(magic, 10);
        
        // 2. Dimensions & Horizon
        int N = solver.N;
        int NX = Model::NX;
        int NU = Model::NU;
        int NP = Model::NP;
        int NC = Model::NC;
        out.write((char*)&N, sizeof(N));
        out.write((char*)&NX, sizeof(NX));
        out.write((char*)&NU, sizeof(NU));
        out.write((char*)&NP, sizeof(NP));
        out.write((char*)&NC, sizeof(NC));

        // 3. Configuration
        out.write((char*)&solver.config, sizeof(SolverConfig));

        // 4. Time Steps
        out.write((char*)solver.dt_traj.data(), sizeof(double) * N);

        // 5. Trajectory (Primal + Dual)
        const auto& traj = solver.trajectory.active();
        for(int k=0; k<=N; ++k) {
            out.write((char*)traj[k].x.data(), sizeof(double) * NX);
            out.write((char*)traj[k].u.data(), sizeof(double) * NU);
            out.write((char*)traj[k].p.data(), sizeof(double) * NP);
            out.write((char*)traj[k].s.data(), sizeof(double) * NC);
            out.write((char*)traj[k].lam.data(), sizeof(double) * NC);
        }

        if (out.bad()) {
             std::cerr << "[Serializer] I/O Error during write.\n";
             return false;
        }
        
        std::cout << "[Serializer] Case saved to " << filename << "\n";
        return true;
    }

    /**
     * @brief Load a case into the solver.
     * Resizes solver, updates config, and populates trajectory (x, u, p, s, lam).
     */
    static bool load_case(const std::string& filename, SolverType& solver) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cerr << "[Serializer] Failed to open " << filename << " for reading.\n";
            return false;
        }

        // 1. Header
        char magic[11] = {0};
        in.read(magic, 10);
        std::string version(magic);
        
        if (version != "MS_CASE_V1" && version != "MS_CASE_V2") {
            std::cerr << "[Serializer] Invalid file format or version: " << version << "\n";
            return false;
        }
        
        bool has_duals = (version == "MS_CASE_V2");

        // 2. Dimensions
        int N, NX, NU, NP, NC;
        in.read((char*)&N, sizeof(N));
        in.read((char*)&NX, sizeof(NX));
        in.read((char*)&NU, sizeof(NU));
        in.read((char*)&NP, sizeof(NP));
        in.read((char*)&NC, sizeof(NC));

        if (NX != Model::NX || NU != Model::NU || NP != Model::NP || NC != Model::NC) {
            std::cerr << "[Serializer] Model dimension mismatch! \n"
                      << "File: " << NX << "," << NU << "," << NP << "," << NC << "\n"
                      << "Code: " << Model::NX << "," << Model::NU << "," << Model::NP << "," << Model::NC << "\n";
            return false;
        }

        // Resize
        if (N > MAX_N) {
            std::cerr << "[Serializer] Warning: Case N (" << N << ") > MAX_N (" << MAX_N << "). Truncating.\n";
            N = MAX_N;
        }
        solver.resize_horizon(N);

        // 3. Configuration
        SolverConfig cfg;
        in.read((char*)&cfg, sizeof(SolverConfig));
        solver.config = cfg;
        
        // Re-init internal state based on new config
        solver.mu = cfg.mu_init;
        solver.reg = cfg.reg_init;
        
        // Update components if strategy changed
        if (cfg.line_search_type == LineSearchType::MERIT) {
            solver.line_search = std::make_unique<MeritLineSearch<Model, MAX_N>>();
        } else {
            solver.line_search = std::make_unique<FilterLineSearch<Model, MAX_N>>();
        }

        // 4. Time Steps
        in.read((char*)solver.dt_traj.data(), sizeof(double) * N);

        // 5. Trajectory
        auto& traj = solver.trajectory.active();
        for(int k=0; k<=N; ++k) {
            in.read((char*)traj[k].x.data(), sizeof(double) * NX);
            in.read((char*)traj[k].u.data(), sizeof(double) * NU);
            in.read((char*)traj[k].p.data(), sizeof(double) * NP);
            
            if (has_duals) {
                in.read((char*)traj[k].s.data(), sizeof(double) * NC);
                in.read((char*)traj[k].lam.data(), sizeof(double) * NC);
            } else {
                // V1 Fallback: Reset to defaults
                traj[k].s.fill(1.0);
                traj[k].lam.fill(1.0);
            }
        }
        
        std::cout << "[Serializer] Case loaded successfully (" << version << ").\n";
        return true;
    }
};

}
