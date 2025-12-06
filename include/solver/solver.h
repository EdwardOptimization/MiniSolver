#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "core/types.h"
#include "solver/kkt_assembler.h"
#include "solver/riccati_recursive.h"
#include "solver/backend_interface.h"

namespace roboopt {

struct SolverConfig {
    IntegratorType integrator = IntegratorType::EULER_EXPLICIT;
    double default_dt = 0.1; // Default if vector not set
    double mu_init = 0.1;
    int max_iters = 20;
    bool verbose = true;
};

template<typename Model>
class PDIPMSolver {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;

    using Knot = KnotPoint<double, NX, NU, NC, NP>;

    std::vector<Knot> traj;
    std::vector<double> dt_traj; // Vector of dt per step
    Backend backend;
    SolverConfig config;
    double mu; // Barrier Parameter

    PDIPMSolver(int N, Backend b, SolverConfig conf = SolverConfig()) 
        : backend(b), config(conf), mu(conf.mu_init) {
        traj.resize(N + 1);
        dt_traj.resize(N, conf.default_dt); // Initialize with default dt
        
        // Initialize slacks/duals strictly positive
        for(auto& kp : traj) kp.initialize_defaults();
    }

    void set_params(const double* p) {
        for(auto& k : traj) {
            for(int i=0; i<NP; ++i) k.p(i) = p[i];
        }
    }

    // Set dynamic dt profile
    void set_dt(const std::vector<double>& dts) {
        if(dts.size() != dt_traj.size()) {
            std::cerr << "Warning: DT vector size mismatch. Expected " << dt_traj.size() << "\n";
            return;
        }
        dt_traj = dts;
    }
    
    // Set constant dt
    void set_dt(double dt) {
        std::fill(dt_traj.begin(), dt_traj.end(), dt);
    }

    // --- Line Search (Fraction-to-Boundary) ---
    double fraction_to_boundary_rule(double tau = 0.995) {
        double alpha_s = 1.0;
        double alpha_lam = 1.0;

        for(const auto& kp : traj) {
            for(int i=0; i<NC; ++i) {
                double s = kp.s(i);
                double ds = kp.ds(i);
                double lam = kp.lam(i);
                double dlam = kp.dlam(i);

                if (ds < 0) {
                    alpha_s = std::min(alpha_s, -tau * s / ds);
                }
                if (dlam < 0) {
                    alpha_lam = std::min(alpha_lam, -tau * lam / dlam);
                }
            }
        }
        return std::min(alpha_s, alpha_lam);
    }

    void step() {
        // 1. Compute Derivatives & Residuals at current point
        //    Pass specific dt for each knot point
        for(size_t k=0; k < traj.size(); ++k) {
            // Last point doesn't really have a dt for dynamics, but compute() might use it for costs?
            // Usually compute() calculates derivatives for x_{k+1} = f(x_k, u_k).
            // So we use dt_traj[k] for k < N. For k=N, dt is irrelevant or 0.
            double current_dt = (k < dt_traj.size()) ? dt_traj[k] : 0.0;
            Model::compute(traj[k], config.integrator, current_dt);
        }

        // 2. Update Barrier Parameter (Simple Strategy)
        // Average duality gap
        double total_gap = 0.0;
        int total_con = 0;
        for(const auto& kp : traj) {
            total_gap += kp.s.dot(kp.lam);
            total_con += NC;
        }
        if (total_con > 0) {
            double avg_gap = total_gap / total_con;
            // Aggressive decrease: Target 0.1 of current gap, bounded
            mu = std::min(0.1, avg_gap * 0.1); 
            // Safety floor
            if(mu < 1e-6) mu = 1e-6;
        } else {
            mu = 1e-6; 
        }

        // 3. Solve Newton Step (Modified Riccati)
        cpu_serial_solve(traj, mu);

        // 4. Line Search
        double alpha = fraction_to_boundary_rule(0.99);

        // 5. Update Variables
        // Standard Damped Update
        for(size_t k=0; k<traj.size(); ++k) {
            auto& kp = traj[k];
            
            // Primal
            kp.x += alpha * kp.dx;
            kp.u += alpha * kp.du;

            // Dual / Slack
            kp.s += alpha * kp.ds;
            kp.lam += alpha * kp.dlam;
        }

        // 6. Simulate Dynamics (Rollout) to ensure physical consistency
        rollout_dynamics();
    }

    void rollout_dynamics() {
        // Start from x0
        // kp.x is updated by dynamics, kp.u is kept from the update
        for(size_t k=0; k<traj.size()-1; ++k) {
            // Use the Model's integrator to maintain consistency
            double current_dt = dt_traj[k];
            traj[k+1].x = Model::integrate(traj[k].x, traj[k].u, current_dt, config.integrator);
        }
    }
};
}
