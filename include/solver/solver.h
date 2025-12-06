#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "core/types.h"
#include "solver/kkt_assembler.h"
#include "solver/riccati_recursive.h"
#include "solver/backend_interface.h"

namespace roboopt {

template<typename Model>
class PDIPMSolver {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;

    using Knot = KnotPoint<double, NX, NU, NC, NP>;

    std::vector<Knot> traj;
    Backend backend;
    double mu; // Barrier Parameter

    PDIPMSolver(int N, Backend b) : backend(b), mu(0.1) {
        traj.resize(N + 1);
        // Initialize slacks/duals strictly positive
        for(auto& kp : traj) kp.initialize_defaults();
    }

    void set_params(const double* p) {
        for(auto& k : traj) {
            for(int i=0; i<NP; ++i) k.p(i) = p[i];
        }
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
        for(auto& kp : traj) Model::compute(kp);

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
        // Note: For GPU support, we would pass 'mu' to the GPU solver logic too.
        // Currently adapting CPU only as requested for the logic rewrite.
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

        // 6. Simulate Dynamics (Shooting / Rollout) to ensure physical consistency
        // In full multiple-shooting, we would treat 'x' as decision variables.
        // Here we do a single shooting rollout to close the dynamic gaps, 
        // effectively projecting x onto the manifold for the next linearization.
        // (Optional, but good for DDP stability)
        rollout_dynamics();
    }

    void rollout_dynamics() {
        // Start from x0
        // kp.x is updated by dynamics, kp.u is kept from the update
        for(size_t k=0; k<traj.size()-1; ++k) {
            double dt = 0.1; 
            // We need to re-evaluate f(x,u) for the rollout.
            // Simplified: Use the Model logic manually or call a helper.
            // Since Model::compute computes derivatives, let's just do a simple integration 
            // using the same physics as CarModel.
            // Ideally, Model should have a `integrate` static method.
            // I will manually integrate here matching CarModel to keep it simple.
            
            // Unpack
            double px = traj[k].x(0);
            double py = traj[k].x(1);
            double th = traj[k].x(2);
            double v  = traj[k].x(3);
            double acc = traj[k].u(0);
            double st  = traj[k].u(1);
            double L = 2.5;

            double nx = px + v * cos(th) * dt;
            double ny = py + v * sin(th) * dt;
            double nth = th + (v/L) * tan(st) * dt;
            double nv = v + acc * dt;

            // Set next state
            traj[k+1].x(0) = nx;
            traj[k+1].x(1) = ny;
            traj[k+1].x(2) = nth;
            traj[k+1].x(3) = nv;
        }
    }
};
}
