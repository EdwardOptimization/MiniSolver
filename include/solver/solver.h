#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "core/types.h"
#include "core/solver_options.h"

#include "solver/kkt_assembler.h"
#include "solver/riccati.h"          
#include "solver/line_search.h"      
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
    std::vector<double> dt_traj;
    Backend backend;
    SolverConfig config;
    
    // Solver State
    double mu; 
    double reg;
    int current_iter = 0;

    PDIPMSolver(int N, Backend b, SolverConfig conf = SolverConfig()) 
        : backend(b), config(conf), mu(conf.mu_init), reg(conf.reg_init) {
        traj.resize(N + 1);
        dt_traj.resize(N, conf.default_dt);
        
        for(auto& kp : traj) kp.initialize_defaults();
    }

    void set_params(const double* p) {
        for(auto& k : traj) {
            for(int i=0; i<NP; ++i) k.p(i) = p[i];
        }
    }

    void set_dt(const std::vector<double>& dts) {
        if(dts.size() != dt_traj.size()) {
            std::cerr << "Warning: DT vector size mismatch.\n";
            return;
        }
        dt_traj = dts;
    }
    
    void set_dt(double dt) {
        std::fill(dt_traj.begin(), dt_traj.end(), dt);
    }

    // Warm Start: Initialize with a previous trajectory guess
    void warm_start(const std::vector<Knot>& init_traj) {
        if (init_traj.size() != traj.size()) return;

        for(size_t k=0; k < traj.size(); ++k) {
            traj[k].x = init_traj[k].x;
            traj[k].u = init_traj[k].u;
            
            // "Shifted Warm Start":
            // Reset s and lam to be "sufficiently positive" to match mu_init.
            // If they are too small, the initial KKT error will be huge due to barrier terms.
            double eps = 1e-2;
            traj[k].s = init_traj[k].s.cwiseMax(eps);
            traj[k].lam = init_traj[k].lam.cwiseMax(eps);
            
            // Centering: ensure s * lam >= mu_init roughly
            for(int i=0; i<NC; ++i) {
                if (traj[k].s(i) * traj[k].lam(i) < config.mu_init) {
                    double shift = std::sqrt(config.mu_init);
                    traj[k].s(i) = std::max(traj[k].s(i), shift);
                    traj[k].lam(i) = std::max(traj[k].lam(i), shift);
                }
            }
        }
        // Reset barrier parameter for the new solve
        mu = config.mu_init;
    }

    // Check convergence: Mu is small enough AND constraints are satisfied
    bool check_convergence(double max_viol) {
        if(mu <= config.mu_min && max_viol <= config.tol_con) return true;
        return false;
    }

    // --- Core: Barrier Update Strategy ---
    // Decides how to update the barrier parameter mu based on current progress.
    void update_barrier(double max_kkt_error, double avg_gap) {
        switch(config.barrier_strategy) {
            case BarrierStrategy::MONOTONE:
                // Robust Fiacco-McCormick Strategy:
                // Only decrease mu if we have solved the current barrier problem accurately enough.
                // This prevents the solver from racing ahead into ill-conditioned territory 
                // before finding a central path point.
                // Criterion: Error < kappa * mu
                if (max_kkt_error < config.barrier_tolerance_factor * mu) {
                    // Reduce mu linearly
                    double next_mu = std::max(config.mu_min, mu * config.mu_linear_decrease_factor);
                    mu = next_mu;
                }
                // Else: Keep solving with current mu to improve centrality/feasibility
                break;

            case BarrierStrategy::ADAPTIVE:
                // Fast Strategy:
                // Track duality gap but ensure monotonic decrease to avoid stagnation.
                // Suitable for well-scaled problems where speed is priority.
                {
                    double target = avg_gap * config.mu_safety_margin; // e.g. 0.1 * gap
                    // Safety: Force at least small decrease (0.9) but allow big drop if gap closes fast
                    double forced = mu * 0.9; 
                    mu = std::max(config.mu_min, std::min(forced, target));
                }
                break;
                
            case BarrierStrategy::MEHROTRA:
                // Heuristic Predictor-Corrector-like:
                // If progress is good (gap is small relative to mu), drop fast.
                // If progress is bad, drop slow.
                // Mehrotra sigma approx = (mu_aff / mu)^3
                {
                    double ratio = avg_gap / mu;
                    if(ratio > 1.0) ratio = 1.0; 
                    double sigma = std::pow(ratio, 3);
                    // Bounded sigma to prevent too aggressive or too slow updates
                    if(sigma < 0.05) sigma = 0.05;
                    if(sigma > 0.8) sigma = 0.8;
                    
                    double next_mu = std::max(config.mu_min, mu * sigma);
                    mu = next_mu;
                }
                break;
        }
    }

    void print_debug_info(double alpha) {
        if (!config.debug_mode) return;

        // Collect detailed stats for debugging convergence issues
        double max_dyn_viol = 0.0; // Dynamics defect (f_resid - x_next)
        double max_con_viol = 0.0; // Hard Constraint Violation
        int worst_con_idx = -1;
        int worst_con_step = -1;
        double max_stat_viol = 0.0; // Stationarity (Dual Inf)
        double min_slack = 1e9;
        
        for(size_t k=0; k < traj.size(); ++k) {
            // 1. Dynamics Violation (Defect)
            // Note: traj[k+1].x is from previous rollout. kp.f_resid is from current compute() (f(x_k, u_k)).
            if (k < traj.size() - 1) {
                double dyn_err = (traj[k].f_resid - traj[k+1].x).template lpNorm<Eigen::Infinity>();
                if (dyn_err > max_dyn_viol) max_dyn_viol = dyn_err;
            }

            // 2. Constraint Violation
            for(int i=0; i<NC; ++i) {
                // Check actual physical violation: max(0, g)
                double g_val = traj[k].g_val(i);
                if (g_val > max_con_viol) {
                    max_con_viol = g_val;
                    worst_con_idx = i;
                    worst_con_step = k;
                }
                if (traj[k].s(i) < min_slack) min_slack = traj[k].s(i);
            }

            // 3. Stationarity
            double stat = std::max(traj[k].q_bar.template lpNorm<Eigen::Infinity>(), traj[k].r_bar.template lpNorm<Eigen::Infinity>());
            if (stat > max_stat_viol) max_stat_viol = stat;
        }

        std::cout << "[DEBUG] Iter " << current_iter << ": "
                  << "Alpha=" << std::fixed << std::setprecision(4) << alpha
                  << " | DynViol=" << std::scientific << max_dyn_viol
                  << " | ConViol=" << max_con_viol 
                  << " (Idx " << worst_con_idx << " @ k=" << worst_con_step << ")"
                  << " | Stat=" << max_stat_viol
                  << " | MinSlack=" << min_slack
                  << std::endl;
    }

    // --- Main Solver Step (SQP / Newton Iteration) ---
    void step() {
        current_iter++;
        
        // 1. Compute Derivatives & Residuals
        // Linearizes the dynamics and constraints around the current trajectory.
        // Also evaluates the Cost function gradients.
        double max_kkt_error = 0.0;
        double total_gap = 0.0;
        int total_con = 0;

        for(size_t k=0; k < traj.size(); ++k) {
            double current_dt = (k < dt_traj.size()) ? dt_traj[k] : 0.0;
            Model::compute(traj[k], config.integrator, current_dt);
            
            // Compute metrics for update strategy
            // KKT Error approx = max(|grad_L|, |constraints|, |complementarity|)
            for(int i=0; i<NC; ++i) {
                double viol = std::abs(traj[k].g_val(i) + traj[k].s(i)); // Primal feasibility
                double comp = std::abs(traj[k].s(i) * traj[k].lam(i) - mu); // Centrality
                if(viol > max_kkt_error) max_kkt_error = viol;
                if(comp > max_kkt_error) max_kkt_error = comp;
            }

            total_gap += traj[k].s.dot(traj[k].lam);
            total_con += NC;
        }

        double avg_gap = (total_con > 0) ? (total_gap / total_con) : 0.0;

        // 2. Update Barrier Parameter using selected strategy
        // This adjusts the "hardness" of the log-barrier terms.
        update_barrier(max_kkt_error, avg_gap);

        // 3. Solve Newton Step (Direction Finding)
        // Computes the search direction (dx, du, ds, dlam) using the Riccati recursion.
        // This solves the KKT system of the barrier-augmented subproblem.
        cpu_serial_solve(traj, mu, reg); 

        // 4. Line Search
        // Determines the step size 'alpha' to ensure we stay within the feasible region (s > 0, lam > 0)
        // and sufficiently decrease the merit function (or residual).
        double alpha = fraction_to_boundary_rule(traj, config.line_search_tau);
        
        print_debug_info(alpha);

        // 5. Update Variables
        // Apply the step: x_new = x + alpha * dx, etc.
        for(size_t k=0; k<traj.size(); ++k) {
            auto& kp = traj[k];
            kp.x += alpha * kp.dx;
            kp.u += alpha * kp.du;
            kp.s += alpha * kp.ds;
            kp.lam += alpha * kp.dlam;
        }

        // 6. Rollout (Shooting)
        // Re-integrates the dynamics with the new controls u_new to ensure physical consistency
        // x_{k+1} = f(x_k, u_k). This closes the "dynamic gap" (defect).
        rollout_dynamics();
    }

    void rollout_dynamics() {
        for(size_t k=0; k<traj.size()-1; ++k) {
            double current_dt = dt_traj[k];
            traj[k+1].x = Model::integrate(traj[k].x, traj[k].u, current_dt, config.integrator);
        }
    }
};
}
