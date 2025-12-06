#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <map>

#include "core/types.h"
#include "core/solver_options.h"

#include "solver/kkt_assembler.h"
#include "solver/riccati.h"          
#include "solver/line_search.h"      
#include "solver/backend_interface.h"

namespace roboopt {

// Simple Profiler
class SolverTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    std::map<std::string, double> times;
    std::string current_scope;
    std::chrono::time_point<Clock> start_time;

    void start(const std::string& name) {
        current_scope = name;
        start_time = Clock::now();
    }

    void stop() {
        auto end_time = Clock::now();
        std::chrono::duration<double, std::milli> ms = end_time - start_time;
        times[current_scope] += ms.count();
    }
    
    void reset() { times.clear(); }
    
    void print() {
        std::cout << "\n--- Solver Profiling (ms) ---\n";
        for(auto const& [name, time] : times) {
            std::cout << std::left << std::setw(20) << name << ": " << time << "\n";
        }
        std::cout << "-----------------------------\n";
    }
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
    std::vector<Knot> traj_candidate; 
    std::vector<double> dt_traj;
    Backend backend;
    SolverConfig config;
    SolverTimer timer; // [NEW] Timer
    
    double mu; 
    double reg;
    double merit_nu; 
    int current_iter = 0;
    
    std::vector<std::pair<double, double>> filter;

    PDIPMSolver(int N, Backend b, SolverConfig conf = SolverConfig()) 
        : backend(b), config(conf), mu(conf.mu_init), reg(conf.reg_init), merit_nu(conf.merit_nu_init) {
        traj.resize(N + 1);
        traj_candidate.resize(N + 1); 
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

    void warm_start(const std::vector<Knot>& init_traj) {
        if (init_traj.size() != traj.size()) return;
        for(size_t k=0; k < traj.size(); ++k) {
            traj[k].x = init_traj[k].x;
            traj[k].u = init_traj[k].u;
            double eps = 1e-2;
            traj[k].s = init_traj[k].s.cwiseMax(eps);
            traj[k].lam = init_traj[k].lam.cwiseMax(eps);
            for(int i=0; i<NC; ++i) {
                if (traj[k].s(i) * traj[k].lam(i) < config.mu_init) {
                    double shift = std::sqrt(config.mu_init);
                    traj[k].s(i) = std::max(traj[k].s(i), shift);
                    traj[k].lam(i) = std::max(traj[k].lam(i), shift);
                }
            }
        }
        mu = config.mu_init;
        filter.clear(); 
    }

    bool check_convergence(double max_viol) {
        if(mu <= config.mu_min && max_viol <= config.tol_con) return true;
        return false;
    }

    void update_barrier(double max_kkt_error, double avg_gap) {
        switch(config.barrier_strategy) {
            case BarrierStrategy::MONOTONE:
                if (max_kkt_error < config.barrier_tolerance_factor * mu) {
                    double next_mu = std::max(config.mu_min, mu * config.mu_linear_decrease_factor);
                    mu = next_mu;
                }
                break;
            case BarrierStrategy::ADAPTIVE: {
                double target = avg_gap * config.mu_safety_margin; 
                double forced = mu * 0.9; 
                mu = std::max(config.mu_min, std::min(forced, target));
                break;
            }
            case BarrierStrategy::MEHROTRA: {
                double ratio = avg_gap / mu;
                if(ratio > 1.0) ratio = 1.0; 
                double sigma = std::pow(ratio, 3);
                if(sigma < 0.05) sigma = 0.05;
                if(sigma > 0.8) sigma = 0.8;
                double next_mu = std::max(config.mu_min, mu * sigma);
                mu = next_mu;
                break;
            }
        }
    }

    void print_iteration_log(double alpha, bool header = false) {
        if (!config.verbose && !config.debug_mode) return;

        if (header) {
            std::cout << std::left 
                      << std::setw(5) << "Iter" 
                      << std::setw(12) << "Cost" 
                      << std::setw(10) << "Log(Mu)" 
                      << std::setw(10) << "Log(Reg)" 
                      << std::setw(10) << "PrimInf" 
                      << std::setw(10) << "DualInf" 
                      << std::setw(10) << "Alpha" 
                      << std::endl;
            std::cout << std::string(70, '-') << std::endl;
            return;
        }

        double total_cost = 0.0;
        double max_prim_inf = 0.0;
        double max_dual_inf = 0.0;

        for(const auto& kp : traj) {
            total_cost += kp.cost;
            for(int i=0; i<NC; ++i) {
                double viol = std::abs(kp.g_val(i) + kp.s(i));
                if(viol > max_prim_inf) max_prim_inf = viol;
            }
            double g_norm = kp.q_bar.template lpNorm<Eigen::Infinity>(); 
            double r_norm = kp.r_bar.template lpNorm<Eigen::Infinity>(); 
            double dual = std::max(g_norm, r_norm);
            if(dual > max_dual_inf) max_dual_inf = dual;
        }

        std::cout << std::left 
                  << std::setw(5) << current_iter 
                  << std::scientific << std::setprecision(3) 
                  << std::setw(12) << total_cost 
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << std::log10(mu) 
                  << std::setw(10) << std::log10(reg) 
                  << std::scientific << std::setprecision(2)
                  << std::setw(10) << max_prim_inf 
                  << std::setw(10) << max_dual_inf 
                  << std::fixed << std::setprecision(3)
                  << std::setw(10) << alpha 
                  << std::endl;
                  
        if (config.debug_mode) {
            double min_slack = 1e9;
            for(const auto& kp : traj) for(int i=0; i<NC; ++i) if(kp.s(i) < min_slack) min_slack = kp.s(i);
            std::cout << "      [DEBUG] MinSlack=" << std::scientific << min_slack << std::endl;
        }
    }

    double compute_merit(const std::vector<Knot>& t, double nu_param) {
        double total_merit = 0.0;
        for(const auto& kp : t) {
            total_merit += kp.cost; 
            for(int i=0; i<NC; ++i) {
                if(kp.s(i) > 1e-20) 
                    total_merit -= mu * std::log(kp.s(i));
                else 
                    total_merit += 1e9; 
            }
            for(int i=0; i<NC; ++i) {
                total_merit += nu_param * std::abs(kp.g_val(i) + kp.s(i));
            }
        }
        return total_merit;
    }
    
    std::pair<double, double> compute_filter_metrics(const std::vector<Knot>& t) {
        double theta = 0.0; 
        double phi = 0.0;   
        for(const auto& kp : t) {
            phi += kp.cost;
            for(int i=0; i<NC; ++i) {
                if(kp.s(i) > 1e-20) phi -= mu * std::log(kp.s(i));
                else phi += 1e9;
            }
            for(int i=0; i<NC; ++i) {
                theta += std::abs(kp.g_val(i) + kp.s(i));
            }
        }
        return {theta, phi};
    }
    
    bool is_acceptable_to_filter(double theta, double phi) {
        double gamma_theta = 1e-5;
        double gamma_phi = 1e-5;
        for(const auto& entry : filter) {
            double theta_j = entry.first;
            double phi_j = entry.second;
            bool sufficient_theta = theta < (1.0 - gamma_theta) * theta_j;
            bool sufficient_phi = phi < phi_j - gamma_phi * theta_j;
            if (!sufficient_theta && !sufficient_phi) return false; 
        }
        return true;
    }
    
    void add_to_filter(double theta, double phi) {
        filter.push_back({theta, phi});
    }

    void update_merit_penalty() {
        double max_dual = 0.0;
        for(const auto& kp : traj) {
            double local_max = kp.lam.template lpNorm<Eigen::Infinity>();
            if(local_max > max_dual) max_dual = local_max;
        }
        double required_nu = max_dual * 1.1 + 1.0; 
        if (required_nu > merit_nu) merit_nu = required_nu;
    }

    void solve_soc(std::vector<Knot>& soc_traj, const std::vector<Knot>& trial_traj) {
        soc_traj = traj; 
        for(size_t k=0; k < traj.size(); ++k) {
            soc_traj[k].g_val = trial_traj[k].g_val + (trial_traj[k].s - traj[k].s);
        }
        bool success = cpu_serial_solve(soc_traj, mu, reg, config.inertia_strategy);
        if (!success) {
            for(auto& kp : soc_traj) {
                kp.dx.setZero(); kp.du.setZero(); kp.ds.setZero(); kp.dlam.setZero();
            }
        }
    }
    
    // UPDATED: Feasibility Restoration with Early Exit
    void feasibility_restoration() {
        if (config.debug_mode) std::cout << "      [DEBUG] Entering Feasibility Restoration Phase.\n";
        double saved_mu = mu;
        double saved_reg = reg;
        mu = 1e-1; 
        reg = 1e-2; 
        
        for(int r_iter=0; r_iter < 10; ++r_iter) { // Allow slightly more iters
            // 1. Compute Derivatives & Update Cost for Min-Norm
            for(size_t k=0; k < traj.size(); ++k) {
                double current_dt = dt_traj[k]; 
                Model::compute(traj[k], config.integrator, current_dt);
                
                // Regularized LS objective: 0.5*||dx||^2 + ...
                traj[k].Q.setIdentity(); 
                traj[k].q.setZero();
                traj[k].R.setIdentity(); 
                traj[k].r.setZero();
            }
            
            // 2. Check if current point is acceptable to Filter
            if (config.line_search_type == LineSearchType::FILTER) {
                auto metrics = compute_filter_metrics(traj);
                if (is_acceptable_to_filter(metrics.first, metrics.second)) {
                    if (config.debug_mode) std::cout << "      [DEBUG] Restoration Successful (Filter Accepted).\n";
                    add_to_filter(metrics.first, metrics.second); // Add restoration point
                    break;
                }
            }

            // 3. Solve & Step
            cpu_serial_solve(traj, mu, reg, config.inertia_strategy);
            double alpha = fraction_to_boundary_rule(traj, 0.95);
            for(size_t k=0; k<traj.size(); ++k) {
                traj[k].x += alpha * traj[k].dx;
                traj[k].u += alpha * traj[k].du;
                traj[k].s += alpha * traj[k].ds;
                traj[k].lam += alpha * traj[k].dlam;
            }
            rollout_dynamics();
        }
        mu = saved_mu;
        reg = saved_reg;
    }

    void solve() {
        current_iter = 0;
        print_iteration_log(0.0, true); 
        timer.reset(); // Reset timer

        for(int iter=0; iter < config.max_iters; ++iter) {
            timer.start("Total Step");
            bool converged = step();
            timer.stop();
            
            if (converged) {
                if (config.verbose) std::cout << ">> Converged in " << iter+1 << " iterations.\n";
                break;
            }
        }
        if (config.verbose) timer.print();
    }

    bool step() {
        current_iter++;
        
        timer.start("Derivatives");
        double max_kkt_error = 0.0;
        double max_prim_inf = 0.0;
        double total_gap = 0.0;
        int total_con = 0;

        for(size_t k=0; k < traj.size(); ++k) {
            double current_dt = (k < dt_traj.size()) ? dt_traj[k] : 0.0;
            Model::compute(traj[k], config.integrator, current_dt);
            
            for(int i=0; i<NC; ++i) {
                double viol = std::abs(traj[k].g_val(i) + traj[k].s(i)); 
                double comp = std::abs(traj[k].s(i) * traj[k].lam(i) - mu); 
                if(viol > max_kkt_error) max_kkt_error = viol;
                if(comp > max_kkt_error) max_kkt_error = comp;
                if(viol > max_prim_inf) max_prim_inf = viol;
            }
            total_gap += traj[k].s.dot(traj[k].lam);
            total_con += NC;
        }
        timer.stop();

        double avg_gap = (total_con > 0) ? (total_gap / total_con) : 0.0;
        update_barrier(max_kkt_error, avg_gap);
        if (config.line_search_type == LineSearchType::MERIT) update_merit_penalty();

        timer.start("Linear Solve");
        bool solve_success = false;
        for(int try_count=0; try_count < 5; ++try_count) {
            solve_success = cpu_serial_solve(traj, mu, reg, config.inertia_strategy);
            if (solve_success) break;
            
            if (reg < config.reg_min) reg = config.reg_min;
            reg *= config.reg_scale_up;
            if (reg > config.reg_max) reg = config.reg_max;
        }
        if (solve_success && reg > config.reg_min) {
             reg = std::max(config.reg_min, reg / config.reg_scale_down);
        }
        timer.stop();

        timer.start("Line Search");
        double alpha = fraction_to_boundary_rule(traj, config.line_search_tau);
        
        double merit_0 = 0.0;
        double theta_0 = 0.0, phi_0 = 0.0;
        
        if (config.line_search_type == LineSearchType::MERIT) {
            merit_0 = compute_merit(traj, merit_nu);
        } else {
            auto metrics = compute_filter_metrics(traj);
            theta_0 = metrics.first;
            phi_0 = metrics.second;
        }

        bool accepted = false;
        int ls_iter = 0;
        bool soc_attempted = false;
        
        while (ls_iter < config.line_search_max_iters) {
            traj_candidate = traj; 
            for(size_t k=0; k<traj.size(); ++k) {
                traj_candidate[k].x += alpha * traj[k].dx;
                traj_candidate[k].u += alpha * traj[k].du;
                traj_candidate[k].s += alpha * traj[k].ds;
                traj_candidate[k].lam += alpha * traj[k].dlam;
                double current_dt = (k < dt_traj.size()) ? dt_traj[k] : 0.0;
                Model::compute(traj_candidate[k], config.integrator, current_dt);
            }
            
            if (config.line_search_type == LineSearchType::MERIT) {
                double merit_alpha = compute_merit(traj_candidate, merit_nu);
                if (merit_alpha < merit_0) accepted = true;
            } 
            else if (config.line_search_type == LineSearchType::FILTER) {
                auto m_alpha = compute_filter_metrics(traj_candidate);
                if (is_acceptable_to_filter(m_alpha.first, m_alpha.second)) accepted = true;
            }
            else {
                accepted = true; 
            }

            if (!accepted && config.enable_soc && !soc_attempted && ls_iter == 0 && alpha > 0.5) {
                if (config.debug_mode) std::cout << "      [DEBUG] Step rejected. Attempting SOC.\n";
                std::vector<Knot> soc_data; 
                solve_soc(soc_data, traj_candidate); 
                for(size_t k=0; k<traj.size(); ++k) {
                    traj[k].dx += soc_data[k].dx;
                    traj[k].du += soc_data[k].du;
                    traj[k].ds += soc_data[k].ds;
                    traj[k].dlam += soc_data[k].dlam;
                }
                soc_attempted = true;
                continue; 
            }

            if (accepted) break;
            alpha *= 0.5; 
            ls_iter++;
        }
        
        if (accepted) {
            traj = traj_candidate;
            if (config.line_search_type == LineSearchType::FILTER) {
                add_to_filter(theta_0, phi_0);
            }
        } else {
             bool recovered = false;
             if (config.enable_slack_reset && alpha < config.slack_reset_trigger) {
                 if (config.debug_mode) std::cout << "      [DEBUG] Triggering Slack Reset.\n";
                 for(auto& kp : traj) {
                     for(int i=0; i<NC; ++i) {
                         double min_s = std::abs(kp.g_val(i)) + std::sqrt(mu);
                         if (kp.s(i) < min_s) kp.s(i) = min_s;
                         kp.lam(i) = mu / kp.s(i);
                     }
                 }
                 recovered = true;
                 cpu_serial_solve(traj, mu, reg, config.inertia_strategy);
             }
             
             if (!recovered && config.enable_feasibility_restoration) {
                 feasibility_restoration();
             }
        }
        timer.stop();

        print_iteration_log(alpha);
        
        timer.start("Rollout");
        rollout_dynamics();
        timer.stop();
        
        return check_convergence(max_prim_inf);
    }

    void rollout_dynamics() {
        for(size_t k=0; k<traj.size()-1; ++k) {
            double current_dt = dt_traj[k];
            traj[k+1].x = Model::integrate(traj[k].x, traj[k].u, current_dt, config.integrator);
        }
    }
};
}
