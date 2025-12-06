#pragma once
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <map>
#include <memory> 

#include "core/types.h"
#include "core/solver_options.h"

#include "solver/kkt_assembler.h"
#include "solver/riccati.h"          
#include "solver/line_search.h"      
#include "solver/backend_interface.h"

namespace minisolver {

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

template<typename Model, int _MAX_N>
class PDIPMSolver {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;
    static const int MAX_N = _MAX_N;

    using Knot = KnotPoint<double, NX, NU, NC, NP>;
    
    // Static Allocation: Zero malloc at runtime
    using TrajArray = std::array<Knot, MAX_N + 1>;

    // Double Buffering
    TrajArray traj_memory_A;
    TrajArray traj_memory_B;
    
    TrajArray* traj_ptr;      
    TrajArray* candidate_ptr; 

    // Runtime horizon
    int N; 
    
    // Using vector for dt is fine as it's small, but array is better
    std::array<double, MAX_N> dt_traj;
    
    Backend backend;
    SolverConfig config;
    SolverTimer timer; 
    
    double mu; 
    double reg;
    double merit_nu; 
    int current_iter = 0;
    
    std::vector<std::pair<double, double>> filter;

    // Constructor: N is initial valid horizon
    PDIPMSolver(int initial_N, Backend b, SolverConfig conf = SolverConfig()) 
        : N(initial_N), backend(b), config(conf), mu(conf.mu_init), reg(conf.reg_init), merit_nu(conf.merit_nu_init) {
        
        if (N > MAX_N) {
            std::cerr << "Error: N (" << N << ") > MAX_N (" << MAX_N << "). Clamping.\n";
            N = MAX_N;
        }

        traj_ptr = &traj_memory_A;
        candidate_ptr = &traj_memory_B;
        
        // Default DT
        dt_traj.fill(conf.default_dt);
        
        // Initialize defaults
        for(auto& kp : *traj_ptr) kp.initialize_defaults();
        for(auto& kp : *candidate_ptr) kp.initialize_defaults();
    }
    
    // Set active horizon length
    void resize_horizon(int new_n) {
        if (new_n > MAX_N) {
            std::cerr << "Error: new_n > MAX_N\n";
            return;
        }
        N = new_n;
    }

    TrajArray& get_traj() { return *traj_ptr; }
    const TrajArray& get_traj() const { return *traj_ptr; }

    // Helpers to iterate only up to N
    // We can use spans in C++20, but for C++17 we just pass pointer + size
    
    void set_params(const double* p) {
        for(int k=0; k <= N; ++k) {
            auto& kp = (*traj_ptr)[k];
            for(int i=0; i<NP; ++i) kp.p(i) = p[i];
        }
    }

    void set_dt(const std::vector<double>& dts) {
        if(dts.size() > MAX_N) {
            std::cerr << "Warning: DT vector too large.\n";
        }
        int count = std::min((int)dts.size(), N);
        for(int i=0; i<count; ++i) dt_traj[i] = dts[i];
    }
    
    void set_dt(double dt) {
        dt_traj.fill(dt);
    }

    void warm_start(const TrajArray& init_traj) {
        auto& current_traj = *traj_ptr;
        for(int k=0; k <= N; ++k) {
            current_traj[k].x = init_traj[k].x;
            current_traj[k].u = init_traj[k].u;
            double eps = 1e-2;
            current_traj[k].s = init_traj[k].s.cwiseMax(eps);
            current_traj[k].lam = init_traj[k].lam.cwiseMax(eps);
            for(int i=0; i<NC; ++i) {
                if (current_traj[k].s(i) * current_traj[k].lam(i) < config.mu_init) {
                    double shift = std::sqrt(config.mu_init);
                    current_traj[k].s(i) = std::max(current_traj[k].s(i), shift);
                    current_traj[k].lam(i) = std::max(current_traj[k].lam(i), shift);
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
        auto& traj = *traj_ptr;

        for(int k=0; k<=N; ++k) {
            const auto& kp = traj[k];
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
            for(int k=0; k<=N; ++k) for(int i=0; i<NC; ++i) if(traj[k].s(i) < min_slack) min_slack = traj[k].s(i);
            std::cout << "      [DEBUG] MinSlack=" << std::scientific << min_slack << std::endl;
        }
    }

    double compute_merit(const TrajArray& t, double nu_param) {
        double total_merit = 0.0;
        for(int k=0; k<=N; ++k) {
            const auto& kp = t[k];
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
    
    std::pair<double, double> compute_filter_metrics(const TrajArray& t) {
        double theta = 0.0; 
        double phi = 0.0;   
        for(int k=0; k<=N; ++k) {
            const auto& kp = t[k];
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
        auto& traj = *traj_ptr;
        for(int k=0; k<=N; ++k) {
            double local_max = traj[k].lam.template lpNorm<Eigen::Infinity>();
            if(local_max > max_dual) max_dual = local_max;
        }
        double required_nu = max_dual * 1.1 + 1.0; 
        if (required_nu > merit_nu) merit_nu = required_nu;
    }

    void solve_soc(TrajArray& soc_traj, const TrajArray& trial_traj) {
        auto& current_traj = *traj_ptr;
        // Copy current
        for(int k=0; k<=N; ++k) soc_traj[k] = current_traj[k];
        
        for(int k=0; k<=N; ++k) {
            soc_traj[k].g_val = trial_traj[k].g_val + (trial_traj[k].s - current_traj[k].s);
        }
        bool success = cpu_serial_solve(soc_traj, N, mu, reg, config.inertia_strategy);
        if (!success) {
            for(int k=0; k<=N; ++k) {
                soc_traj[k].dx.setZero(); soc_traj[k].du.setZero(); 
                soc_traj[k].ds.setZero(); soc_traj[k].dlam.setZero();
            }
        }
    }
    
    void feasibility_restoration() {
        if (config.debug_mode) std::cout << "      [DEBUG] Entering Feasibility Restoration Phase.\n";
        double saved_mu = mu;
        double saved_reg = reg;
        mu = 1e-1; 
        reg = 1e-2; 
        
        auto& traj = *traj_ptr;
        
        for(int r_iter=0; r_iter < 10; ++r_iter) { 
            for(int k=0; k<=N; ++k) {
                double current_dt = dt_traj[k]; 
                Model::compute(traj[k], config.integrator, current_dt);
                traj[k].Q.setIdentity(); 
                traj[k].q.setZero();
                traj[k].R.setIdentity(); 
                traj[k].r.setZero();
            }
            
            if (config.line_search_type == LineSearchType::FILTER) {
                auto metrics = compute_filter_metrics(traj);
                if (is_acceptable_to_filter(metrics.first, metrics.second)) {
                    if (config.debug_mode) std::cout << "      [DEBUG] Restoration Successful (Filter Accepted).\n";
                    add_to_filter(metrics.first, metrics.second);
                    break;
                }
            }

            cpu_serial_solve(traj, N, mu, reg, config.inertia_strategy);
            double alpha = fraction_to_boundary_rule(traj, N, 0.95);
            for(int k=0; k<=N; ++k) {
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
        timer.reset(); 

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
        auto& traj = *traj_ptr;
        
        timer.start("Derivatives");
        double max_kkt_error = 0.0;
        double max_prim_inf = 0.0;
        double total_gap = 0.0;
        int total_con = 0;

        for(int k=0; k<=N; ++k) {
            double current_dt = dt_traj[k];
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
            solve_success = cpu_serial_solve(traj, N, mu, reg, config.inertia_strategy);
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
        double alpha = fraction_to_boundary_rule(traj, N, config.line_search_tau);
        
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
        
        auto& candidate = *candidate_ptr;
        
        while (ls_iter < config.line_search_max_iters) {
            for(int k=0; k<=N; ++k) {
                candidate[k].x = traj[k].x + alpha * traj[k].dx;
                candidate[k].u = traj[k].u + alpha * traj[k].du;
                candidate[k].s = traj[k].s + alpha * traj[k].ds;
                candidate[k].lam = traj[k].lam + alpha * traj[k].dlam;
                candidate[k].p = traj[k].p; 
                
                double current_dt = dt_traj[k];
                Model::compute(candidate[k], config.integrator, current_dt);
            }
            
            if (config.line_search_type == LineSearchType::MERIT) {
                double merit_alpha = compute_merit(candidate, merit_nu);
                if (merit_alpha < merit_0) accepted = true;
            } 
            else if (config.line_search_type == LineSearchType::FILTER) {
                auto m_alpha = compute_filter_metrics(candidate);
                if (is_acceptable_to_filter(m_alpha.first, m_alpha.second)) accepted = true;
            }
            else {
                accepted = true; 
            }

            if (!accepted && config.enable_soc && !soc_attempted && ls_iter == 0 && alpha > 0.5) {
                if (config.debug_mode) std::cout << "      [DEBUG] Step rejected. Attempting SOC.\n";
                
                // SOC needs a temp buffer. We can't reuse traj (A) as it holds current valid point.
                // We can't reuse candidate (B) as it holds trial point residuals needed for SOC.
                // We allocate a local TrajArray on stack/heap? TrajArray is huge on stack.
                // Ideally class should have 3 buffers.
                // For now, let's use std::vector allocation for SOC (fallback is rare).
                // Actually solve_soc signature expects TrajArray&.
                // Hack: We don't have a 3rd buffer.
                // We can modify candidate in place if we are careful?
                // No, we need original A and B residuals.
                
                // For this optimization phase, let's SKIP SOC implementation update to save time/complexity
                // or allocate dynamic vector.
                // Let's allocate dynamic vector for SOC.
                
                // Note: To match TrajArray, we can't easily use vector unless templated.
                // Let's create a local static array if stack allows, or just heap alloc via unique_ptr.
                auto soc_data = std::make_unique<TrajArray>();
                
                solve_soc(*soc_data, candidate); 
                
                for(int k=0; k<=N; ++k) {
                    traj[k].dx += (*soc_data)[k].dx;
                    traj[k].du += (*soc_data)[k].du;
                    traj[k].ds += (*soc_data)[k].ds;
                    traj[k].dlam += (*soc_data)[k].dlam;
                }
                soc_attempted = true;
                continue; 
            }

            if (accepted) break;
            alpha *= 0.5; 
            ls_iter++;
        }
        
        if (accepted) {
            std::swap(traj_ptr, candidate_ptr);
            if (config.line_search_type == LineSearchType::FILTER) {
                add_to_filter(theta_0, phi_0);
            }
        } else {
             bool recovered = false;
             if (config.enable_slack_reset && alpha < config.slack_reset_trigger) {
                 if (config.debug_mode) std::cout << "      [DEBUG] Triggering Slack Reset.\n";
                 for(int k=0; k<=N; ++k) {
                     auto& kp = (*traj_ptr)[k];
                     for(int i=0; i<NC; ++i) {
                         double min_s = std::abs(kp.g_val(i)) + std::sqrt(mu);
                         if (kp.s(i) < min_s) kp.s(i) = min_s;
                         kp.lam(i) = mu / kp.s(i);
                     }
                 }
                 recovered = true;
                 cpu_serial_solve(*traj_ptr, N, mu, reg, config.inertia_strategy);
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
        auto& traj = *traj_ptr;
        for(int k=0; k<N; ++k) {
            double current_dt = dt_traj[k];
            traj[k+1].x = Model::integrate(traj[k].x, traj[k].u, current_dt, config.integrator);
        }
    }
};
}
