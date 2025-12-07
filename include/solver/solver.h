#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <map>
#include <memory> 
#include <string>
#include <unordered_map> 

#include "core/types.h"
#include "core/solver_options.h"
#include "core/trajectory.h" 

#include "algorithms/linear_solver.h" 
#include "algorithms/riccati_solver.h" 
#include "algorithms/line_search.h" 

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
    using TrajectoryType = Trajectory<Knot, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    TrajectoryType trajectory;

    // Components
    std::unique_ptr<LinearSolver<TrajArray>> linear_solver;
    std::unique_ptr<LineSearchStrategy<Model, MAX_N>> line_search;

    int N; 
    std::array<double, MAX_N> dt_traj;
    
    Backend backend;
    SolverConfig config;
    SolverTimer timer; 
    
    double mu; 
    double reg;
    double merit_nu; 
    int current_iter = 0;
    
    std::vector<std::pair<double, double>> filter;
    
    // Lookup Maps
    std::unordered_map<std::string, int> state_map;
    std::unordered_map<std::string, int> control_map;
    std::unordered_map<std::string, int> param_map;

    PDIPMSolver(int initial_N, Backend b, SolverConfig conf = SolverConfig()) 
        : trajectory(initial_N), N(initial_N), backend(b), config(conf), mu(conf.mu_init), reg(conf.reg_init) {
        
        if (N > MAX_N) {
            std::cerr << "Error: N (" << N << ") > MAX_N (" << MAX_N << "). Clamping.\n";
            N = MAX_N;
        }

        dt_traj.fill(conf.default_dt);
        
        // Initialize Components
        linear_solver = std::make_unique<RiccatiSolver<TrajArray>>();
        
        if (config.line_search_type == LineSearchType::MERIT) {
            line_search = std::make_unique<MeritLineSearch<Model, MAX_N>>();
        } else {
            line_search = std::make_unique<FilterLineSearch<Model, MAX_N>>();
        }
        
        // Initialize Maps
        for (int i = 0; i < NX; ++i) state_map[Model::state_names[i]] = i;
        for (int i = 0; i < NU; ++i) control_map[Model::control_names[i]] = i;
        for (int i = 0; i < NP; ++i) param_map[Model::param_names[i]] = i;
    }
    
    int get_state_idx(const std::string& name) const {
        auto it = state_map.find(name);
        if (it != state_map.end()) return it->second;
        return -1;
    }

    int get_control_idx(const std::string& name) const {
        auto it = control_map.find(name);
        if (it != control_map.end()) return it->second;
        return -1;
    }

    int get_param_idx(const std::string& name) const {
        auto it = param_map.find(name);
        if (it != param_map.end()) return it->second;
        return -1;
    }

    void resize_horizon(int new_n) {
        if (new_n > MAX_N) {
            std::cerr << "Error: new_n > MAX_N\n";
            return;
        }
        N = new_n;
        trajectory.resize(N);
    }

    // --- High-Level API ---

    // 1. Initial State
    void set_initial_state(const std::vector<double>& x0) {
        if (x0.size() != NX) return;
        auto& kp = trajectory[0];
        for(int i=0; i<NX; ++i) kp.x(i) = x0[i];
    }
    
    void set_initial_state(const std::string& name, double value) {
        int idx = get_state_idx(name);
        if (idx != -1) trajectory[0].x(idx) = value;
        else std::cerr << "Warning: Unknown state " << name << "\n";
    }

    // 2. Parameters
    void set_parameter(int stage, int idx, double value) {
        if (stage > N || stage < 0) return;
        if (idx >= NP || idx < 0) return;
        trajectory[stage].p(idx) = value;
    }

    void set_parameter(int stage, const std::string& name, double value) {
        int idx = get_param_idx(name);
        if (idx != -1) set_parameter(stage, idx, value);
        else std::cerr << "Warning: Unknown param " << name << "\n";
    }
    
    void set_global_parameter(int idx, double value) {
        if (idx >= NP || idx < 0) return;
        for(int k=0; k <= N; ++k) {
            trajectory[k].p(idx) = value;
        }
    }

    void set_global_parameter(const std::string& name, double value) {
        int idx = get_param_idx(name);
        if (idx != -1) set_global_parameter(idx, value);
        else std::cerr << "Warning: Unknown param " << name << "\n";
    }

    double get_parameter(int stage, int idx) const {
        if (stage > N || stage < 0 || idx >= NP || idx < 0) return 0.0;
        return trajectory[stage].p(idx);
    }
    
    std::vector<double> get_parameters(int stage) const {
        if (stage > N || stage < 0) return {};
        const auto& kp = trajectory[stage];
        std::vector<double> res(NP);
        for(int i=0; i<NP; ++i) res[i] = kp.p(i);
        return res;
    }

    // 3. State Access
    void set_state_guess(int stage, int idx, double value) {
        if (stage > N || stage < 0 || idx >= NX || idx < 0) return;
        trajectory[stage].x(idx) = value;
    }

    void set_state_guess(int stage, const std::string& name, double value) {
        int idx = get_state_idx(name);
        if (idx != -1) set_state_guess(stage, idx, value);
    }
    
    void set_state_guess_traj(const std::string& name, const std::vector<double>& values) {
        int idx = get_state_idx(name);
        if (idx == -1) return;
        int count = std::min((int)values.size(), N + 1);
        for(int k=0; k<count; ++k) {
            trajectory[k].x(idx) = values[k];
        }
    }

    std::vector<double> get_state_traj(const std::string& name) const {
        int idx = get_state_idx(name);
        if (idx == -1) return {};
        std::vector<double> res;
        res.reserve(N + 1);
        for(int k=0; k<=N; ++k) res.push_back(trajectory[k].x(idx));
        return res;
    }
    
    std::vector<double> get_state(int stage) const {
        if (stage > N || stage < 0) return {};
        const auto& kp = trajectory[stage];
        std::vector<double> res(NX);
        for(int i=0; i<NX; ++i) res[i] = kp.x(i);
        return res;
    }

    double get_state(int stage, int idx) const {
        if (stage > N || stage < 0 || idx >= NX || idx < 0) return 0.0;
        return trajectory[stage].x(idx);
    }

    // 4. Control Access
    void set_control_guess(int stage, int idx, double value) {
        if (stage >= N || stage < 0 || idx >= NU || idx < 0) return;
        trajectory[stage].u(idx) = value;
    }

    void set_control_guess(int stage, const std::string& name, double value) {
        int idx = get_control_idx(name);
        if (idx != -1) set_control_guess(stage, idx, value);
    }
    
    void set_control_guess_traj(const std::string& name, const std::vector<double>& values) {
        int idx = get_control_idx(name);
        if (idx == -1) return;
        int count = std::min((int)values.size(), N);
        for(int k=0; k<count; ++k) {
            trajectory[k].u(idx) = values[k];
        }
    }

    std::vector<double> get_control_traj(const std::string& name) const {
        int idx = get_control_idx(name);
        if (idx == -1) return {};
        std::vector<double> res;
        res.reserve(N);
        for(int k=0; k<N; ++k) res.push_back(trajectory[k].u(idx));
        return res;
    }
    
    std::vector<double> get_control(int stage) const {
        if (stage >= N || stage < 0) return {};
        const auto& kp = trajectory[stage];
        std::vector<double> res(NU);
        for(int i=0; i<NU; ++i) res[i] = kp.u(i);
        return res;
    }

    double get_control(int stage, int idx) const {
        if (stage >= N || stage < 0 || idx >= NU || idx < 0) return 0.0;
        return trajectory[stage].u(idx);
    }

    // 5. Cost Access
    double get_stage_cost(int stage) const {
        if (stage > N || stage < 0) return 0.0;
        return trajectory[stage].cost;
    }
    
    // Helper to get constraint value
    double get_constraint_val(int stage, int idx) const {
        if (stage > N || idx >= NC) return 0.0;
        return trajectory[stage].g_val(idx);
    }
    
    // Direct access (discouraged but available)
    typename TrajectoryType::TrajArray& get_raw_traj() { return *trajectory.active_ptr; }

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

    void warm_start(const typename TrajectoryType::TrajArray& init_traj) {
        auto& current_traj = trajectory.active();
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
        line_search->reset();
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
        if (config.print_level < PrintLevel::ITER) return;

        if (header) {
            std::cout << std::left 
                      << std::setw(5) << "Iter" 
                      << std::setw(12) << "Cost" 
                      << std::setw(10) << "Log(Mu)" 
                      << std::setw(10) << "Log(Reg)" 
                      << std::setw(10) << "PrimInf" 
                      << std::setw(10) << "DualInf" 
                      << std::setw(10) << "Alpha";
            
            if (config.print_level >= PrintLevel::DEBUG) {
                std::cout << std::setw(12) << "MinSlack";
            }
            std::cout << std::endl;
            std::cout << std::string(80, '-') << std::endl;
            return;
        }

        double total_cost = 0.0;
        double max_prim_inf = 0.0;
        double max_dual_inf = 0.0;
        double min_slack = 1e9;
        auto& traj = trajectory.active();

        for(int k=0; k<=N; ++k) {
            const auto& kp = traj[k];
            total_cost += kp.cost;
            for(int i=0; i<NC; ++i) {
                double viol = std::abs(kp.g_val(i) + kp.s(i));
                if(viol > max_prim_inf) max_prim_inf = viol;
                if(kp.s(i) < min_slack) min_slack = kp.s(i);
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
                  << std::setw(10) << alpha;

        if (config.print_level >= PrintLevel::DEBUG) {
            std::cout << std::scientific << std::setprecision(2) << std::setw(12) << min_slack;
        }
        std::cout << std::endl;
    }

    double compute_merit(const typename TrajectoryType::TrajArray& t, double nu_param) {
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
    
    std::pair<double, double> compute_filter_metrics(const typename TrajectoryType::TrajArray& t) {
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
        auto& traj = trajectory.active();
        for(int k=0; k<=N; ++k) {
            double local_max = traj[k].lam.template lpNorm<Eigen::Infinity>();
            if(local_max > max_dual) max_dual = local_max;
        }
        double required_nu = max_dual * 1.1 + 1.0; 
        if (required_nu > merit_nu) merit_nu = required_nu;
    }

    bool has_nans(const typename TrajectoryType::TrajArray& t) {
        // Checking all variables is expensive (O(N * (NX+NU+NC))).
        // Instead, we only check the updates (dx, du, ds, dlam) and cost/constraints 
        // which are the sources of NaNs.
        for(int k=0; k<=N; ++k) {
            const auto& kp = t[k];
            // Check Search Directions (Most likely place for NaN from Linear Solve)
            if(!kp.dx.allFinite()) return true;
            if(!kp.du.allFinite()) return true;
            if(!kp.ds.allFinite()) return true;
            if(!kp.dlam.allFinite()) return true;
            
            // Check key scalar values
            if(!std::isfinite(kp.cost)) return true;
        }
        return false;
    }

    void solve_soc(typename TrajectoryType::TrajArray& soc_traj, const typename TrajectoryType::TrajArray& trial_traj) {
        auto& current_traj = trajectory.active();
        for(int k=0; k<=N; ++k) soc_traj[k] = current_traj[k];
        for(int k=0; k<=N; ++k) {
            soc_traj[k].g_val = trial_traj[k].g_val + (trial_traj[k].s - current_traj[k].s);
        }
        bool success = linear_solver->solve(soc_traj, N, mu, reg, config.inertia_strategy, config);
        if (!success) {
            for(int k=0; k<=N; ++k) {
                soc_traj[k].dx.setZero(); soc_traj[k].du.setZero(); 
                soc_traj[k].ds.setZero(); soc_traj[k].dlam.setZero();
            }
        }
    }
    
    bool feasibility_restoration() {
        if (config.print_level >= PrintLevel::DEBUG) 
            std::cout << "      [DEBUG] Entering Feasibility Restoration Phase.\n";
        double saved_mu = mu;
        double saved_reg = reg;
        mu = config.restoration_mu; 
        reg = config.restoration_reg; 
        
        auto& traj = trajectory.active();
        bool success = false;
        
        for(int r_iter=0; r_iter < config.max_restoration_iters; ++r_iter) { 
            for(int k=0; k<=N; ++k) {
                double current_dt = dt_traj[k]; 
                Model::compute_dynamics(traj[k], config.integrator, current_dt);
                Model::compute_constraints(traj[k]);
                
                traj[k].Q.setIdentity(); 
                traj[k].q.setZero();
                
                if (k < N) {
                    traj[k].R.setIdentity(); 
                    traj[k].r.setZero();
                } else {
                    traj[k].R.setZero();
                    traj[k].r.setZero();
                }
            }
            
            if (config.line_search_type == LineSearchType::FILTER) {
                auto metrics = compute_filter_metrics(traj);
                if (is_acceptable_to_filter(metrics.first, metrics.second)) {
                    if (config.print_level >= PrintLevel::DEBUG) 
                        std::cout << "      [DEBUG] Restoration Successful (Filter Accepted).\n";
                    add_to_filter(metrics.first, metrics.second);
                    success = true;
                    break;
                }
            }

            // Restoration linear solve
            if(!linear_solver->solve(traj, N, mu, reg, config.inertia_strategy, config)) {
                // If linear solve fails even in restoration, we are in trouble
                break;
            }
            
            double alpha = fraction_to_boundary_rule(traj, N, config.restoration_alpha);
            
            // Simple line search in restoration (could be improved)
            if(alpha < 1e-4) {
                 // Stuck
                 break; 
            }

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
        return success;
    }

    SolverStatus solve() {
        current_iter = 0;
        print_iteration_log(0.0, true); 
        timer.reset(); 

        for(int iter=0; iter < config.max_iters; ++iter) {
            timer.start("Total Step");
            SolverStatus status = step();
            timer.stop();
            
            if (status == SolverStatus::SOLVED) {
                if (config.print_level >= PrintLevel::INFO) 
                    std::cout << ">> Converged in " << iter+1 << " iterations.\n";
                if (config.print_level >= PrintLevel::INFO) timer.print();
                return SolverStatus::SOLVED;
            } else if (status != SolverStatus::UNSOLVED) {
                // Error or Infeasible
                if (config.print_level >= PrintLevel::INFO)
                     std::cout << ">> Solver terminated with status: " << status_to_string(status) << "\n";
                if (config.print_level >= PrintLevel::INFO) timer.print();
                return status;
            }
        }
        if (config.print_level >= PrintLevel::INFO) std::cout << ">> Max iterations reached.\n";
        if (config.print_level >= PrintLevel::INFO) timer.print();
        
        // Final Feasibility Check
        auto& traj = trajectory.active();
        double max_viol = 0.0;
        for(int k=0; k<=N; ++k) {
             for(int i=0; i<NC; ++i) {
                 double v = std::abs(traj[k].g_val(i) + traj[k].s(i));
                 if(v > max_viol) max_viol = v;
             }
        }
        
        if (max_viol <= config.tol_con) return SolverStatus::FEASIBLE;
        
        return SolverStatus::MAX_ITER;
    }

    SolverStatus step() {
        current_iter++;
        auto& traj = trajectory.active();
        
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

        // Check for Numerical Instability (NaN/Inf)
        if (has_nans(traj)) {
             if (config.print_level >= PrintLevel::INFO) 
                std::cout << ">> Numerical Error: NaNs detected in derivatives or state.\n";
             return SolverStatus::NUMERICAL_ERROR;
        }

        // Check Convergence
        if (check_convergence(max_prim_inf)) return SolverStatus::SOLVED;

        double avg_gap = (total_con > 0) ? (total_gap / total_con) : 0.0;
        update_barrier(max_kkt_error, avg_gap);
        if (config.line_search_type == LineSearchType::MERIT) update_merit_penalty();

        timer.start("Linear Solve");
        bool solve_success = false;
        for(int try_count=0; try_count < config.inertia_max_retries; ++try_count) {
            solve_success = linear_solver->solve(traj, N, mu, reg, config.inertia_strategy, config);
            if (solve_success) break;
            if (reg < config.reg_min) reg = config.reg_min;
            reg *= config.reg_scale_up;
            if (reg > config.reg_max) reg = config.reg_max;
        }
        if (solve_success && reg > config.reg_min) {
             reg = std::max(config.reg_min, reg / config.reg_scale_down);
        }
        timer.stop();

        if (!solve_success) return SolverStatus::NUMERICAL_ERROR;

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
        
        // Prepare Candidate
        trajectory.prepare_candidate();
        auto& candidate = trajectory.candidate();
        
        while (ls_iter < config.line_search_max_iters) {
            for(int k=0; k<=N; ++k) {
                candidate[k].x = traj[k].x + alpha * traj[k].dx;
                candidate[k].u = traj[k].u + alpha * traj[k].du;
                candidate[k].s = traj[k].s + alpha * traj[k].ds;
                candidate[k].lam = traj[k].lam + alpha * traj[k].dlam;
                // Params copied by prepare_candidate
                
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
                if (config.print_level >= PrintLevel::DEBUG) 
                    std::cout << "      [DEBUG] Step rejected. Attempting SOC.\n";
                
                auto soc_data = std::make_unique<typename TrajectoryType::TrajArray>();
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
            trajectory.swap();
            if (config.line_search_type == LineSearchType::FILTER) {
                add_to_filter(theta_0, phi_0);
            }
        } else {
             // Step 1: Slack Reset
             bool recovered = false;
             if (config.enable_slack_reset && alpha < config.slack_reset_trigger) {
                 if (config.print_level >= PrintLevel::DEBUG) 
                    std::cout << "      [DEBUG] Triggering Slack Reset.\n";
                 for(int k=0; k<=N; ++k) {
                     auto& kp = traj[k];
                     for(int i=0; i<NC; ++i) {
                         double min_s = std::abs(kp.g_val(i)) + std::sqrt(mu);
                         if (kp.s(i) < min_s) kp.s(i) = min_s;
                         kp.lam(i) = mu / kp.s(i);
                     }
                 }
                 // Try one linear solve step to see if we can move
                 recovered = linear_solver->solve(traj, N, mu, reg, config.inertia_strategy, config);
             }
             
             // Step 2: Feasibility Restoration
             if (!recovered && config.enable_feasibility_restoration) {
                 recovered = feasibility_restoration();
             }

             if (!recovered) {
                 // Both Slack Reset and Feasibility Restoration failed (or were disabled)
                 timer.stop();
                 print_iteration_log(alpha); // Log the fail state
                 return SolverStatus::PRIMAL_INFEASIBLE;
             }
        }
        timer.stop();

        print_iteration_log(alpha);
        
        timer.start("Rollout");
        rollout_dynamics();
        timer.stop();
        
        // Final check after step update
        if(check_convergence(max_prim_inf)) return SolverStatus::SOLVED;

        return SolverStatus::UNSOLVED;
    }

    void rollout_dynamics() {
        auto& traj = trajectory.active();
        for(int k=0; k<N; ++k) {
            double current_dt = dt_traj[k];
            traj[k+1].x = Model::integrate(traj[k].x, traj[k].u, traj[k].p, current_dt, config.integrator);
        }
    }
};
}
