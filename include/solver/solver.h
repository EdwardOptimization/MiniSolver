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
#include "core/logger.h"

#include "algorithms/linear_solver.h" 
#include "algorithms/riccati_solver.h" 
#include "algorithms/line_search.h" 

#include "solver/backend_interface.h"

namespace minisolver {

class SolverTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    std::map<std::string, double> times;
    std::vector<std::pair<std::string, std::chrono::time_point<Clock>>> stack;

    void start(const std::string& name) {
        stack.push_back({name, Clock::now()});
    }

    void stop() {
        if (stack.empty()) return;
        auto end_time = Clock::now();
        auto& entry = stack.back();
        std::chrono::duration<double, std::milli> ms = end_time - entry.second;
        times[entry.first] += ms.count();
        stack.pop_back();
    }
    
    void reset() { 
        times.clear(); 
        stack.clear();
    }
    
    void print() {
        std::stringstream ss;
        ss << "\n--- Solver Profiling (ms) ---\n";
        for(auto const& [name, time] : times) {
            ss << std::left << std::setw(20) << name << ": " << time << "\n";
        }
        ss << "-----------------------------";
        MLOG_INFO(ss.str());
    }
};

template<typename Model, int _MAX_N>
class MiniSolver {
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
    // Pass Model type to RiccatiSolver for static constraint info access
    std::unique_ptr<LinearSolver<TrajArray>> linear_solver;
    std::unique_ptr<LineSearchStrategy<Model, MAX_N>> line_search;

    int N; 
    std::array<double, MAX_N> dt_traj;
    
    Backend backend;
    SolverConfig config;
    SolverTimer timer; 
    
    double mu; 
    double reg;
    int current_iter = 0;
    
    // Lookup Maps
    std::unordered_map<std::string, int> state_map;
    std::unordered_map<std::string, int> control_map;
    std::unordered_map<std::string, int> param_map;

    MiniSolver(int initial_N, Backend b, SolverConfig conf = SolverConfig()) 
        : trajectory(initial_N), N(initial_N), backend(b), config(conf), mu(conf.mu_init), reg(conf.reg_init) {
        
        if (N > MAX_N) {
            std::cerr << "Error: N (" << N << ") > MAX_N (" << MAX_N << "). Clamping.\n";
            N = MAX_N;
        }

        dt_traj.fill(conf.default_dt);
        
        // Initialize Components with Model type
        linear_solver = std::make_unique<RiccatiSolver<TrajArray, Model>>();
        
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
    
    // Shifts trajectory for MPC warm start
    void shift_trajectory() {
        trajectory.shift();
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

    bool check_convergence(double max_viol, double max_dual) {
        if(mu <= config.mu_min && max_viol <= config.tol_con && max_dual <= config.tol_dual) return true;
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
        // Use MLOG_INFO instead of std::cout, respecting the log level.
        // We can check MINISOLVER_LOG_LEVEL against MLOG_LEVEL_INFO to avoid formatting cost if disabled.
        #if MINISOLVER_LOG_LEVEL < MLOG_LEVEL_INFO
            return;
        #endif

        if (config.print_level < PrintLevel::ITER) return;

        std::stringstream ss;
        if (header) {
            ss << std::left 
                      << std::setw(5) << "Iter" 
                      << std::setw(12) << "Cost" 
                      << std::setw(10) << "Log(Mu)" 
                      << std::setw(10) << "Log(Reg)" 
                      << std::setw(10) << "PrimInf" 
                      << std::setw(10) << "DualInf" 
                      << std::setw(10) << "Alpha";
            
            if (config.print_level >= PrintLevel::DEBUG) {
                ss << std::setw(12) << "MinSlack";
            }
            MLOG_INFO(ss.str());
            MLOG_INFO(std::string(80, '-'));
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
            // q_bar contains Vx (cost-to-go gradient / dynamics multiplier).
            // It is NOT a residual and should not be zero.
            // Only r_bar (control gradient) should be zero.
            double g_norm = 0.0; // MatOps::norm_inf(kp.q_bar);
            double r_norm = MatOps::norm_inf(kp.r_bar);
            double dual = std::max(g_norm, r_norm);
            if(dual > max_dual_inf) max_dual_inf = dual;
        }

        ss << std::left  
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
            ss << std::scientific << std::setprecision(2) << std::setw(12) << min_slack;
        }
        MLOG_INFO(ss.str());
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
    
    bool feasibility_restoration() {
        if (config.print_level >= PrintLevel::DEBUG) 
            MLOG_DEBUG("Entering Feasibility Restoration Phase.");
        double saved_mu = mu;
        double saved_reg = reg;
        mu = config.restoration_mu; 
        reg = config.restoration_reg; 
        
        auto& traj = trajectory.active();
        bool success = false;
        
        for(int r_iter=0; r_iter < config.max_restoration_iters; ++r_iter) { 
            for(int k=0; k<=N; ++k) {
                double current_dt = (k < N) ? dt_traj[k] : 0.0; 
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
                 success = true;
                 break;
            }

            // Restoration linear solve
            if(!linear_solver->solve(traj, N, mu, reg, config.inertia_strategy, config)) {
                break;
            }
            
            double alpha = 1.0;
            for(int k=0; k<=N; ++k) {
                for(int i=0; i<NC; ++i) {
                    double s = traj[k].s(i);
                    double ds = traj[k].ds(i);
                    double lam = traj[k].lam(i);
                    double dlam = traj[k].dlam(i);
                    if (ds < 0) alpha = std::min(alpha, -config.restoration_alpha * s / ds);
                    if (dlam < 0) alpha = std::min(alpha, -config.restoration_alpha * lam / dlam);
                }
            }
            
            if(alpha < 1e-4) {
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
        
        // --- Initialization of Slacks ---
        // Ensure slacks are consistent with initial constraints to avoid artificial PrimalInf
        {
            auto& traj = trajectory.active();
            for(int k=0; k<=N; ++k) {
                 double current_dt = (k < N) ? dt_traj[k] : 0.0;
                 Model::compute(traj[k], config.integrator, current_dt);
                 for(int i=0; i<NC; ++i) {
                     double g = traj[k].g_val(i);
                     double s_val = std::max(1.0, -g);
                     traj[k].s(i) = s_val;
                     traj[k].lam(i) = mu / s_val;
                 }
            }
        }
        
        print_iteration_log(0.0, true); 
        timer.reset(); 

        for(int iter=0; iter < config.max_iters; ++iter) {
            timer.start("Total Step");
            SolverStatus status = step();
            timer.stop();
            
            if (config.enable_rti) {
                if (config.print_level >= PrintLevel::INFO) 
                    MLOG_INFO("RTI Step Completed.");
                return SolverStatus::SOLVED; // RTI treats one step as 'done' for real-time loop
            }
            
            if (status == SolverStatus::SOLVED) {
                if (config.print_level >= PrintLevel::INFO) 
                    MLOG_INFO("Converged in " << iter+1 << " iterations.");
                if (config.print_level >= PrintLevel::INFO) timer.print();
                return SolverStatus::SOLVED;
            } else if (status != SolverStatus::UNSOLVED) {
                // Error or Infeasible
                if (config.print_level >= PrintLevel::INFO)
                     MLOG_INFO("Solver terminated with status: " << status_to_string(status));
                if (config.print_level >= PrintLevel::INFO) timer.print();
                return status;
            }
        }
        if (config.print_level >= PrintLevel::INFO) MLOG_INFO("Max iterations reached.");
        if (config.print_level >= PrintLevel::INFO) timer.print();
        
        // Final Feasibility Check
        auto& traj = trajectory.active();
        double max_viol = 0.0;
        double max_dual = 0.0;
        bool any_nan = false;
        for(int k=0; k<=N; ++k) {
             for(int i=0; i<NC; ++i) {
                 double v = std::abs(traj[k].g_val(i) + traj[k].s(i));
                 if (std::isnan(v)) any_nan = true;
                 if(v > max_viol) max_viol = v;
             }
             // double g_norm = MatOps::norm_inf(traj[k].q_bar);
             double r_norm = MatOps::norm_inf(traj[k].r_bar);
             if(r_norm > max_dual) max_dual = r_norm;
        }
        
        if (any_nan) return SolverStatus::NUMERICAL_ERROR;
        // If we reached here (MaxIter), check if we are at least feasible and stationary enough
        // to call it "FEASIBLE" (suboptimal but safe) or even "SOLVED" (if we just missed the loop check)
        if (check_convergence(max_viol, max_dual)) return SolverStatus::SOLVED;
        if (max_viol <= config.tol_con) return SolverStatus::FEASIBLE;
        
        return SolverStatus::MAX_ITER;
    }

    SolverStatus step() {
        current_iter++;
        auto& traj = trajectory.active();
        
        timer.start("Derivatives");
        double max_kkt_error = 0.0;
        double max_prim_inf = 0.0;
        double max_dual_inf = 0.0; 
        double total_gap = 0.0;
        int total_con = 0;

        for(int k=0; k<=N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;
            
            // Conditionally use GN or Exact compute
            if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                 Model::compute_cost_gn(traj[k]);
                 Model::compute_dynamics(traj[k], config.integrator, current_dt);
                 Model::compute_constraints(traj[k]);
            } else {
                 Model::compute_cost_exact(traj[k]);
                 Model::compute_dynamics(traj[k], config.integrator, current_dt);
                 Model::compute_constraints(traj[k]);
            }
            
            for(int i=0; i<NC; ++i) {
                // Correct Primal Inf calculation for Soft Constraints
                double viol = 0.0;
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                     if (i < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                     }
                }

                if (type == 1 && w > 1e-6) { // L1
                     viol = std::abs(traj[k].g_val(i) + traj[k].s(i) - traj[k].soft_s(i));
                } else if (type == 2 && w > 1e-6) { // L2
                     // g + s - lam/w = 0
                     viol = std::abs(traj[k].g_val(i) + traj[k].s(i) - traj[k].lam(i)/w);
                } else {
                     viol = std::abs(traj[k].g_val(i) + traj[k].s(i)); 
                }

                double comp = std::abs(traj[k].s(i) * traj[k].lam(i) - mu); 
                if(viol > max_kkt_error) max_kkt_error = viol;
                if(comp > max_kkt_error) max_kkt_error = comp;
                if(viol > max_prim_inf) max_prim_inf = viol;
            }
            total_gap += traj[k].s.dot(traj[k].lam);
            total_con += NC;
            
            // Approximate Dual Inf from r_bar (Control Gradient Stationarity)
            // Note: q_bar is Cost-to-Go gradient (lambda), which is NOT zero in general.
            double r_norm = MatOps::norm_inf(traj[k].r_bar);
            if(r_norm > max_dual_inf) max_dual_inf = r_norm;
        }
        timer.stop();
        
        // Initial convergence check (Primal + Dual + Mu)
        if(check_convergence(max_prim_inf, max_dual_inf)) return SolverStatus::SOLVED;

        // Check for Numerical Instability (NaN/Inf)
        if (has_nans(traj)) {
             if (config.print_level >= PrintLevel::INFO) 
                MLOG_ERROR("Numerical Error: NaNs detected in derivatives or state.");
             return SolverStatus::NUMERICAL_ERROR;
        }

        double avg_gap = (total_con > 0) ? (total_gap / total_con) : 0.0;
        
        // In Mehrotra mode, mu is updated dynamically inside the step via predictor-corrector logic.
        // We only use update_barrier for Monotone/Adaptive strategies or as a fallback.
        if (config.barrier_strategy != BarrierStrategy::MEHROTRA) {
            update_barrier(max_kkt_error, avg_gap);
        }
        
        if (config.line_search_type == LineSearchType::MERIT) {
             // line_search->prepare_step(traj); // If needed
        }

        timer.start("Linear Solve");
        bool solve_success = false;
        
        // Mehrotra Predictor-Corrector Logic
        if (config.barrier_strategy == BarrierStrategy::MEHROTRA) {
            // 1. Affine Step (Predictor)
            // Reuse candidate trajectory storage for affine step results
            trajectory.prepare_candidate();
            auto& affine_traj = trajectory.candidate();
            // Copy current state to affine traj to serve as linearization point base
            for(int k=0; k<=N; ++k) affine_traj[k] = traj[k];
            
            bool aff_success = false;
            // Solve with mu = 0
            for(int try_count=0; try_count < config.inertia_max_retries; ++try_count) {
                // Use new enum for exact/GN
                aff_success = linear_solver->solve(affine_traj, N, 0.0, reg, config.inertia_strategy, config);
                if (aff_success) break;
                if (reg < config.reg_min) reg = config.reg_min;
                reg *= config.reg_scale_up;
                if (reg > config.reg_max) reg = config.reg_max;
            }
            
            if (!aff_success) {
                timer.stop();
                return SolverStatus::NUMERICAL_ERROR;
            }
            
            // Calc max step for affine direction
            // Re-implement fraction-to-boundary locally since it's simple
            double alpha_aff = 1.0;
            for(int k=0; k<=N; ++k) {
                for(int i=0; i<NC; ++i) {
                    double s = affine_traj[k].s(i);
                    double ds = affine_traj[k].ds(i);
                    double lam = affine_traj[k].lam(i);
                    double dlam = affine_traj[k].dlam(i);
                    
                    if (ds < 0) {
                        double a = -s / ds; 
                        if (a < alpha_aff) alpha_aff = a;
                    }
                    if (dlam < 0) {
                        double a = -lam / dlam;
                        if (a < alpha_aff) alpha_aff = a;
                    }
                }
            }
            
            double mu_curr = mu;
            double mu_aff = 0.0;
            double total_comp = 0.0;
            int total_dim = 0;
            
            for(int k=0; k<=N; ++k) {
                for(int i=0; i<NC; ++i) {
                    double s_new = traj[k].s(i) + alpha_aff * affine_traj[k].ds(i);
                    double lam_new = traj[k].lam(i) + alpha_aff * affine_traj[k].dlam(i);
                    if(s_new < 0) s_new = 1e-8; // Should not happen with fraction_to_boundary
                    if(lam_new < 0) lam_new = 1e-8;
                    total_comp += s_new * lam_new;
                    total_dim++;
                }
            }
            mu_aff = total_comp / std::max(1, total_dim);
            
            // Aggressive Update: Use sigma^k with k >= 1
            // Heuristic: If affine step is good (large alpha_aff), be aggressive.
            // If alpha_aff is small, be conservative.
            double sigma_base = std::pow(mu_aff / mu_curr, 3);
            double sigma = sigma_base;
            
            // [NEW] Aggressive Strategy
            // If alpha_aff close to 1, we can reduce mu significantly
            if (config.enable_aggressive_barrier) {
                if (alpha_aff > 0.9) {
                    sigma = std::min(sigma, 0.1); // Force at least 10x reduction
                }
                // If we are far from solution (large gap), allow faster drop
                if (mu_curr > 1.0) {
                    sigma = std::min(sigma, 0.2); 
                }
            }
            
            if (sigma > 1.0) sigma = 1.0;
            double mu_target = sigma * mu_curr;
            if (mu_target < config.mu_min) mu_target = config.mu_min; // [FIX] Enforce lower bound
            
            // 2. Corrector Step
            // Solve with mu_target and affine correction term
            if (config.enable_corrector) {
                // Pass affine_traj to solve()
                for(int try_count=0; try_count < config.inertia_max_retries; ++try_count) {
                    solve_success = linear_solver->solve(traj, N, mu_target, reg, config.inertia_strategy, config, &affine_traj);
                    if (solve_success) break;
                    // If failed, regularize and retry (note: reg might have increased in affine step already)
                    reg *= config.reg_scale_up;
                    if (reg > config.reg_max) reg = config.reg_max;
                }
            } else {
                // Predictor only (just update mu but don't add correction term)
                for(int try_count=0; try_count < config.inertia_max_retries; ++try_count) {
                    solve_success = linear_solver->solve(traj, N, mu_target, reg, config.inertia_strategy, config);
                    if (solve_success) break;
                    reg *= config.reg_scale_up;
                    if (reg > config.reg_max) reg = config.reg_max;
                }
            }
            
            mu = mu_target;
            
        } else {
            // Standard IPM
            for(int try_count=0; try_count < config.inertia_max_retries; ++try_count) {
                solve_success = linear_solver->solve(traj, N, mu, reg, config.inertia_strategy, config);
                if (solve_success) break;
                if (reg < config.reg_min) reg = config.reg_min;
                reg *= config.reg_scale_up;
                if (reg > config.reg_max) reg = config.reg_max;
            }
        }

        if (solve_success && reg > config.reg_min) {
             reg = std::max(config.reg_min, reg / config.reg_scale_down);
        }
        timer.stop();

        if (!solve_success) return SolverStatus::NUMERICAL_ERROR;

        timer.start("Line Search");
        double alpha = line_search->search(trajectory, *linear_solver, dt_traj, mu, reg, config);
        
        if (alpha <= 1e-8) {
             // Step 1: Slack Reset
             bool recovered = false;
             if (config.enable_slack_reset && alpha < config.slack_reset_trigger) {
                 if (config.print_level >= PrintLevel::DEBUG) 
                    MLOG_DEBUG("Triggering Slack Reset.");
                 for(int k=0; k<=N; ++k) {
                     auto& kp = traj[k];
                     for(int i=0; i<NC; ++i) {
                         double min_s = std::abs(kp.g_val(i)) + std::sqrt(mu);
                         if (kp.s(i) < min_s) {
                             kp.s(i) = min_s;
                             // Fix: Don't kill dual variable if it was active. 
                             // Keep lam at least what it was, or consistent with mu_init (restarting barrier)
                             // kp.lam(i) = mu / kp.s(i); // This was the bug causing constraint loss
                             kp.lam(i) = std::max(kp.lam(i), config.mu_init / kp.s(i));
                         }
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

        print_iteration_log(alpha); // [FIX] Restore logging

        // Use Multiple Shooting: No forced rollout at the end of step
        // Trajectory is allowed to be discontinuous (Defect != 0) until convergence
        
        // Final check after step update - NO, we checked at start.
        // If we want to return SOLVED here, we need to check again.
        // But max_prim_inf is from START of step (x_k).
        // If x_k was feasible, we returned SOLVED at start.
        // So here we just return UNSOLVED to continue loop.
        // UNLESS we want to check if the step we just took made it feasible?
        // But we don't have new max_prim_inf.
        
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
