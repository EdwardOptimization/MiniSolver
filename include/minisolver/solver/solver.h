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

#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/trajectory.h" 
#include "minisolver/core/logger.h"

#include "minisolver/algorithms/linear_solver.h" 
#include "minisolver/algorithms/riccati_solver.h" 
#include "minisolver/algorithms/line_search.h" 

#include "minisolver/backend/backend_interface.h"

namespace minisolver {

class SolverTimer {
public:
    using Clock = std::chrono::steady_clock;
    std::map<std::string, double> times;
    std::vector<std::pair<std::string, std::chrono::time_point<Clock>>> stack;
    bool enabled = false; // Default disabled, use for test memory allocation

    void start(const std::string& name) {
        if (!enabled) return;
        stack.push_back({name, Clock::now()});
    }

    void stop() {
        if (!enabled) return;
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
        if (!enabled) return;
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

    using Knot = KnotPointV2<double, NX, NU, NC, NP>;
    using TrajectoryType = Trajectory<Knot, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    TrajectoryType trajectory;

    // Components
    // Pass Model type to RiccatiSolver for static constraint info access
    std::unique_ptr<RiccatiSolver<TrajArray, Model>> linear_solver;
    std::unique_ptr<LineSearchStrategy<Model, MAX_N>> line_search;

    int N; 
    std::array<double, MAX_N> dt_traj;
    
    Backend backend;
    SolverConfig config;
    SolverTimer timer; 
    
    double mu; 
    double reg;
    int current_iter = 0;
    double last_prim_inf = 0.0;
    double last_dual_inf = 0.0;

    // Continuous Slack Reset count, prevent infinite loop
    int slack_reset_consecutive_count = 0;

    bool is_warm_started = false;
    
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
    
    // Reset Function
    void reset(ResetOption option = ResetOption::ALG_STATE) {
        // 1. Reset Algorithmic Scalars
        mu = config.mu_init;
        reg = config.reg_init;
        current_iter = 0;
        last_prim_inf = 0.0;
        last_dual_inf = 0.0;
        slack_reset_consecutive_count = 0;
        is_warm_started = false; // Force Cold Start next time
        
        // 2. Reset Components
        if (line_search) line_search->reset();
        timer.reset();
        
        // 3. Reset Trajectory Data (Optional)
        if (option == ResetOption::FULL) {
            trajectory.reset();
            
            // Reset Time Steps to default configuration
            dt_traj.fill(config.default_dt);
            
            // Note: This clears parameters (p) too. 
            // You will need to call set_parameter() and set_dt() again after a FULL reset.
        }
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

    double get_parameter(int stage, const std::string& name) const {
        int idx = get_param_idx(name);
        if (idx != -1) return get_parameter(stage, idx);
        return 0.0;
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
        
        // [FIX] Initialize remaining steps to avoid garbage values
        double fill_val = (count > 0) ? dts[count-1] : config.default_dt;
        for(int i=count; i<MAX_N; ++i) dt_traj[i] = fill_val;
    }
    
    void set_dt(double dt) {
        dt_traj.fill(dt);
    }
    
    // Shifts trajectory for MPC warm start
    // Only shift values (x, u, s, lam), keep parameters (p) unchanged
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
        is_warm_started = true;
    }

    bool check_convergence(double max_viol, double max_dual, double max_kkt_error) {
        // must satisfy all of the following conditions:
        // 1. barrier parameter mu is small enough (target precision)
        // 2. primal constraints are satisfied (Feasible)
        // 3. dual gradient is satisfied (Stationary)
        // 4. complementarity is satisfied (Complementarity): s*lam is close to mu
        
        bool mu_converged = (mu <= config.mu_final);
        bool primal_ok = (max_viol <= config.tol_con);
        bool dual_ok = (max_dual <= config.tol_dual);
        
        // complementarity error tolerance is usually set to kappa * mu or a slightly relaxed fixed value
        // here we require it to converge to tol_cost or tol_mu level
        bool kkt_ok = (max_kkt_error <= std::max(config.tol_mu, 10.0 * mu));
        
        return mu_converged && primal_ok && dual_ok && kkt_ok;
    }

    void update_barrier(double max_kkt_error, double avg_gap) {
        switch(config.barrier_strategy) {
            case BarrierStrategy::MONOTONE:
                if (max_kkt_error < config.barrier_tolerance_factor * mu) {
                    double next_mu = std::max(config.mu_final, mu * config.mu_linear_decrease_factor);
                    mu = next_mu;
                }
                break;
            case BarrierStrategy::ADAPTIVE: {
                double target = avg_gap * config.mu_safety_margin; 
                // Removed forced decrease to allow mu to hold steady if needed for convergence
                mu = std::max(config.mu_final, std::min(mu, target));
                break;
            }
            case BarrierStrategy::MEHROTRA: {
                double ratio = avg_gap / mu;
                if(ratio > 1.0) ratio = 1.0; 
                double sigma = std::pow(ratio, 3);
                if(sigma < 0.05) sigma = 0.05;
                if(sigma > 0.8) sigma = 0.8;
                double next_mu = std::max(config.mu_final, mu * sigma);
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
        double max_dual_inf = 0.0;
        double min_slack = 1e9;
        auto& traj = trajectory.active();

        // Use helper for Primal Inf
        double max_prim_inf = compute_max_violation(traj);

        for(int k=0; k<=N; ++k) {
            const auto& kp = traj[k];
            total_cost += kp.cost;
            for(int i=0; i<NC; ++i) {
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
        
        auto* active_state = trajectory.get_active_state();
        auto* model_data = trajectory.get_model_data();
        bool success = false;
        
        for(int r_iter=0; r_iter < config.max_restoration_iters; ++r_iter) { 
            for(int k=0; k<=N; ++k) {
                double current_dt = (k < N) ? dt_traj[k] : 0.0; 
                Model::compute_dynamics(active_state[k], model_data[k], config.integrator, current_dt);
                Model::compute_constraints(active_state[k], model_data[k]);
                
                // NEW: Write to model_data
                model_data[k].Q.setIdentity(); 
                model_data[k].q.setZero();
                
                if (k < N) {
                    model_data[k].R.setIdentity(); 
                    model_data[k].r.setZero();
                } else {
                    model_data[k].R.setZero();
                    model_data[k].r.setZero();
                }
            }
            
            // Restoration linear solve
            // [ALADIN-Inspired] Augmented Lagrangian Restoration
            // Minimizing 0.5*||dx||^2 + 0.5*rho*||C*dx + D*du + g + s||^2
            // This pulls the solution towards the constraint manifold more aggressively than simple min-norm.
            if (config.barrier_strategy != BarrierStrategy::MEHROTRA) { 
                 double rho = 1000.0; // Penalty weight from ALADIN concepts
                 for(int k=0; k<=N; ++k) {
                     auto& md = model_data[k];
                     
                     // Q += rho * C^T * C
                     md.Q.noalias() += rho * md.C.transpose() * md.C;
                     
                     // R += rho * D^T * D
                     md.R.noalias() += rho * md.D.transpose() * md.D;
                     
                     // H += rho * D^T * C (Cross term)
                     md.H.noalias() += rho * md.D.transpose() * md.C;
                     
                     // q += rho * C^T * g_val
                     // Note: Restoration usually ignores 's' in the quadratic penalty approximation 
                     // or treats it as fixed residuals g_val.
                     md.q.noalias() += rho * md.C.transpose() * active_state[k].g_val;
                     
                     // r += rho * D^T * g_val
                     md.r.noalias() += rho * md.D.transpose() * active_state[k].g_val;
                 }
            }

            auto* workspace = trajectory.get_workspace();
            
            if(!linear_solver->solve(trajectory, N, mu, reg, config.inertia_strategy, config)) {
                break;
            }
            
            double alpha = 1.0;
            for(int k=0; k<=N; ++k) {
                for(int i=0; i<NC; ++i) {
                    double s = active_state[k].s(i);
                    double ds = workspace[k].ds(i);
                    double lam = active_state[k].lam(i);
                    double dlam = workspace[k].dlam(i);
                    if (ds < 0) alpha = std::min(alpha, -config.restoration_alpha * s / ds);
                    if (dlam < 0) alpha = std::min(alpha, -config.restoration_alpha * lam / dlam);
                }
            }
            
            if(alpha < 1e-4) {
                 break; 
            }

            for(int k=0; k<=N; ++k) {
                active_state[k].x += alpha * workspace[k].dx;
                active_state[k].u += alpha * workspace[k].du;
                active_state[k].s += alpha * workspace[k].ds;
                active_state[k].lam += alpha * workspace[k].dlam;
            }
            // rollout_dynamics(); // Don't force rollout in restoration. Allow defects to be handled by solver.
        }
        
        // Reset Lagrange Multipliers for the original problem to avoid dual contamination
        // form the restoration phase (which solves a different problem).
        for(int k=0; k<=N; ++k) {
             for(int i=0; i<NC; ++i) {
                  // Ensure s is positive
                  if(active_state[k].s(i) < 1e-9) active_state[k].s(i) = 1e-9;
                  
                  // Preserve dual info from restoration if valuable, but ensure complementarity lower bound
                  double reset_val = saved_mu / active_state[k].s(i);
                  active_state[k].lam(i) = std::max(active_state[k].lam(i), reset_val);
             }
        }

        mu = saved_mu;
        reg = saved_reg;

        return success;
    }

    SolverStatus solve() {
        // 1. Presolve: 数据准备、冷热启动处理、内存复位
        presolve();
        
        SolverStatus loop_exit_status = SolverStatus::UNSOLVED;
        double last_cost = 1e30;

        // 2. Solve Loop: 数值迭代
        for(int iter=0; iter < config.max_iters; ++iter) {
            timer.start("Total Step");
            SolverStatus step_stat = step();
            timer.stop();
            
            // SQP-RTI 模式：做一步即走，状态由 postsolve 判定
            if (config.enable_rti) {
                loop_exit_status = SolverStatus::UNSOLVED; 
                break;
            }

            // A. 完美收敛 -> 记录状态并跳出，交给 postsolve 复核
            if (step_stat == SolverStatus::OPTIMAL) {
                loop_exit_status = SolverStatus::OPTIMAL;
                if (config.print_level >= PrintLevel::INFO) 
                    MLOG_INFO("Converged in " << iter+1 << " iterations.");
                break;
            }
            
            // B. 数值错误 -> 立即中止
            if (step_stat == SolverStatus::NUMERICAL_ERROR) {
                loop_exit_status = SolverStatus::NUMERICAL_ERROR;
                break;
            }

            // C. 停滞检查 (Cost Stagnation)
            if (mu <= config.mu_final) {
                // 计算当前 Cost
                double current_cost = 0.0;
                for(int k=0; k<=N; ++k) current_cost += trajectory.active()[k].cost;
                
                // 只有在满足一定可行性时，Cost 停滞才有意义
                // 使用上一步计算的 max_prim_inf (last_prim_inf)
                double feasible_bound = config.tol_con * config.feasible_tol_scale;
                
                if (last_prim_inf <= feasible_bound) {
                    double cost_diff = std::abs(current_cost - last_cost);
                    if (cost_diff < config.tol_cost) {
                        if (config.print_level >= PrintLevel::INFO) {
                            MLOG_INFO("Cost Stagnation detected. Stopping early.");
                        }
                        // 标记为 UNSOLVED，表示非自然收敛，交给 postsolve 判定是 Feasible 还是 Optimal
                        loop_exit_status = SolverStatus::UNSOLVED; 
                        break; 
                    }
                }
                last_cost = current_cost;
            }
        }
        
        // 3. Postsolve: 扫尾、刷新导数、统一评级
        return postsolve(loop_exit_status);
    }

    SolverStatus step() {
        current_iter++;
        
        timer.start("Derivatives");
        double max_kkt_error = 0.0;
        double max_prim_inf = 0.0;
        double max_dual_inf = 0.0; 
        double total_gap = 0.0;
        int total_con = 0;

        // NEW: Get pointers to three layers
        auto* active_state = trajectory.get_active_state();
        auto* model_data = trajectory.get_model_data();
        auto* workspace = trajectory.get_workspace();
        
        for(int k=0; k<=N; ++k) {
            double current_dt = (k < N) ? dt_traj[k] : 0.0;
            
            // NEW: Pass state and model separately
            if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                 Model::compute_cost_gn(active_state[k], model_data[k]);
                 Model::compute_dynamics(active_state[k], model_data[k], config.integrator, current_dt);
                 Model::compute_constraints(active_state[k], model_data[k]);
            } else {
                 Model::compute_cost_exact(active_state[k], model_data[k]);
                 Model::compute_dynamics(active_state[k], model_data[k], config.integrator, current_dt);
                 Model::compute_constraints(active_state[k], model_data[k]);
            }
            
            for(int i=0; i<NC; ++i) {
                double comp = std::abs(active_state[k].s(i) * active_state[k].lam(i) - mu); 
                if(comp > max_kkt_error) max_kkt_error = comp;
            }
            total_gap += active_state[k].s.dot(active_state[k].lam);
            total_con += NC;
            
            // Approximate Dual Inf from r_bar (Control Gradient Stationarity)
            // Note: q_bar is Cost-to-Go gradient (lambda), which is NOT zero in general.
            double r_norm = MatOps::norm_inf(workspace[k].r_bar);
            if(r_norm > max_dual_inf) max_dual_inf = r_norm;
        }
        
        // Use helper for Primal Inf
        max_prim_inf = compute_max_violation(trajectory);
        
        timer.stop();
        
        // Initial convergence check (Primal + Dual + Mu)
        if(check_convergence(max_prim_inf, max_dual_inf, max_kkt_error)) return SolverStatus::OPTIMAL;

        // Check for Numerical Instability (NaN/Inf)
        if (has_nans(trajectory)) {
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
        
        // [NEW] Zero-Malloc Iterative Refinement Preparation
        // If IR is enabled, we backup the current active trajectory (linearized system)
        // to the candidate buffer. This allows us to access the original matrices (Q, R, A, B...)
        // during the refinement step, even after the Riccati solver overwrites them in 'active'.
        // bool do_refinement = config.enable_iterative_refinement && (current_iter % config.max_refinement_steps == 0); // Example trigger
        if (config.enable_iterative_refinement) {
             trajectory.prepare_candidate();
             auto& backup = trajectory.candidate();
             // Copy active to candidate. Since TrajArray is std::array<Knot>, this is a contiguous copy.
             // But Knot contains Eigen matrices. std::copy should handle it correctly via assignment operators.
             std::copy(traj.begin(), traj.end(), backup.begin());
        }
        
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
            
            // DEBUG PRINT
            if (config.print_level >= PrintLevel::DEBUG) {
                 MLOG_INFO("Mehrotra Debug: mu_curr=" << mu_curr << ", mu_aff=" << mu_aff << ", alpha_aff=" << alpha_aff);
            }

            // Aggressive Update: Use sigma^k with k >= 1
            // Heuristic: If affine step is good (large alpha_aff), be aggressive.
            // If alpha_aff is small, be conservative.
            double sigma_base = std::pow(mu_aff / mu_curr, 3);
            double sigma = sigma_base;
            
            // Aggressive Strategy
            // If alpha_aff close to 1, we can reduce mu significantly
            if (config.enable_aggressive_barrier) {
                if (alpha_aff > 0.9) {
                    sigma = std::min(sigma, 0.01); // [AGGRESSIVE] Force 100x reduction if direction is good (was 0.1)
                }
                // If we are far from solution (large gap), allow faster drop
                if (mu_curr > 1.0) {
                    sigma = std::min(sigma, 0.1); // [AGGRESSIVE] Was 0.2
                }
            } else {
                // Mehrotra Centering Parameter Heuristic
                // Limit sigma to avoid aggressive reduction when affine step is bad
                if (alpha_aff < 0.1) {
                    // If affine direction is blocked quickly, we are close to boundary.
                    // Be conservative to allow centering.
                    sigma = std::max(sigma, 0.5); 
                } else if (alpha_aff > 0.9) {
                    // If affine direction is good, we can reduce mu significantly
                    sigma = std::min(sigma, 0.1);
                }
            }
            
            if (sigma > 1.0) sigma = 1.0;
            if (sigma < 1e-4) sigma = 1e-4; // Prevent too small sigma
            
            double mu_target = sigma * mu_curr;
            if (mu_target < config.mu_final) mu_target = config.mu_final; // Enforce lower bound
            
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
        
        // Iterative Refinement
        if (solve_success && config.enable_iterative_refinement) {
             // Pass 'traj' (which contains solution dx, du) and 'candidate' (which contains original system)
             linear_solver->refine(traj, trajectory.candidate(), N, mu, reg, config);
        }
        
        timer.stop();

        if (!solve_success) return SolverStatus::NUMERICAL_ERROR;

        timer.start("Line Search");
        double alpha = line_search->search(trajectory, *linear_solver, dt_traj, mu, reg, config);
        
        // If step size is valid, reset the counter
        if (alpha > 1e-8) {
            slack_reset_consecutive_count = 0;
        }
        else {
            // Alpha <= 1e-8 (Step size too small / Stagnation)

            // 1. Check if stagnated at a feasible solution satisfying tolerances (Optimality check)
            //    This prevents triggering incorrect restoration mechanisms due to numerical noise near a perfect solution
            if (max_prim_inf <= config.tol_con) {
                // Further check dual feasibility to distinguish between SOLVED and FEASIBLE
                if (max_dual_inf <= config.tol_dual) {
                    if (config.print_level >= PrintLevel::INFO) {
                        MLOG_INFO("Line search stagnated at optimal point (PrimInf: " << max_prim_inf 
                                << ", DualInf: " << max_dual_inf << "). Terminating as SOLVED.");
                    }
                    timer.stop();
                    return SolverStatus::OPTIMAL;
                } else {
                    if (config.print_level >= PrintLevel::INFO) {
                        MLOG_INFO("Line search stagnated at feasible point (PrimInf: " << max_prim_inf 
                                << "). Terminating as FEASIBLE.");
                    }
                    timer.stop();
                    return SolverStatus::FEASIBLE;
                }
            }

            // 2. Step 1: Slack Reset (With counter protection)
            bool recovered = false;
            
            // Logic explanation:
            // Only allow attempt when slack_reset_consecutive_count < 1.
            // This means if SlackReset was used in the last iteration but Alpha=0 still (dead loop),
            // this time we force skipping Step 1 and go directly to Step 2 (Restoration).
            if (config.enable_slack_reset && 
                alpha < config.slack_reset_trigger && 
                slack_reset_consecutive_count < 1) 
            {
                if (config.print_level >= PrintLevel::DEBUG) 
                MLOG_DEBUG("Triggering Slack Reset (Attempt " << slack_reset_consecutive_count + 1 << ").");
                
                for(int k=0; k<=N; ++k) {
                    auto& kp = traj[k];
                    for(int i=0; i<NC; ++i) {
                        double min_s = std::abs(kp.g_val(i)) + std::sqrt(mu);
                        if (kp.s(i) < min_s) {
                            kp.s(i) = min_s;
                            // Maintain consistency of dual variables, prevent aggressive reset
                            kp.lam(i) = std::max(kp.lam(i), config.mu_init / kp.s(i));
                        }
                    }
                }
                // Try one solve to see if a valid direction can be obtained
                recovered = linear_solver->solve(traj, N, mu, reg, config.inertia_strategy, config);
                
                if (recovered) {
                    // Mark: If this Reset failed to get us out of trouble, disallow it next time
                    slack_reset_consecutive_count++;
                }
            }
            else if (config.enable_slack_reset && slack_reset_consecutive_count >= 1) {
                if (config.print_level >= PrintLevel::DEBUG) 
                MLOG_DEBUG("Skipping Slack Reset to prevent cycle. Forcing Restoration.");
            }
            
            // 3. Step 2: Feasibility Restoration (If Step 1 failed or was skipped)
            if (!recovered && config.enable_feasibility_restoration) {
                recovered = feasibility_restoration();
                // If restoration succeeded (state x moved), we can reset the counter, allowing SlackReset to be used again in the future
                if (recovered) {
                    slack_reset_consecutive_count = 0;
                }
            }

            if (!recovered) {
                // Both Slack Reset and Feasibility Restoration failed (or were disabled)
                timer.stop();
                print_iteration_log(alpha); // Log the fail state
                return SolverStatus::INFEASIBLE;
            }
        }
        timer.stop();

        print_iteration_log(alpha); // Restore logging

        // Final Convergence Check using Step Size and Residuals
        // Avoids wasting a full derivative computation in next step if we are already done.
        bool is_feasible = (max_prim_inf < config.tol_con);
        bool is_dual_feasible = (max_dual_inf < config.tol_dual);
        
        // 1. Standard "Small Mu" Convergence
        if (mu <= config.mu_final && alpha > 1e-5) {
            // Check stationarity via step size
            double max_dx = 0.0;
            for(int k=0; k<=N; ++k) {
                // Use MatOps::norm_inf to support both Eigen and MiniMatrix
                double dx_norm = MatOps::norm_inf(trajectory.active()[k].dx);
                if (dx_norm > max_dx) max_dx = dx_norm;
            }
            // Use unscaled Newton step (max_dx) to check stationarity.
            // Using (alpha * max_dx) is dangerous because small alpha (blocked step) 
            // can look like convergence.
            if (max_dx < config.tol_grad && is_feasible && is_dual_feasible) {
                return SolverStatus::OPTIMAL;
            }
        }
        
        last_prim_inf = max_prim_inf;
        last_dual_inf = max_dual_inf;
        return SolverStatus::UNSOLVED;
    }

    void rollout_dynamics() {
        auto& traj = trajectory.active();
        for(int k=0; k<N; ++k) {
            double current_dt = dt_traj[k];
            traj[k+1].x = Model::integrate(traj[k].x, traj[k].u, traj[k].p, current_dt, config.integrator);
        }
    }

private:
    // ============================================================
    // [Phase 1] Presolve: Preparation
    // ============================================================
    void presolve() {
        // [Enable/Disable Profiling]
        timer.enabled = config.enable_profiling;
        
        current_iter = 0;
        slack_reset_consecutive_count = 0; 
        reg = config.reg_init;
        // 1. Reset Logic
        if (!is_warm_started) {
            mu = config.mu_init;
        }
        // 2. Initialization of Slacks/Duals
        bool need_init = !is_warm_started;
        
        // 安全检查：如果 Warm Start 数据损坏，强制重置
        if (is_warm_started) {
             auto& traj = trajectory.active();
             for(int k=0; k<=N; ++k) {
                 if (traj[k].s.minCoeff() <= 0 || traj[k].lam.minCoeff() <= 0 || has_nans(traj)) {
                     need_init = true;
                     break;
                 }
             }
        }
        is_warm_started = false;
        if (need_init) {
            auto& traj = trajectory.active();
            for(int k=0; k<=N; ++k) {
                 double current_dt = (k < N) ? dt_traj[k] : 0.0;
                 Model::compute(traj[k], config.integrator, current_dt);
                 
                 for(int i=0; i<NC; ++i) {
                     double g = traj[k].g_val(i);
                     double w = 0.0;
                     int type = 0;
                     if constexpr (NC > 0) {
                         if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                            type = Model::constraint_types[i];
                            w = Model::constraint_weights[i];
                         }
                     }
                     
                     if (type == 1 && w > 1e-6) { // L1 Soft Constraint
                         // Central Path: 
                         // 1) g + s - soft_s = 0
                         // 2) s * lam = mu
                         // 3) soft_s * (w - lam) = mu
                         // Reduce to quadratic in lam: g*lam^2 - (g*w - 2*mu)*lam - mu*w = 0
                         double a = g;
                         double b = -(g * w - 2 * mu);
                         double c = -mu * w;
                         
                         double lam_val;
                         if (std::abs(a) < 1e-9) {
                             // Linear case (g approx 0): -(-2mu)lam - mu*w = 0 -> 2mu*lam = mu*w -> lam = w/2
                             lam_val = w / 2.0;
                         } else {
                            double delta = b*b - 4*a*c;
                            if (delta < 0) delta = 0;
                            if (std::abs(a) < 1e-9) {
                                lam_val = w / 2.0;
                            } else {
                                double delta = b*b - 4*a*c;
                                if (delta < 0) delta = 0;
                                // Use plus sign formula
                                lam_val = (-b + std::sqrt(delta)) / (2*a);
                            }
                         }
                         
                         // Clamp for safety
                         lam_val = std::max(1e-8, std::min(w - 1e-8, lam_val));
                         
                         traj[k].lam(i) = lam_val;
                         traj[k].s(i) = mu / lam_val;
                         traj[k].soft_s(i) = mu / (w - lam_val);
                     } 
                     else if (type == 2 && w > 1e-6) { // L2 Soft Constraint
                         // Central Path:
                         // 1) g + s - lam/w = 0
                         // 2) s * lam = mu
                         // Reduce to quadratic in lam: lam^2 - g*w*lam - mu*w = 0
                         double b = -g * w;
                         double c = -mu * w;
                         // lam = (-b + sqrt(b^2 - 4ac)) / 2a, here a=1
                         // lam = (g*w + sqrt(g^2*w^2 + 4*mu*w)) / 2
                         double delta = b*b - 4*c; // b^2 + 4*mu*w > 0 always
                         double lam_val = (-b + std::sqrt(delta)) / 2.0;
                         
                         traj[k].lam(i) = std::max(1e-8, lam_val);
                         traj[k].s(i) = mu / traj[k].lam(i);
                         // soft_s not used in L2
                     } 
                     else { // Hard Constraint
                         double s_val = std::max(1e-6, -g);
                         traj[k].s(i) = s_val;
                         traj[k].lam(i) = mu / s_val;
                     }
                 }
            }
        }
        
        print_iteration_log(0.0, true); 
        timer.reset(); 
    }

    // ============================================================
    // [Phase 3] Postsolve: Finalization & Verdict
    // ============================================================
    SolverStatus postsolve(SolverStatus loop_status) {
        if (loop_status == SolverStatus::NUMERICAL_ERROR) {
            return SolverStatus::NUMERICAL_ERROR;
        }
        if (config.print_level >= PrintLevel::INFO) {
            if (loop_status == SolverStatus::UNSOLVED) MLOG_INFO("Max iterations or stagnation.");
        }
        // [Fix 2 Logic] 强制刷新导数
        // 无论是因为收敛还是因为耗尽步数退出，我们都重新计算一次精确的残差，
        // 以便做出最公正的最终评判。
        auto* active_state = trajectory.get_active_state();
        auto* model_data = trajectory.get_model_data();
        double max_kkt_error = 0.0;
        double max_dual_inf = 0.0;
        auto& riccati_workspace = linear_solver->workspace;
        for(int k=0; k<=N; ++k) {
             double current_dt = (k < N) ? dt_traj[k] : 0.0;
             
             // 1. Recompute Primal/Dual properties
             if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                 Model::compute_cost_gn(active_state[k], model_data[k]);
                 Model::compute_dynamics(active_state[k], model_data[k], config.integrator, current_dt);
                 Model::compute_constraints(active_state[k], model_data[k]);
             } else {
                 Model::compute_cost_exact(active_state[k], model_data[k]);
                 Model::compute_dynamics(active_state[k], model_data[k], config.integrator, current_dt);
                 Model::compute_constraints(active_state[k], model_data[k]);
             }
             
             // 2. Recompute Barrier Gradients (check riccati.h)
             compute_barrier_derivatives<Knot, Model>(traj[k], mu, config, riccati_workspace, nullptr, nullptr);
             // 3. Check NaNs
             for(int i=0; i<NC; ++i) {
                 if (std::isnan(traj[k].g_val(i)) || std::isnan(traj[k].s(i))) 
                     return SolverStatus::NUMERICAL_ERROR;
             }
             // 4. Collect Metrics
             double r_norm = MatOps::norm_inf(traj[k].r_bar);
             if(r_norm > max_dual_inf) max_dual_inf = r_norm;
             for(int i=0; i<NC; ++i) {
                double comp = std::abs(traj[k].s(i) * traj[k].lam(i) - mu);
                if(comp > max_kkt_error) max_kkt_error = comp;
             }
        }
        double max_viol = compute_max_violation(traj);
        // [最终评级]
        
        // Level 1: SOLVED (Optimal)
        // 即使 Loop 是因为 Stagnation 退出的，如果此时恰好满足最优性，也给 SOLVED
        if (check_convergence(max_viol, max_dual_inf, max_kkt_error)) {
            return SolverStatus::OPTIMAL;
        }
        // Level 2: FEASIBLE (Acceptable)
        double feasible_bound = config.tol_con * config.feasible_tol_scale;
        if (max_viol <= feasible_bound) {
            if (config.print_level >= PrintLevel::INFO) {
                MLOG_INFO("Result: FEASIBLE (Viol: " << max_viol << " <= " << feasible_bound << ")");
            }
            return SolverStatus::FEASIBLE;
        }
        // Level 3: INFEASIBLE (Failed)
        if (config.print_level >= PrintLevel::WARN) {
            MLOG_WARN("Result: INFEASIBLE (Viol: " << max_viol << " > " << feasible_bound << ")");
        }
        return SolverStatus::INFEASIBLE;
    }

private:
    // helper function: calculate the maximum constraint violation of the current trajectory
    double compute_max_violation(const TrajArray& traj) const {
        double max_viol = 0.0;
        for(int k=0; k<=N; ++k) {
            const auto& kp = traj[k];
            for(int i=0; i<NC; ++i) {
                double viol = 0.0;
                
                // get the constraint weight and type
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                     if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                     }
                }
                // unified violation calculation logic
                if (type == 1 && w > 1e-6) { // L1
                     viol = std::abs(kp.g_val(i) + kp.s(i) - kp.soft_s(i));
                } else if (type == 2 && w > 1e-6) { // L2
                     viol = std::abs(kp.g_val(i) + kp.s(i) - kp.lam(i)/w);
                } else { // Hard
                     viol = std::abs(kp.g_val(i) + kp.s(i));
                }
                if(viol > max_viol) max_viol = viol;
            }
        }
        return max_viol;
    }
};
}
