#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

#include "core/types.h"
#include "core/solver_options.h"
#include "core/trajectory.h"
#include "algorithms/linear_solver.h"
#include "solver/line_search.h" // Helper

namespace minisolver {

template<typename Model, int MAX_N>
class LineSearchStrategy {
public:
    using TrajectoryType = Trajectory<KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    virtual ~LineSearchStrategy() = default;

    virtual double search(TrajectoryType& trajectory, 
                          LinearSolver<TrajArray>& linear_solver,
                          const std::array<double, MAX_N>& dt_traj,
                          double mu, double reg,
                          const SolverConfig& config) = 0;
                          
    virtual void prepare_step(const TrajArray& traj) {}
    virtual void reset() {}
};

// --- Merit Function Strategy ---
template<typename Model, int MAX_N>
class MeritLineSearch : public LineSearchStrategy<Model, MAX_N> {
    using Base = LineSearchStrategy<Model, MAX_N>;
    using typename Base::TrajectoryType;
    using typename Base::TrajArray;
    
    double merit_nu = 1000.0;

    double compute_merit_N(const TrajArray& t, int N, double mu, const SolverConfig& config) {
        double total_merit = 0.0;
        const int NC = Model::NC;
        for(int k=0; k<=N; ++k) {
            const auto& kp = t[k];
            total_merit += kp.cost; 
            for(int i=0; i<NC; ++i) {
                if(kp.s(i) > config.min_barrier_slack) 
                    total_merit -= mu * std::log(kp.s(i));
                else 
                    total_merit += config.barrier_inf_cost; 
            }
            for(int i=0; i<NC; ++i) {
                total_merit += merit_nu * std::abs(kp.g_val(i) + kp.s(i));
            }
        }
        return total_merit;
    }

public:
    void reset() override {
        merit_nu = 1000.0;
    }

    double search(TrajectoryType& trajectory, 
                  LinearSolver<TrajArray>& linear_solver,
                  const std::array<double, MAX_N>& dt_traj,
                  double mu, double reg,
                  const SolverConfig& config) override 
    {
        int N = trajectory.N;
        auto& active = trajectory.active();
        
        // 1. Update Nu
        double max_dual = 0.0;
        for(int k=0; k<=N; ++k) {
            double local_max = active[k].lam.template lpNorm<Eigen::Infinity>();
            if(local_max > max_dual) max_dual = local_max;
        }
        double required_nu = max_dual * 1.1 + 1.0; 
        if (required_nu > merit_nu) merit_nu = required_nu;

        // 2. Initial Merit
        double phi_0 = compute_merit_N(active, N, mu, config);
        
        // 3. Calc max alpha
        double alpha = fraction_to_boundary_rule(active, N, config.line_search_tau);

        trajectory.prepare_candidate();
        auto& candidate = trajectory.candidate();
        
        bool accepted = false;
        int ls_iter = 0;
        
        while (ls_iter < config.line_search_max_iters) {
            for(int k=0; k<=N; ++k) {
                candidate[k].x = active[k].x + alpha * active[k].dx;
                candidate[k].u = active[k].u + alpha * active[k].du;
                candidate[k].s = active[k].s + alpha * active[k].ds;
                candidate[k].lam = active[k].lam + alpha * active[k].dlam;
                candidate[k].p = active[k].p; 
                
                double current_dt = dt_traj[k];
                Model::compute(candidate[k], config.integrator, current_dt);
            }
            
            double phi_alpha = compute_merit_N(candidate, N, mu, config);
            if (phi_alpha < phi_0) {
                accepted = true;
            }

            if (accepted) break;
            alpha *= config.line_search_backtrack_factor; // Use config
            ls_iter++;
        }

        if (accepted) {
            trajectory.swap();
        }
        
        return alpha;
    }
};

// --- Filter Strategy ---
template<typename Model, int MAX_N>
class FilterLineSearch : public LineSearchStrategy<Model, MAX_N> {
    using Base = LineSearchStrategy<Model, MAX_N>;
    using typename Base::TrajectoryType;
    using typename Base::TrajArray;
    
    std::vector<std::pair<double, double>> filter;

    std::pair<double, double> compute_metrics(const TrajArray& t, int N, double mu, const SolverConfig& config) {
        double theta = 0.0; 
        double phi = 0.0;   
        const int NC = Model::NC;
        for(int k=0; k<=N; ++k) {
            const auto& kp = t[k];
            phi += kp.cost;
            for(int i=0; i<NC; ++i) {
                if(kp.s(i) > config.min_barrier_slack) phi -= mu * std::log(kp.s(i));
                else phi += config.barrier_inf_cost;
            }
            for(int i=0; i<NC; ++i) {
                theta += std::abs(kp.g_val(i) + kp.s(i));
            }
        }
        return {theta, phi};
    }
    
    bool is_acceptable(double theta, double phi, const SolverConfig& config) {
        for(const auto& entry : filter) {
            double theta_j = entry.first;
            double phi_j = entry.second;
            bool sufficient_theta = theta < (1.0 - config.filter_gamma_theta) * theta_j;
            bool sufficient_phi = phi < phi_j - config.filter_gamma_phi * theta_j;
            if (!sufficient_theta && !sufficient_phi) return false; 
        }
        return true;
    }

public:
    void reset() override {
        filter.clear();
    }
    
    void solve_soc(TrajArray& soc_traj, const TrajArray& active_traj, const TrajArray& trial_traj, 
                   int N, double mu, double reg, InertiaStrategy inertia,
                   LinearSolver<TrajArray>& solver, const SolverConfig& config) {
        for(int k=0; k<=N; ++k) soc_traj[k] = active_traj[k];
        
        for(int k=0; k<=N; ++k) {
            soc_traj[k].g_val = trial_traj[k].g_val + (trial_traj[k].s - active_traj[k].s);
        }
        
        bool success = solver.solve(soc_traj, N, mu, reg, inertia, config);
        if (!success) {
            for(int k=0; k<=N; ++k) {
                soc_traj[k].dx.setZero(); soc_traj[k].du.setZero(); 
                soc_traj[k].ds.setZero(); soc_traj[k].dlam.setZero();
            }
        }
    }

    double search(TrajectoryType& trajectory, 
                  LinearSolver<TrajArray>& linear_solver,
                  const std::array<double, MAX_N>& dt_traj,
                  double mu, double reg,
                  const SolverConfig& config) override 
    {
        int N = trajectory.N;
        auto& active = trajectory.active();
        
        auto m_0 = compute_metrics(active, N, mu, config);
        double theta_0 = m_0.first;
        double phi_0 = m_0.second;
        
        double alpha = fraction_to_boundary_rule(active, N, config.line_search_tau);
        
        trajectory.prepare_candidate();
        auto& candidate = trajectory.candidate();
        
        bool accepted = false;
        int ls_iter = 0;
        bool soc_attempted = false;
        
        while (ls_iter < config.line_search_max_iters) {
            for(int k=0; k<=N; ++k) {
                candidate[k].x = active[k].x + alpha * active[k].dx;
                candidate[k].u = active[k].u + alpha * active[k].du;
                candidate[k].s = active[k].s + alpha * active[k].ds;
                candidate[k].lam = active[k].lam + alpha * active[k].dlam;
                candidate[k].p = active[k].p; 
                
                double current_dt = dt_traj[k];
                Model::compute(candidate[k], config.integrator, current_dt);
            }
            
            auto m_alpha = compute_metrics(candidate, N, mu, config);
            if (is_acceptable(m_alpha.first, m_alpha.second, config)) {
                accepted = true;
            }
            
            // SOC Logic
            if (!accepted && config.enable_soc && !soc_attempted && ls_iter == 0 && alpha > config.soc_trigger_alpha) {
                if (config.print_level >= PrintLevel::DEBUG) 
                     std::cout << "      [DEBUG] Step rejected. Attempting SOC.\n";
                
                auto soc_data = std::make_unique<TrajArray>();
                solve_soc(*soc_data, active, candidate, N, mu, reg, config.inertia_strategy, linear_solver, config);
                
                for(int k=0; k<=N; ++k) {
                    active[k].dx += (*soc_data)[k].dx;
                    active[k].du += (*soc_data)[k].du;
                    active[k].ds += (*soc_data)[k].ds;
                    active[k].dlam += (*soc_data)[k].dlam;
                }
                soc_attempted = true;
                continue;
            }

            if (accepted) break;
            alpha *= config.line_search_backtrack_factor; // Use Config
            ls_iter++;
        }
        
        if (accepted) {
            trajectory.swap();
            filter.push_back({theta_0, phi_0}); 
        }
        
        return accepted ? alpha : 0.0; 
    }
};

}
