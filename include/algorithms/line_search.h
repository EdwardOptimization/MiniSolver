#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

#include "core/types.h"
#include "core/solver_options.h"
#include "core/trajectory.h"
#include "core/logger.h" // [NEW] Needed for MLOG_DEBUG
#include "algorithms/linear_solver.h"
#include "solver/line_search_utils.h" // [NEW] Needed for fraction_to_boundary_rule

namespace minisolver {

template<typename Model, int MAX_N>
class LineSearchStrategy {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;

    using TrajectoryType = Trajectory<KnotPoint<double, NX, NU, NC, NP>, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    virtual ~LineSearchStrategy() = default;

    virtual double search(TrajectoryType& trajectory, 
                          LinearSolver<TrajArray>& linear_solver,
                          const std::array<double, MAX_N>& dt_traj,
                          double mu, double reg,
                          const SolverConfig& config) = 0;
                          
    virtual void reset() {}
};

// --- Merit Function Strategy ---
template<typename Model, int MAX_N>
class MeritLineSearch : public LineSearchStrategy<Model, MAX_N> {
    using Base = LineSearchStrategy<Model, MAX_N>;
    using typename Base::TrajectoryType;
    using typename Base::TrajArray;
    
    double merit_nu = 1000.0;

    double compute_merit(const TrajArray& t, int N, double mu, const SolverConfig& config) {
        double total_merit = 0.0;
        const int NC = Model::NC;
        const int NX = Model::NX;
        
        for(int k=0; k<=N; ++k) {
            const auto& kp = t[k];
            total_merit += kp.cost; 
            
            // Barrier
            for(int i=0; i<NC; ++i) {
                if(kp.s(i) > config.min_barrier_slack) 
                    total_merit -= mu * std::log(kp.s(i));
                else 
                    total_merit += config.barrier_inf_cost; 
                
                // L1 Soft Constraint: Dual Barrier
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                     if (i < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                     }
                }
                // For L1, cost += w * s_soft (in kp.cost via objective function? No, in soft slack setup).
                // In extended KKT, s_soft is explicit.
                // We need to add barrier for s_soft >= 0 and w-lam >= 0.
                if (type == 1 && w > 1e-6) {
                    if (kp.soft_s(i) > config.min_barrier_slack)
                        total_merit -= mu * std::log(kp.soft_s(i));
                    else
                        total_merit += config.barrier_inf_cost;

                    if (w - kp.lam(i) > 1e-9)
                        total_merit -= mu * std::log(w - kp.lam(i));
                    else 
                        total_merit += config.barrier_inf_cost;
                }
            }
            
            // Inequality Violation
            for(int i=0; i<NC; ++i) {
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                     if (i < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                     }
                }

                if (type == 1 && w > 1e-6) {
                    total_merit += merit_nu * std::abs(kp.g_val(i) + kp.s(i) - kp.soft_s(i));
                }
                else if (type == 2 && w > 1e-6) {
                    // L2 Soft: No hard violation penalty (handled in Cost)
                }
                else {
                    total_merit += merit_nu * std::abs(kp.g_val(i) + kp.s(i));
                }
            }
            
            // Dynamic Defect Violation (Multiple Shooting)
            if (k < N) {
                MSVec<double, NX> defect = t[k+1].x - kp.f_resid;
                // L1 Norm of defect
                for(int j=0; j<NX; ++j) {
                    total_merit += merit_nu * std::abs(defect(j));
                }
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
            double local_max = MatOps::norm_inf(active[k].lam);
            if(local_max > max_dual) max_dual = local_max;
        }
        double required_nu = max_dual * 1.1 + 1.0; 
        if (required_nu > merit_nu) merit_nu = required_nu;

        // 2. Initial Merit
        double phi_0 = compute_merit(active, N, mu, config);
        
        // 3. Calc max alpha
        double alpha = fraction_to_boundary_rule<TrajArray, Model>(active, N, config.line_search_tau);

        trajectory.prepare_candidate();
        auto& candidate = trajectory.candidate();
        
        bool accepted = false;
        int ls_iter = 0;
        
        while (ls_iter < config.line_search_max_iters) {
            // Update
            for(int k=0; k<=N; ++k) {
                candidate[k].x = active[k].x + alpha * active[k].dx;
                candidate[k].u = active[k].u + alpha * active[k].du;
                candidate[k].s = active[k].s + alpha * active[k].ds;
                candidate[k].lam = active[k].lam + alpha * active[k].dlam;
                
                // Update soft vars
                candidate[k].soft_s = active[k].soft_s + alpha * active[k].dsoft_s;
                
                candidate[k].p = active[k].p; 
                
                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                
                // Optional Rollout (Single Shooting)
                if (config.enable_line_search_rollout && k < N) {
                    if (k==0) candidate[0].x = active[0].x; 
                    candidate[k+1].x = Model::integrate(candidate[k].x, candidate[k].u, candidate[k].p, current_dt, config.integrator);
                }

                // Compute Residuals
                if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                    Model::compute_cost_gn(candidate[k]);
                    Model::compute_dynamics(candidate[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate[k]);
                } else {
                    Model::compute_cost_exact(candidate[k]);
                    Model::compute_dynamics(candidate[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate[k]);
                }
            }
            
            double phi_alpha = compute_merit(candidate, N, mu, config);
            
            // Armijo condition could be added here: phi_alpha < phi_0 - eta * alpha * ...
            // For now, simple decrease.
            if (phi_alpha < phi_0) {
                accepted = true;
            }

            if (accepted) break;
            alpha *= config.line_search_backtrack_factor; 
            ls_iter++;
        }

        if (accepted) {
            trajectory.swap();
        } else {
            return 0.0; // Fail
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
        const int NX = Model::NX;
        
        for(int k=0; k<=N; ++k) {
            const auto& kp = t[k];
            
            // Objective (Phi)
            phi += kp.cost;
            for(int i=0; i<NC; ++i) {
                if(kp.s(i) > config.min_barrier_slack) phi -= mu * std::log(kp.s(i));
                else phi += config.barrier_inf_cost;
                
                // L1 Soft Constraint: Dual Barrier
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                     if (i < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                     }
                }
                if (type == 1 && w > 1e-6) {
                    if (kp.soft_s(i) > config.min_barrier_slack)
                        phi -= mu * std::log(kp.soft_s(i));
                    else
                        phi += config.barrier_inf_cost;

                    if (w - kp.lam(i) > 1e-9)
                        phi -= mu * std::log(w - kp.lam(i));
                    else 
                        phi += config.barrier_inf_cost;
                }
            }
            
            // Infeasibility (Theta)
            for(int i=0; i<NC; ++i) {
                // Correct residual for L1/L2
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                     if (i < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                     }
                }
                
                if (type == 1 && w > 1e-6) {
                    // L1: Check extended system residual
                    theta += std::abs(kp.g_val(i) + kp.s(i) - kp.soft_s(i));
                } 
                else if (type == 2 && w > 1e-6) {
                    // L2: Soft constraint means no hard infeasibility.
                    // The penalty is in the objective (Phi).
                    // However, we still have the equality g + s - lam/w = 0 in KKT?
                    // No, primal form is unconstrained (penalty).
                    // Ideally theta contribution is 0.
                    // But to keep 's' consistent with 'g', we might want to check g+s?
                    // If we treat it as unconstrained, theta=0 for this index.
                    // Check if Model added penalty to Cost. Yes.
                }
                else {
                    // Hard
                    theta += std::abs(kp.g_val(i) + kp.s(i));
                }
            }
            
            // Dynamic Defect
            if (k < N) {
                MSVec<double, NX> defect = t[k+1].x - kp.f_resid;
                for(int j=0; j<NX; ++j) {
                    theta += std::abs(defect(j));
                }
            }
        }
        return {theta, phi};
    }
    
    bool is_acceptable(double theta, double phi, double theta_0, double phi_0, const SolverConfig& config) {
        // Check against current point (Sufficient Decrease)
        // Condition: theta <= (1-gamma)*theta_0 OR phi <= phi_0 - gamma*theta_0
        bool sufficient_decrease = (theta <= (1.0 - config.filter_gamma_theta) * theta_0) ||
                                   (phi <= phi_0 - config.filter_gamma_phi * theta_0);
        
        if (!sufficient_decrease) return false;

        // Check against filter
        for(const auto& entry : filter) {
            double theta_j = entry.first;
            double phi_j = entry.second;
            bool sufficient_wrt_filter = (theta <= (1.0 - config.filter_gamma_theta) * theta_j) ||
                                         (phi <= phi_j - config.filter_gamma_phi * theta_j);
            if (!sufficient_wrt_filter) return false; 
        }
        return true;
    }
    
    // Helper removed, using linear_solver.solve_soc directly
    /*
    void solve_soc(TrajArray& soc_traj, const TrajArray& active_traj, const TrajArray& trial_traj, 
                   int N, double mu, double reg, InertiaStrategy inertia,
                   LinearSolver<TrajArray>& solver, const SolverConfig& config) {
        ...
    }
    */

public:
    void reset() override {
        filter.clear();
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
        
        // Fraction to Boundary
        double alpha = fraction_to_boundary_rule<TrajArray, Model>(active, N, config.line_search_tau);
        
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
                candidate[k].soft_s = active[k].soft_s + alpha * active[k].dsoft_s;
                candidate[k].p = active[k].p; 
                
                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                
                // Optional Rollout
                if (config.enable_line_search_rollout && k < N) {
                    if (k==0) candidate[0].x = active[0].x;
                    candidate[k+1].x = Model::integrate(candidate[k].x, candidate[k].u, candidate[k].p, current_dt, config.integrator);
                }
                
                // Compute Residuals
                if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                    Model::compute_cost_gn(candidate[k]);
                    Model::compute_dynamics(candidate[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate[k]);
                } else {
                    Model::compute_cost_exact(candidate[k]);
                    Model::compute_dynamics(candidate[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate[k]);
                }
            }
            
            auto m_alpha = compute_metrics(candidate, N, mu, config);
            if (is_acceptable(m_alpha.first, m_alpha.second, theta_0, phi_0, config)) {
                accepted = true;
            }
            
            // SOC Logic
            if (!accepted && config.enable_soc && !soc_attempted && ls_iter == 0 && alpha > config.soc_trigger_alpha) {
                if (config.print_level >= PrintLevel::DEBUG) 
                     MLOG_DEBUG("Step rejected. Attempting SOC.");
                
                auto soc_data = std::make_unique<TrajArray>();
                // [NEW] Use solve_soc
                // Note: solve_soc takes 'soc_rhs_traj'. We want to pass the current trial point 'candidate' as RHS source.
                // soc_data will store the correction step.
                // The linear_solver needs to solve J * dx_soc = -g(candidate)
                
                bool soc_success = linear_solver.solve_soc(*soc_data, candidate, N, mu, reg, config.inertia_strategy, config);
                
                if (soc_success) {
                    for(int k=0; k<=N; ++k) {
                        // Apply correction: x_new = x_candidate + dx_soc
                        // But wait, we need to re-evaluate merit for this new point in next iteration?
                        // Actually, standard SOC applies correction and tries to accept immediately.
                        // Or we update 'candidate' and let the loop continue with same alpha?
                        // If we update candidate, we are effectively testing alpha=1.0 step + soc.
                        
                        // Let's modify candidate directly and re-check acceptability.
                        candidate[k].x += (*soc_data)[k].dx;
                        candidate[k].u += (*soc_data)[k].du;
                        candidate[k].s += (*soc_data)[k].ds;
                        candidate[k].lam += (*soc_data)[k].dlam;
                        
                        // Re-evaluate dynamics/constraints for SOC candidate
                        double current_dt = (k < N) ? dt_traj[k] : 0.0;
                        if (config.enable_line_search_rollout && k < N) {
                             if (k==0) candidate[0].x = active[0].x; // Base doesn't change? Wait, SOC modifies base? No.
                             // SOC modifies the trial point. 
                             // x_trial = x_k + alpha*dx + dx_soc.
                             // Here candidate is already x_k + alpha*dx.
                             // So we just add dx_soc.
                             // But rollout needs consistent integration.
                             if (k==0) {} // x0 fixed
                             else candidate[k].x = Model::integrate(candidate[k-1].x, candidate[k-1].u, candidate[k-1].p, dt_traj[k-1], config.integrator);
                        }
                        // Compute Residuals
                if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                    Model::compute_cost_gn(candidate[k]);
                    Model::compute_dynamics(candidate[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate[k]);
                } else {
                    Model::compute_cost_exact(candidate[k]);
                    Model::compute_dynamics(candidate[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate[k]);
                }
                    }
                    
                    auto m_soc = compute_metrics(candidate, N, mu, config);
                    if (is_acceptable(m_soc.first, m_soc.second, theta_0, phi_0, config)) {
                        if (config.print_level >= PrintLevel::DEBUG) 
                             MLOG_DEBUG("SOC Accepted.");
                        accepted = true;
                    }
                }
                
                soc_attempted = true;
                if (accepted) break;
            }

            if (accepted) break;
            alpha *= config.line_search_backtrack_factor; 
            ls_iter++;
        }
        
        if (accepted) {
            trajectory.swap();
            filter.push_back({theta_0, phi_0}); 
        } else {
            return 0.0; // Fail
        }
        
        return alpha;
    }
};

}
