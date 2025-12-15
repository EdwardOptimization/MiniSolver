#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/trajectory.h"
#include "minisolver/core/logger.h" // [NEW] Needed for MLOG_DEBUG
#include "minisolver/algorithms/linear_solver.h"
#include "minisolver/solver/line_search_utils.h" // [NEW] Needed for fraction_to_boundary_rule

namespace minisolver {

template<typename Model, int MAX_N>
class LineSearchStrategy {
public:
    static const int NX = Model::NX;
    static const int NU = Model::NU;
    static const int NC = Model::NC;
    static const int NP = Model::NP;

    using Knot = KnotPointV2<double, NX, NU, NC, NP>;
    using TrajectoryType = Trajectory<Knot, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    virtual ~LineSearchStrategy() = default;

    // Pure virtual interface - accepts any LinearSolver via template 
    template<typename LS>
    double search_impl(TrajectoryType& trajectory, 
                       LS& linear_solver,
                       const std::array<double, MAX_N>& dt_traj,
                       double mu, double reg,
                       const SolverConfig& config) {
        return search(trajectory, linear_solver, dt_traj, mu, reg, config);
    }
    
    virtual double search(TrajectoryType& trajectory, 
                          LinearSolver<TrajectoryType>& linear_solver,
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

    double compute_merit(const TrajectoryType& traj, int N, double mu, const SolverConfig& config) {
        double total_merit = 0.0;
        const int NC = Model::NC;
        const int NX = Model::NX;
        
        auto* state = traj.get_active_state();
        auto* model = traj.get_model_data();
        
        for(int k=0; k<=N; ++k) {
            total_merit += state[k].cost; 
            
            // Barrier & Soft Constraint Penalty Calculation
            for(int i=0; i<NC; ++i) {
                if(state[k].s(i) > config.min_barrier_slack) 
                    total_merit -= mu * std::log(state[k].s(i));
                else 
                    total_merit += config.barrier_inf_cost; 
                
                // L1 Soft Constraint: Dual Barrier
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                // L1 Soft Constraint: Dual Barrier + Linear Penalty
                if (type == 1 && w > 1e-6) {
                    // 1. Barrier terms
                    if (state[k].soft_s(i) > config.min_barrier_slack)
                        total_merit -= mu * std::log(state[k].soft_s(i));
                    else
                        total_merit += config.barrier_inf_cost;

                    if (w - state[k].lam(i) > 1e-9)
                        total_merit -= mu * std::log(w - state[k].lam(i));
                    else 
                        total_merit += config.barrier_inf_cost;
                    
                    // 2. L1 Linear Penalty
                    total_merit += w * state[k].soft_s(i); 
                }
                
                // L2 Soft Constraint: Quadratic Penalty
                else if (type == 2 && w > 1e-6) {
                    // L2 Quadratic Penalty: 0.5 * w * (g + s)^2
                    double viol = state[k].g_val(i) + state[k].s(i);
                    total_merit += 0.5 * w * viol * viol;
                }
            }
            
            // Inequality Violation
            for(int i=0; i<NC; ++i) {
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                     if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                     }
                }

                if (type == 1 && w > 1e-6) {
                    total_merit += merit_nu * std::abs(state[k].g_val(i) + state[k].s(i) - state[k].soft_s(i));
                }
                else if (type == 2 && w > 1e-6) {
                    // L2 Soft: No hard violation penalty (handled in Cost)
                }
                else {
                    total_merit += merit_nu * std::abs(state[k].g_val(i) + state[k].s(i));
                }
            }
            
            // Dynamic Defect Violation (Multiple Shooting)
            if (k < N) {
                MSVec<double, NX> defect = state[k+1].x - model[k].f_resid;
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
                  LinearSolver<TrajectoryType>& /*linear_solver*/,
                  const std::array<double, MAX_N>& dt_traj,
                  double mu, double /*reg*/,
                  const SolverConfig& config) override 
    {
        int N = trajectory.N;
        auto* active_state = trajectory.get_active_state();

        auto* active_workspace = trajectory.get_workspace();
        
        // 1. Update Nu
        double max_dual = 0.0;
        for(int k=0; k<=N; ++k) {
            double local_max = MatOps::norm_inf(active_state[k].lam);
            if(local_max > max_dual) max_dual = local_max;
        }
        double required_nu = max_dual * 1.1 + 1.0; 
        if (required_nu > merit_nu) merit_nu = required_nu;

        // 2. Initial Merit
        double phi_0 = compute_merit(trajectory, N, mu, config);
        
        // 3. Calc max alpha
        double alpha = fraction_to_boundary_rule_split<TrajectoryType, Model>(trajectory, N, config.line_search_tau);

        trajectory.prepare_candidate();
        auto* candidate_state = trajectory.get_candidate_state();

        auto* candidate_model = trajectory.get_candidate_model();
        
        bool accepted = false;
        int ls_iter = 0;
        
        while (ls_iter < config.line_search_max_iters) {
            // Update
            for(int k=0; k<=N; ++k) {
                candidate_state[k].x = active_state[k].x + alpha * active_workspace[k].dx;
                candidate_state[k].u = active_state[k].u + alpha * active_workspace[k].du;
                candidate_state[k].s = active_state[k].s + alpha * active_workspace[k].ds;
                candidate_state[k].lam = active_state[k].lam + alpha * active_workspace[k].dlam;
                
                // Update soft vars
                candidate_state[k].soft_s = active_state[k].soft_s + alpha * active_workspace[k].dsoft_s;
                
                candidate_state[k].p = active_state[k].p; 
                
                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                
                // Optional Rollout (Single Shooting)
                if (config.enable_line_search_rollout && k < N) {
                    if (k==0) candidate_state[0].x = active_state[0].x; 
                    candidate_state[k+1].x = Model::integrate(candidate_state[k].x, candidate_state[k].u, candidate_state[k].p, current_dt, config.integrator);
                }

                // Compute Residuals
                if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                    Model::compute_cost_gn(candidate_state[k], candidate_model[k]);
                    Model::compute_dynamics(candidate_state[k], candidate_model[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate_state[k], candidate_model[k]);
                } else {
                    Model::compute_cost_exact(candidate_state[k], candidate_model[k]);
                    Model::compute_dynamics(candidate_state[k], candidate_model[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate_state[k], candidate_model[k]);
                }
            }
            
            double phi_alpha = compute_merit(trajectory, N, mu, config);
            
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

    std::pair<double, double> compute_metrics(const TrajectoryType& traj, int N, double mu, const SolverConfig& config, bool use_candidate = false) {
        double theta = 0.0; 
        double phi = 0.0;   
        const int NC = Model::NC;
        const int NX = Model::NX;
        
        auto* state = use_candidate ? traj.get_candidate_state() : traj.get_active_state();
        auto* model = traj.get_model_data();
        
        for(int k=0; k<=N; ++k) {
            
            // Objective (Phi) Calculation
            phi += state[k].cost;
            for(int i=0; i<NC; ++i) {
                if(state[k].s(i) > config.min_barrier_slack) phi -= mu * std::log(state[k].s(i));
                else phi += config.barrier_inf_cost;
                
                // L1 Soft Constraint: Dual Barrier
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }

                // L1 Soft Constraint
                if (type == 1 && w > 1e-6) {
                    // Barrier terms
                    if (state[k].soft_s(i) > config.min_barrier_slack)
                        phi -= mu * std::log(state[k].soft_s(i));
                    else
                        phi += config.barrier_inf_cost;

                    if (w - state[k].lam(i) > 1e-9)
                        phi -= mu * std::log(w - state[k].lam(i));
                    else 
                        phi += config.barrier_inf_cost;

                    // L1 Linear Penalty
                    phi += w * state[k].soft_s(i);
                }
                
                // L2 Soft Constraint
                else if (type == 2 && w > 1e-6) {
                    // L2 Quadratic Penalty: 0.5 * w * (g + s)^2
                    double viol = state[k].g_val(i) + state[k].s(i);
                    phi += 0.5 * w * viol * viol;
                }
            }
            
            // Infeasibility (Theta)
            for(int i=0; i<NC; ++i) {
                // Correct residual for L1/L2
                double w = 0.0;
                int type = 0;
                if constexpr (NC > 0) {
                    if (static_cast<size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }
                
                if (type == 1 && w > 1e-6) {
                    // L1: Check extended system residual
                    theta += std::abs(state[k].g_val(i) + state[k].s(i) - state[k].soft_s(i));
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
                    theta += std::abs(state[k].g_val(i) + state[k].s(i));
                }
            }
            
            // Dynamic Defect
            if (k < N) {
                MSVec<double, NX> defect = state[k+1].x - model[k].f_resid;
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
                   LinearSolver<TrajectoryType>& solver, const SolverConfig& config) {
        ...
    }
    */

public:
    void reset() override {
        filter.clear();
    }
    
    double search(TrajectoryType& trajectory, 
                  LinearSolver<TrajectoryType>& linear_solver,
                  const std::array<double, MAX_N>& dt_traj,
                  double mu, double reg,
                  const SolverConfig& config) override 
    {
        int N = trajectory.N;
        auto* active_state = trajectory.get_active_state();

        auto* active_workspace = trajectory.get_workspace();
        
        auto m_0 = compute_metrics(trajectory, N, mu, config);
        double theta_0 = m_0.first;
        double phi_0 = m_0.second;
        
        // Fraction to Boundary
        double alpha = fraction_to_boundary_rule_split<TrajectoryType, Model>(trajectory, N, config.line_search_tau);
        
        trajectory.prepare_candidate();
        auto* candidate_state = trajectory.get_candidate_state();

        auto* candidate_model = trajectory.get_candidate_model();
        
        bool accepted = false;
        int ls_iter = 0;
        bool soc_attempted = false;
        
        while (ls_iter < config.line_search_max_iters) {
            for(int k=0; k<=N; ++k) {
                candidate_state[k].x = active_state[k].x + alpha * active_workspace[k].dx;
                candidate_state[k].u = active_state[k].u + alpha * active_workspace[k].du;
                candidate_state[k].s = active_state[k].s + alpha * active_workspace[k].ds;
                candidate_state[k].lam = active_state[k].lam + alpha * active_workspace[k].dlam;
                candidate_state[k].soft_s = active_state[k].soft_s + alpha * active_workspace[k].dsoft_s;
                candidate_state[k].p = active_state[k].p; 
                
                double current_dt = (k < N) ? dt_traj[k] : 0.0;
                
                // Optional Rollout
                if (config.enable_line_search_rollout && k < N) {
                    if (k==0) candidate_state[0].x = active_state[0].x;
                    candidate_state[k+1].x = Model::integrate(candidate_state[k].x, candidate_state[k].u, candidate_state[k].p, current_dt, config.integrator);
                }
                
                // Compute Residuals
                if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                    Model::compute_cost_gn(candidate_state[k], candidate_model[k]);
                    Model::compute_dynamics(candidate_state[k], candidate_model[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate_state[k], candidate_model[k]);
                } else {
                    Model::compute_cost_exact(candidate_state[k], candidate_model[k]);
                    Model::compute_dynamics(candidate_state[k], candidate_model[k], config.integrator, current_dt);
                    Model::compute_constraints(candidate_state[k], candidate_model[k]);
                }
            }
            
            auto m_alpha = compute_metrics(trajectory, N, mu, config, true); // Use candidate state
            
            if (is_acceptable(m_alpha.first, m_alpha.second, theta_0, phi_0, config)) {
                accepted = true;
            }
            
            // SOC Logic (RESTORED for Split Architecture)
            if (!accepted && config.enable_soc && !soc_attempted && ls_iter == 0 && alpha > config.soc_trigger_alpha) {
                if (config.print_level >= PrintLevel::DEBUG) 
                     MLOG_DEBUG("Step rejected. Attempting SOC.");
                
                // In new architecture, solve_soc modifies workspace in-place with SOC correction
                // Pass current candidate trajectory as the RHS (constraint residuals source)
                bool soc_success = linear_solver.solve_soc(trajectory, trajectory, N, mu, reg, config.inertia_strategy, config);
                
                if (soc_success) {
                    // Apply SOC correction to candidate state
                    // The workspace now contains dx_soc, du_soc, etc.
                    auto* soc_workspace = trajectory.get_workspace();
                    
                    for(int k=0; k<=N; ++k) {
                        // Apply correction: x_new = x_candidate + dx_soc
                        candidate_state[k].x += soc_workspace[k].dx;
                        candidate_state[k].u += soc_workspace[k].du;
                        candidate_state[k].s += soc_workspace[k].ds;
                        candidate_state[k].lam += soc_workspace[k].dlam;
                        candidate_state[k].soft_s += soc_workspace[k].dsoft_s;
                        
                        // Re-evaluate dynamics/constraints for SOC candidate
                        double current_dt = (k < N) ? dt_traj[k] : 0.0;
                        if (config.enable_line_search_rollout && k < N) {
                             // SOC modifies the trial point, so we need to re-integrate
                             if (k==0) {
                                 candidate_state[0].x = active_state[0].x; // Base is fixed
                             } else {
                                 candidate_state[k].x = Model::integrate(candidate_state[k-1].x, candidate_state[k-1].u, 
                                                                         candidate_state[k-1].p, dt_traj[k-1], config.integrator);
                             }
                        }
                        
                        // Recompute residuals for SOC-corrected candidate
                        if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
                            Model::compute_cost_gn(candidate_state[k], candidate_model[k]);
                            Model::compute_dynamics(candidate_state[k], candidate_model[k], config.integrator, current_dt);
                            Model::compute_constraints(candidate_state[k], candidate_model[k]);
                        } else {
                            Model::compute_cost_exact(candidate_state[k], candidate_model[k]);
                            Model::compute_dynamics(candidate_state[k], candidate_model[k], config.integrator, current_dt);
                            Model::compute_constraints(candidate_state[k], candidate_model[k]);
                        }
                    }
                    
                    // Check if SOC-corrected candidate is acceptable
                    auto m_soc = compute_metrics(trajectory, N, mu, config, true); // Use candidate state
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
