#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

#include "core/types.h"
#include "core/solver_options.h"
#include "core/trajectory.h"
#include "algorithms/linear_solver.h"

// Reuse the fraction_to_boundary helper
#include "solver/line_search.h" // The old file with helper function

namespace minisolver {

// Forward decl of LinearSolver to avoid circular dependency if needed
// But we include it above.

template<typename Model, int MAX_N>
class LineSearchStrategy {
public:
    using TrajectoryType = Trajectory<KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>, MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    virtual ~LineSearchStrategy() = default;

    // Returns alpha accepted. Modifies trajectory (swaps candidate to active if accepted).
    // Needs access to linear_solver for SOC.
    virtual double search(TrajectoryType& trajectory, 
                          LinearSolver<TrajArray>& linear_solver,
                          const std::array<double, MAX_N>& dt_traj,
                          double mu, double reg,
                          const SolverConfig& config) = 0;
                          
    // Hook for updates at start of step (e.g. update merit penalty)
    virtual void prepare_step(const TrajArray& traj) {}
    
    // Hook for reset (warm start)
    virtual void reset() {}
};

// --- Merit Function Strategy ---
template<typename Model, int MAX_N>
class MeritLineSearch : public LineSearchStrategy<Model, MAX_N> {
    using Base = LineSearchStrategy<Model, MAX_N>;
    using typename Base::TrajectoryType;
    using typename Base::TrajArray;
    
    double merit_nu = 1000.0;

    double compute_merit(const TrajArray& t, double mu) {
        double total_merit = 0.0;
        for(int k=0; k<=Base::TrajectoryType::MAX_N; ++k) { // Use compile-time constant? No, need runtime N.
            // Wait, Trajectory class knows N. But array is fixed size.
            // We need N from trajectory.N
            // Let's pass N or iterate up to trajectory.N
        }
        // Actually, TrajectoryType has size().
        // But we passed TrajectoryType& to search(), here we only have TrajArray& in helper?
        // Let's implement compute_merit inside search or pass N.
        return 0.0;
    }
    
    // Helper with N
    double compute_merit_N(const TrajArray& t, int N, double mu) {
        double total_merit = 0.0;
        const int NC = Model::NC;
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
                total_merit += merit_nu * std::abs(kp.g_val(i) + kp.s(i));
            }
        }
        return total_merit;
    }

public:
    void reset() override {
        merit_nu = 1000.0;
    }

    void prepare_step(const TrajArray& traj) override {
        // Update penalty parameter nu
        double max_dual = 0.0;
        // Need to know N. We assume traj array is valid up to some N?
        // This interface is tricky without N. 
        // Let's just iterate until end? No, MAX_N. 
        // But unused knots might have garbage.
        // We should pass N to prepare_step too, or just do it in search().
        // Let's do it in search() start if needed, but search() is called after solve().
        // Ideally prepare_step is called before solve().
        // Let's change prepare_step signature or move logic.
        // Actually, we can just iterate MAX_N if we ensure unused are zeroed? No.
        // Let's assume the caller handles prepare logic?
        // Or better: update_merit_penalty takes N.
    }
    
    // Revised prepare_step with N? 
    // Let's make prepare_step part of the interface but we need N.
    // Simpler: Just do it inside search()? No, nu updates BEFORE solve usually.
    // But for this refactor, let's keep it simple: update nu at start of search() based on current traj.
    
    double search(TrajectoryType& trajectory, 
                  LinearSolver<TrajArray>& linear_solver,
                  const std::array<double, MAX_N>& dt_traj,
                  double mu, double reg,
                  const SolverConfig& config) override 
    {
        int N = trajectory.N;
        auto& active = trajectory.active();
        
        // 1. Update Nu (Late update, but okay for prototype)
        double max_dual = 0.0;
        for(int k=0; k<=N; ++k) {
            double local_max = active[k].lam.template lpNorm<Eigen::Infinity>();
            if(local_max > max_dual) max_dual = local_max;
        }
        double required_nu = max_dual * 1.1 + 1.0; 
        if (required_nu > merit_nu) merit_nu = required_nu;

        // 2. Initial Merit
        double phi_0 = compute_merit_N(active, N, mu);
        
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
            
            double phi_alpha = compute_merit_N(candidate, N, mu);
            if (phi_alpha < phi_0) {
                accepted = true;
            }

            if (accepted) break;
            alpha *= 0.5;
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

    std::pair<double, double> compute_metrics(const TrajArray& t, int N, double mu) {
        double theta = 0.0; 
        double phi = 0.0;   
        const int NC = Model::NC;
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
    
    bool is_acceptable(double theta, double phi) {
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

public:
    void reset() override {
        filter.clear();
    }
    
    // Helper for SOC (needs access to cpu_serial_solve from linear_solver)
    void solve_soc(TrajArray& soc_traj, const TrajArray& active_traj, const TrajArray& trial_traj, 
                   int N, double mu, double reg, InertiaStrategy inertia,
                   LinearSolver<TrajArray>& solver) {
        // soc_traj initialized with active linearization
        for(int k=0; k<=N; ++k) soc_traj[k] = active_traj[k];
        
        // Modify RHS with trial residuals
        for(int k=0; k<=N; ++k) {
            soc_traj[k].g_val = trial_traj[k].g_val + (trial_traj[k].s - active_traj[k].s);
        }
        
        bool success = solver.solve(soc_traj, N, mu, reg, inertia);
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
        
        // Compute initial metrics
        auto m_0 = compute_metrics(active, N, mu);
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
            
            auto m_alpha = compute_metrics(candidate, N, mu);
            if (is_acceptable(m_alpha.first, m_alpha.second)) {
                accepted = true;
            }
            
            // SOC Logic
            if (!accepted && config.enable_soc && !soc_attempted && ls_iter == 0 && alpha > 0.5) {
                if (config.print_level >= PrintLevel::DEBUG) 
                     std::cout << "      [DEBUG] Step rejected. Attempting SOC.\n";
                
                // Need temp buffer for SOC.
                // We use a local unique_ptr to avoid large stack usage.
                auto soc_data = std::make_unique<TrajArray>();
                solve_soc(*soc_data, active, candidate, N, mu, reg, config.inertia_strategy, linear_solver);
                
                // Add SOC correction to Active direction
                for(int k=0; k<=N; ++k) {
                    active[k].dx += (*soc_data)[k].dx;
                    active[k].du += (*soc_data)[k].du;
                    active[k].ds += (*soc_data)[k].ds;
                    active[k].dlam += (*soc_data)[k].dlam;
                }
                soc_attempted = true;
                // Retry with same alpha (1.0) but new direction
                continue;
            }

            if (accepted) break;
            alpha *= 0.5;
            ls_iter++;
        }
        
        if (accepted) {
            trajectory.swap();
            filter.push_back({theta_0, phi_0}); // Add OLD point to filter
        }
        
        return accepted ? alpha : 0.0; // Return 0.0 if failed
    }
};

}

