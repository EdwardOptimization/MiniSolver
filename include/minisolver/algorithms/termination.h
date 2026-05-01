#pragma once

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include <algorithm>
#include <cmath>

namespace minisolver::detail {

struct TerminationKernel {
    static bool check_convergence(const SolverConfig& config, double mu, double max_primal_inf,
        double max_dual, double max_barrier_complementarity_residual)
    {
        const bool mu_converged = (mu <= config.mu_final);
        const bool primal_ok = (max_primal_inf <= config.tol_con);
        const bool dual_ok = (max_dual <= config.tol_dual);
        const bool kkt_ok
            = (max_barrier_complementarity_residual <= std::max(config.tol_mu, 10.0 * mu));

        return mu_converged && primal_ok && dual_ok && kkt_ok;
    }

    static bool should_stop_for_cost_stagnation(const SolverConfig& config, double last_prim_inf,
        double current_cost, double last_cost, double current_mu, double last_mu)
    {
        const double feasible_bound = config.tol_con * config.feasible_tol_scale;
        if (last_prim_inf > feasible_bound) {
            return false;
        }

        const double cost_diff = std::abs(current_cost - last_cost);
        if (cost_diff >= config.tol_cost) {
            return false;
        }

        const bool mu_decreased = (current_mu < last_mu);
        const bool mu_small = (current_mu <= config.mu_final);
        return mu_small || !mu_decreased;
    }

    static SolverStatus classify_tiny_step_stagnation(
        const SolverConfig& config, double max_prim_inf, double max_dual_inf)
    {
        if (max_prim_inf > config.tol_con) {
            return SolverStatus::UNSOLVED;
        }
        if (max_dual_inf <= config.tol_dual) {
            return SolverStatus::OPTIMAL;
        }
        return SolverStatus::FEASIBLE;
    }
};

} // namespace minisolver::detail
