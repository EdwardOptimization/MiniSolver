#pragma once

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include <algorithm>
#include <cmath>

namespace minisolver::detail {

struct TerminationSnapshot {
    bool linear_ok = true;
    double primal_inf = 0.0;
    double dual_inf = 0.0;
    double complementarity_inf = 0.0;
    double barrier_centrality_inf = 0.0;
    double mu = 0.0;
};

struct TerminationKernel {
    static bool uses_fixed_iteration_profile(const SolverConfig& config)
    {
        return config.enable_rti
            || config.termination_profile == TerminationProfile::RTI_FIXED_ITERATION;
    }

    static bool check_convergence(const SolverConfig& config, const TerminationSnapshot& snapshot)
    {
        if (!snapshot.linear_ok) {
            return false;
        }

        const bool primal_ok = (snapshot.primal_inf <= config.tol_con);
        const bool dual_ok = (snapshot.dual_inf <= config.tol_dual);
        const bool complementarity_ok = (snapshot.complementarity_inf <= config.tol_mu);

        return primal_ok && dual_ok && complementarity_ok;
    }

    static SolverStatus classify_solution_quality(
        const SolverConfig& config, const TerminationSnapshot& snapshot)
    {
        if (check_convergence(config, snapshot)) {
            return SolverStatus::OPTIMAL;
        }

        const double feasible_bound = config.tol_con * config.feasible_tol_scale;
        if (snapshot.primal_inf <= feasible_bound) {
            return SolverStatus::FEASIBLE;
        }

        return SolverStatus::UNSOLVED;
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
