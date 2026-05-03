#pragma once

#include "minisolver/core/solver_options.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace minisolver::detail {

inline double barrier_floor(const SolverConfig& config)
{
    return std::max(config.min_barrier_slack, std::numeric_limits<double>::epsilon());
}

inline double initial_slack_floor(const SolverConfig& config)
{
    return std::max(config.warm_start_slack_init, barrier_floor(config));
}

inline double l1_soft_dual_floor(double weight, const SolverConfig& config)
{
    return std::min(2.0 * barrier_floor(config), 0.5 * weight);
}

inline double positive_l1_soft_dual_gap(double soft_dual, double weight, const SolverConfig& config)
{
    return std::max(soft_dual, l1_soft_dual_floor(weight, config));
}

inline bool is_l1_soft_constraint(int type, double weight, const SolverConfig& config)
{
    return type == 1 && std::isfinite(weight) && weight > 2.0 * barrier_floor(config);
}

inline bool is_l2_soft_constraint(int type, double weight)
{
    return type == 2 && std::isfinite(weight) && weight > 0.0;
}

inline double coefficient_degeneracy_floor()
{
    return std::sqrt(std::numeric_limits<double>::epsilon());
}

} // namespace minisolver::detail
