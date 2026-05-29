#pragma once

#include "minisolver/core/solver_options.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

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

inline double coefficient_degeneracy_floor()
{
    return std::sqrt(std::numeric_limits<double>::epsilon());
}

template <typename, typename = void> struct has_constraint_has_l1 : std::false_type { };
template <typename Model>
struct has_constraint_has_l1<Model, std::void_t<decltype(Model::constraint_has_l1)>>
    : std::true_type { };

template <typename, typename = void> struct has_constraint_has_l2 : std::false_type { };
template <typename Model>
struct has_constraint_has_l2<Model, std::void_t<decltype(Model::constraint_has_l2)>>
    : std::true_type { };

template <typename Model, typename Knot, typename = void>
struct has_update_soft_constraint_weights : std::false_type { };
template <typename Model, typename Knot>
struct has_update_soft_constraint_weights<Model, Knot,
    std::void_t<decltype(Model::template update_soft_constraint_weights<double>(
        std::declval<Knot&>()))>> : std::true_type { };

template <typename Model> bool constraint_has_l1(int row)
{
    if constexpr (has_constraint_has_l1<Model>::value) {
        return row >= 0 && static_cast<std::size_t>(row) < Model::constraint_has_l1.size()
            && Model::constraint_has_l1[static_cast<std::size_t>(row)];
    }
    return false;
}

template <typename Model> bool constraint_has_l2(int row)
{
    if constexpr (has_constraint_has_l2<Model>::value) {
        return row >= 0 && static_cast<std::size_t>(row) < Model::constraint_has_l2.size()
            && Model::constraint_has_l2[static_cast<std::size_t>(row)];
    }
    return false;
}

template <typename Model, typename Knot> void update_soft_constraint_weights(Knot& kp)
{
    kp.l1_weight.setZero();
    kp.l2_weight.setZero();
    if constexpr (has_update_soft_constraint_weights<Model, Knot>::value) {
        Model::template update_soft_constraint_weights<double>(kp);
    }
}

template <typename Model, typename Knot>
bool active_l1_soft_constraint(const Knot& kp, int row, const SolverConfig& config)
{
    return constraint_has_l1<Model>(row) && std::isfinite(kp.l1_weight(row))
        && kp.l1_weight(row) > 2.0 * barrier_floor(config);
}

template <typename Model, typename Knot> bool active_l2_soft_constraint(const Knot& kp, int row)
{
    return constraint_has_l2<Model>(row) && std::isfinite(kp.l2_weight(row))
        && kp.l2_weight(row) > 0.0;
}

} // namespace minisolver::detail
