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

template <typename Model, typename Knot>
bool active_mixed_l1_l2_soft_constraint(const Knot& kp, int row, const SolverConfig& config)
{
    return active_l1_soft_constraint<Model>(kp, row, config)
        && active_l2_soft_constraint<Model>(kp, row);
}

template <typename Model, typename Knot>
double l1_soft_dual_gap_from_values(const Knot& kp, int row, double lam, double soft_s)
{
    double soft_dual = kp.l1_weight(row) - lam;
    if (active_l2_soft_constraint<Model>(kp, row)) {
        soft_dual += kp.l2_weight(row) * soft_s;
    }
    return soft_dual;
}

template <typename Model, typename Knot> double l1_soft_dual_gap(const Knot& kp, int row)
{
    return l1_soft_dual_gap_from_values<Model>(kp, row, kp.lam(row), kp.soft_s(row));
}

template <typename Model, typename Knot> double l1_soft_dual_direction(const Knot& kp, int row)
{
    double direction = -kp.dlam(row);
    if (active_l2_soft_constraint<Model>(kp, row)) {
        direction += kp.l2_weight(row) * kp.dsoft_s(row);
    }
    return direction;
}

template <typename Model, typename Knot>
double l1_soft_stationarity_denominator(const Knot& kp, int row, double soft_s, double soft_dual)
{
    double denominator = soft_dual;
    if (active_l2_soft_constraint<Model>(kp, row)) {
        denominator += kp.l2_weight(row) * soft_s;
    }
    return denominator;
}

template <typename Model, typename Knot> double l1_soft_penalty_value(const Knot& kp, int row)
{
    const double soft_s = kp.soft_s(row);
    double value = kp.l1_weight(row) * soft_s;
    if (active_l2_soft_constraint<Model>(kp, row)) {
        value += 0.5 * kp.l2_weight(row) * soft_s * soft_s;
    }
    return value;
}

template <typename Model, typename Knot> double l1_soft_penalty_direction(const Knot& kp, int row)
{
    double slope = kp.l1_weight(row);
    if (active_l2_soft_constraint<Model>(kp, row)) {
        slope += kp.l2_weight(row) * kp.soft_s(row);
    }
    return slope * kp.dsoft_s(row);
}

template <typename Model, typename Knot>
double l1_soft_slack_from_dual(
    const Knot& kp, int row, double lam, double mu, const SolverConfig& config)
{
    const double w1 = kp.l1_weight(row);
    const double w2 = active_l2_soft_constraint<Model>(kp, row) ? kp.l2_weight(row) : 0.0;
    if (w2 > 0.0) {
        const double b = w1 - lam;
        const double disc = std::max(0.0, b * b + 4.0 * w2 * mu);
        const double soft_s = (-b + std::sqrt(disc)) / (2.0 * w2);
        return std::max(config.min_barrier_slack, soft_s);
    }
    const double soft_dual = positive_l1_soft_dual_gap(w1 - lam, w1, config);
    return std::max(config.min_barrier_slack, mu / soft_dual);
}

} // namespace minisolver::detail
