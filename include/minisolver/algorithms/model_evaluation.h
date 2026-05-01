#pragma once

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/integrator/implicit_integrator.h"

#include <type_traits>
#include <utility>

namespace minisolver {
namespace detail {

    template <typename Model, typename Knot, typename = void>
    struct has_compute_terminal_cost_gn : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_terminal_cost_gn<Model, Knot,
        std::void_t<decltype(Model::template compute_terminal_cost_gn<double>(
            std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot, typename = void>
    struct has_compute_terminal_cost_exact : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_terminal_cost_exact<Model, Knot,
        std::void_t<decltype(Model::template compute_terminal_cost_exact<double>(
            std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot, typename = void>
    struct has_compute_terminal_constraints : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_terminal_constraints<Model, Knot,
        std::void_t<decltype(Model::template compute_terminal_constraints<double>(
            std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot>
    void evaluate_model_stage(
        Knot& kp, const SolverConfig& config, double dt, bool is_terminal = false)
    {
        if (config.hessian_approximation == HessianApproximation::OBJECTIVE_HESSIAN_ONLY) {
            if constexpr (has_compute_terminal_cost_gn<Model, Knot>::value) {
                if (is_terminal) {
                    Model::template compute_terminal_cost_gn<double>(kp);
                } else {
                    Model::template compute_cost_gn<double>(kp);
                }
            } else {
                Model::template compute_cost_gn<double>(kp);
            }
        } else {
            if constexpr (has_compute_terminal_cost_exact<Model, Knot>::value) {
                if (is_terminal) {
                    Model::template compute_terminal_cost_exact<double>(kp);
                } else {
                    Model::template compute_cost_exact<double>(kp);
                }
            } else {
                Model::template compute_cost_exact<double>(kp);
            }
        }

        detail::dispatch_compute_dynamics<Model>(kp, config.integrator, dt, config.newton_config);
        if constexpr (has_compute_terminal_constraints<Model, Knot>::value) {
            if (is_terminal) {
                Model::template compute_terminal_constraints<double>(kp);
            } else {
                Model::template compute_constraints<double>(kp);
            }
        } else {
            Model::template compute_constraints<double>(kp);
        }
    }

} // namespace detail
} // namespace minisolver
