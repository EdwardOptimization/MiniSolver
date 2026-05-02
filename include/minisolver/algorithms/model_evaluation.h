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

    template <typename Model, typename Knot, typename = void>
    struct has_compute_qp_constraints : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_qp_constraints<Model, Knot,
        std::void_t<decltype(Model::template compute_qp_constraints<double>(
            std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot, typename = void>
    struct has_compute_terminal_qp_constraints : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_terminal_qp_constraints<Model, Knot,
        std::void_t<decltype(Model::template compute_terminal_qp_constraints<double>(
            std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot, typename = void>
    struct has_compute_true_constraints : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_true_constraints<Model, Knot,
        std::void_t<decltype(Model::template compute_true_constraints<double>(
            std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot, typename = void>
    struct has_compute_terminal_true_constraints : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_terminal_true_constraints<Model, Knot,
        std::void_t<decltype(Model::template compute_terminal_true_constraints<double>(
            std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot, typename = void>
    struct has_compute_soc_constraints : std::false_type { };

    template <typename Model, typename Knot>
    struct has_compute_soc_constraints<Model, Knot,
        std::void_t<decltype(Model::template compute_soc_constraints<double>(
            std::declval<const Knot&>(), std::declval<Knot&>()))>> : std::true_type { };

    template <typename Model, typename Knot>
    void evaluate_true_constraints(Knot& kp, bool is_terminal = false)
    {
        if constexpr (has_compute_terminal_true_constraints<Model, Knot>::value) {
            if (is_terminal) {
                Model::template compute_terminal_true_constraints<double>(kp);
            } else if constexpr (has_compute_true_constraints<Model, Knot>::value) {
                Model::template compute_true_constraints<double>(kp);
            } else {
                kp.g_true = kp.g_val;
            }
        } else if constexpr (has_compute_true_constraints<Model, Knot>::value) {
            Model::template compute_true_constraints<double>(kp);
        } else {
            kp.g_true = kp.g_val;
        }
    }

    template <typename Model, typename Knot>
    void evaluate_soc_constraints(const Knot& active_kp, Knot& trial_kp)
    {
        if constexpr (has_compute_soc_constraints<Model, Knot>::value) {
            Model::template compute_soc_constraints<double>(active_kp, trial_kp);
        }
    }

    template <typename Model, typename Knot> double true_constraint_value(const Knot& kp, int idx)
    {
        if constexpr (has_compute_true_constraints<Model, Knot>::value
            || has_compute_terminal_true_constraints<Model, Knot>::value) {
            return kp.g_true(idx);
        } else {
            return kp.g_val(idx);
        }
    }

    template <typename Model, typename Knot>
    void evaluate_qp_constraints(Knot& kp, bool is_terminal = false)
    {
        if constexpr (has_compute_terminal_qp_constraints<Model, Knot>::value) {
            if (is_terminal) {
                Model::template compute_terminal_qp_constraints<double>(kp);
            } else if constexpr (has_compute_qp_constraints<Model, Knot>::value) {
                Model::template compute_qp_constraints<double>(kp);
            } else {
                Model::template compute_constraints<double>(kp);
            }
        } else if constexpr (has_compute_terminal_constraints<Model, Knot>::value) {
            if (is_terminal) {
                Model::template compute_terminal_constraints<double>(kp);
            } else if constexpr (has_compute_qp_constraints<Model, Knot>::value) {
                Model::template compute_qp_constraints<double>(kp);
            } else {
                Model::template compute_constraints<double>(kp);
            }
        } else if constexpr (has_compute_qp_constraints<Model, Knot>::value) {
            Model::template compute_qp_constraints<double>(kp);
        } else {
            Model::template compute_constraints<double>(kp);
        }
    }

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

        if (is_terminal) {
            kp.f_resid.setZero();
            kp.A.setZero();
            kp.B.setZero();
        } else {
            detail::dispatch_compute_dynamics<Model>(
                kp, config.integrator, dt, config.newton_config);
        }
        evaluate_qp_constraints<Model>(kp, is_terminal);

        evaluate_true_constraints<Model>(kp, is_terminal);
    }

} // namespace detail
} // namespace minisolver
