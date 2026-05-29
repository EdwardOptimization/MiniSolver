#pragma once

#include "minisolver/core/constraint_semantics.h"
#include "minisolver/core/model_traits.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/integrator/implicit_integrator.h"

#include <algorithm>
#include <cmath>
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
    double unscaled_true_constraint_value(const Knot& kp, int idx, const SolverConfig& config)
    {
        if (constraint_row_scaling_active(config)) {
            return kp.g_unscaled(idx);
        }
        return true_constraint_value<Model>(kp, idx);
    }

    template <typename Knot>
    double active_constraint_row_scale(const Knot& kp, const SolverConfig& config, int row)
    {
        if (!constraint_row_scaling_active(config)) {
            return 1.0;
        }
        return kp.constraint_row_scale(row);
    }

    template <typename Knot>
    double compute_hessian_gershgorin_objective_scale(const Knot& kp, const SolverConfig& config)
    {
        double max_row_sum = 0.0;

        for (int row = 0; row < Knot::NX; ++row) {
            double row_sum = 0.0;
            for (int col = 0; col < Knot::NX; ++col) {
                row_sum += std::abs(kp.Q(row, col));
            }
            for (int col = 0; col < Knot::NU; ++col) {
                row_sum += std::abs(kp.H(col, row));
            }
            max_row_sum = std::max(max_row_sum, row_sum);
        }

        for (int row = 0; row < Knot::NU; ++row) {
            double row_sum = 0.0;
            for (int col = 0; col < Knot::NX; ++col) {
                row_sum += std::abs(kp.H(row, col));
            }
            for (int col = 0; col < Knot::NU; ++col) {
                row_sum += std::abs(kp.R(row, col));
            }
            max_row_sum = std::max(max_row_sum, row_sum);
        }

        if (!std::isfinite(max_row_sum)) {
            return 1.0;
        }

        max_row_sum = std::max(max_row_sum, 1.0);
        const double raw_scale = 1.0 / max_row_sum;
        return std::min(
            config.objective_scale_max, std::max(config.objective_scale_min, raw_scale));
    }

    template <typename Model, typename Knot>
    void apply_objective_scaling(Knot& kp, const SolverConfig& config)
    {
        kp.cost_unscaled = kp.cost;
        kp.objective_scale = 1.0;
        if (!objective_scaling_active(config)) {
            return;
        }

        const double scale = compute_hessian_gershgorin_objective_scale(kp, config);
        kp.objective_scale = scale;

        kp.cost *= scale;
        kp.q = kp.q * scale;
        kp.r = kp.r * scale;
        kp.Q = kp.Q * scale;
        kp.R = kp.R * scale;
        kp.H = kp.H * scale;
    }

    template <typename Knot>
    double compute_auto_constraint_row_scale(const Knot& kp, const SolverConfig& config, int row)
    {
        double row_norm = std::abs(kp.g_unscaled(row));
        for (int j = 0; j < Knot::NX; ++j) {
            row_norm = std::max(row_norm, std::abs(kp.C(row, j)));
        }
        for (int j = 0; j < Knot::NU; ++j) {
            row_norm = std::max(row_norm, std::abs(kp.D(row, j)));
        }
        if (!std::isfinite(row_norm)) {
            return 1.0;
        }

        // First automatic profile only down-scales large rows. Scaling up tiny rows is useful,
        // but more aggressive and should be tied to a later tolerance/normalization pass.
        row_norm = std::max(row_norm, 1.0);
        const double raw_scale = 1.0 / row_norm;
        return std::min(
            config.constraint_row_scale_max, std::max(config.constraint_row_scale_min, raw_scale));
    }

    template <typename Model, typename Knot>
    void apply_constraint_row_scaling(
        Knot& kp, const SolverConfig& config, bool refresh_auto_scaling = false)
    {
        kp.g_unscaled = kp.g_true;
        if (!constraint_row_scaling_active(config)) {
            kp.constraint_row_scale.setOnes();
            return;
        }

        for (int i = 0; i < Knot::NC; ++i) {
            if (refresh_auto_scaling) {
                kp.constraint_row_scale(i) = compute_auto_constraint_row_scale(kp, config, i);
            }
            const double scale = kp.constraint_row_scale(i);
            kp.g_val(i) *= scale;
            kp.g_true(i) *= scale;
            for (int j = 0; j < Knot::NX; ++j) {
                kp.C(i, j) *= scale;
            }
            for (int j = 0; j < Knot::NU; ++j) {
                kp.D(i, j) *= scale;
            }
        }
    }

    template <typename Model, typename Knot>
    void apply_soc_constraint_row_scaling(Knot& kp, const SolverConfig& config)
    {
        if (!constraint_row_scaling_active(config)) {
            return;
        }
        for (int i = 0; i < Knot::NC; ++i) {
            kp.g_val(i) *= active_constraint_row_scale(kp, config, i);
        }
    }

    template <typename Model, typename Knot>
    void evaluate_soc_constraints(const Knot& active_kp, Knot& trial_kp, const SolverConfig& config)
    {
        update_soft_constraint_weights<Model>(trial_kp);
        if constexpr (has_compute_soc_constraints<Model, Knot>::value) {
            Model::template compute_soc_constraints<double>(active_kp, trial_kp);
            apply_soc_constraint_row_scaling<Model>(trial_kp, config);
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
    void evaluate_model_stage(Knot& kp, const SolverConfig& config, double dt,
        bool is_terminal = false, bool refresh_auto_constraint_scaling = false)
    {
        update_soft_constraint_weights<Model>(kp);

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
        apply_objective_scaling<Model>(kp, config);

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
        apply_constraint_row_scaling<Model>(kp, config, refresh_auto_constraint_scaling);
    }

} // namespace detail
} // namespace minisolver
