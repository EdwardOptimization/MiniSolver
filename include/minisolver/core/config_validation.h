#pragma once

#include <cmath>

#include "minisolver/core/model_traits.h"
#include "minisolver/core/types.h"

namespace minisolver {
namespace detail {

    inline bool finite_config_value(double value)
    {
        return std::isfinite(value);
    }

    inline ApiStatus validate_positive_finite_config_value(double value)
    {
        if (!finite_config_value(value)) {
            return ApiStatus::NonFiniteValue;
        }
        return value > 0.0 ? ApiStatus::OK : ApiStatus::InvalidArgument;
    }

    inline ApiStatus validate_nonnegative_finite_config_value(double value)
    {
        if (!finite_config_value(value)) {
            return ApiStatus::NonFiniteValue;
        }
        return value >= 0.0 ? ApiStatus::OK : ApiStatus::InvalidArgument;
    }

    inline ApiStatus validate_unit_interval_config_value(double value)
    {
        if (!finite_config_value(value)) {
            return ApiStatus::NonFiniteValue;
        }
        return (value > 0.0 && value <= 1.0) ? ApiStatus::OK : ApiStatus::InvalidArgument;
    }

    inline ApiStatus validate_solver_config(const SolverConfig& conf)
    {
        if (conf.max_iters < 0 || conf.line_search_max_iters < 0 || conf.max_restoration_iters < 0
            || conf.inertia_max_retries < 0 || conf.newton_config.max_iters < 0) {
            return ApiStatus::InvalidArgument;
        }

        const double finite_values[] = { conf.default_dt, conf.mu_init, conf.mu_final,
            conf.mu_linear_decrease_factor, conf.barrier_tolerance_factor, conf.mu_safety_margin,
            conf.reg_init, conf.reg_min, conf.reg_max, conf.reg_scale_up, conf.reg_scale_down,
            conf.regularization_step, conf.singular_threshold, conf.huge_penalty, conf.tol_con,
            conf.tol_dual, conf.tol_mu, conf.tol_cost, conf.feasible_tol_scale,
            conf.line_search_tau, conf.line_search_backtrack_factor, conf.filter_gamma_theta,
            conf.filter_gamma_phi, conf.filter_theta_max_factor, conf.armijo_c1,
            conf.min_barrier_slack, conf.barrier_inf_cost, conf.slack_reset_trigger,
            conf.warm_start_slack_init, conf.soc_trigger_alpha, conf.merit_nu_init,
            conf.eta_suff_descent, conf.restoration_mu, conf.restoration_reg,
            conf.restoration_alpha, conf.constraint_row_scale_min, conf.constraint_row_scale_max,
            conf.objective_scale_min, conf.objective_scale_max, conf.newton_config.tol,
            conf.newton_config.regularization };
        for (double value : finite_values) {
            if (!finite_config_value(value)) {
                return ApiStatus::NonFiniteValue;
            }
        }

        ApiStatus status = validate_positive_finite_config_value(conf.mu_init);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.mu_final);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.mu_linear_decrease_factor);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_nonnegative_finite_config_value(conf.barrier_tolerance_factor);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.mu_safety_margin);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_nonnegative_finite_config_value(conf.reg_init);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.reg_min);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.reg_max);
        if (status != ApiStatus::OK) {
            return status;
        }
        if (conf.reg_max < conf.reg_min) {
            return ApiStatus::InvalidArgument;
        }
        status = validate_positive_finite_config_value(conf.reg_scale_up);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.reg_scale_down);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.min_barrier_slack);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_unit_interval_config_value(conf.line_search_tau);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_unit_interval_config_value(conf.line_search_backtrack_factor);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_unit_interval_config_value(conf.restoration_alpha);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.newton_config.tol);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_nonnegative_finite_config_value(conf.newton_config.regularization);
        if (status != ApiStatus::OK) {
            return status;
        }

        if (!constraint_scaling_plan_valid(conf) || !objective_scaling_plan_valid(conf)
            || !problem_scaling_plan_valid(conf)) {
            return ApiStatus::InvalidArgument;
        }
        return ApiStatus::OK;
    }

} // namespace detail
} // namespace minisolver
