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

    inline ApiStatus validate_half_open_unit_interval_config_value(double value)
    {
        if (!finite_config_value(value)) {
            return ApiStatus::NonFiniteValue;
        }
        return (value >= 0.0 && value < 1.0) ? ApiStatus::OK : ApiStatus::InvalidArgument;
    }

    inline ApiStatus validate_open_unit_interval_config_value(double value)
    {
        if (!finite_config_value(value)) {
            return ApiStatus::NonFiniteValue;
        }
        return (value > 0.0 && value < 1.0) ? ApiStatus::OK : ApiStatus::InvalidArgument;
    }

    inline bool valid_backend(Backend value)
    {
        switch (value) {
        case Backend::CPU_SERIAL:
        case Backend::GPU_MPX:
        case Backend::GPU_PCR:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_initialization_mode(InitializationMode value)
    {
        switch (value) {
        case InitializationMode::COLD_START:
        case InitializationMode::REUSE_PRIMAL:
        case InitializationMode::REUSE_PRIMAL_DUAL:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_warm_start_barrier_mode(WarmStartBarrierMode value)
    {
        switch (value) {
        case WarmStartBarrierMode::RESET_TO_MU_INIT:
        case WarmStartBarrierMode::REUSE_PREVIOUS_MU:
        case WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_warm_start_regularization_mode(WarmStartRegularizationMode value)
    {
        switch (value) {
        case WarmStartRegularizationMode::RESET_TO_REG_INIT:
        case WarmStartRegularizationMode::REUSE_PREVIOUS_REG:
        case WarmStartRegularizationMode::DECAY_PREVIOUS_REG:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_termination_profile(TerminationProfile value)
    {
        switch (value) {
        case TerminationProfile::STRICT_KKT:
        case TerminationProfile::ACCEPTABLE_NMPC:
        case TerminationProfile::RTI_FIXED_ITERATION:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_constraint_scaling_method(ConstraintScalingMethod value)
    {
        switch (value) {
        case ConstraintScalingMethod::NONE:
        case ConstraintScalingMethod::ROW_INF_NORM:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_objective_scaling_method(ObjectiveScalingMethod value)
    {
        switch (value) {
        case ObjectiveScalingMethod::NONE:
        case ObjectiveScalingMethod::HESSIAN_GERSHGORIN:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_problem_scaling_method(ProblemScalingMethod value)
    {
        switch (value) {
        case ProblemScalingMethod::NONE:
        case ProblemScalingMethod::RUIZ_EQUILIBRATION:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_integrator_type(IntegratorType value)
    {
        switch (value) {
        case IntegratorType::EULER_EXPLICIT:
        case IntegratorType::EULER_IMPLICIT:
        case IntegratorType::RK2_EXPLICIT:
        case IntegratorType::RK2_IMPLICIT:
        case IntegratorType::RK4_EXPLICIT:
        case IntegratorType::RK4_IMPLICIT:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_barrier_strategy(BarrierStrategy value)
    {
        switch (value) {
        case BarrierStrategy::MONOTONE:
        case BarrierStrategy::ADAPTIVE:
        case BarrierStrategy::MEHROTRA:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_inertia_strategy(InertiaStrategy value)
    {
        switch (value) {
        case InertiaStrategy::REGULARIZATION:
        case InertiaStrategy::SATURATION:
        case InertiaStrategy::IGNORE_SINGULAR:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_line_search_type(LineSearchType value)
    {
        switch (value) {
        case LineSearchType::MERIT:
        case LineSearchType::FILTER:
        case LineSearchType::NONE:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_print_level(PrintLevel value)
    {
        switch (value) {
        case PrintLevel::NONE:
        case PrintLevel::WARN:
        case PrintLevel::INFO:
        case PrintLevel::ITER:
        case PrintLevel::DEBUG:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_hessian_approximation(HessianApproximation value)
    {
        switch (value) {
        case HessianApproximation::EXACT:
        case HessianApproximation::OBJECTIVE_HESSIAN_ONLY:
            return true;
        default:
            return false;
        }
    }

    inline bool valid_direction_refinement_mode(DirectionRefinementMode value)
    {
        switch (value) {
        case DirectionRefinementMode::NONE:
        case DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT:
            return true;
        default:
            return false;
        }
    }

    inline bool validate_config_enums(const SolverConfig& conf)
    {
        return valid_backend(conf.backend) && valid_initialization_mode(conf.initialization)
            && valid_warm_start_barrier_mode(conf.warm_start_barrier)
            && valid_warm_start_regularization_mode(conf.warm_start_regularization)
            && valid_termination_profile(conf.termination_profile)
            && valid_constraint_scaling_method(conf.constraint_scaling)
            && valid_objective_scaling_method(conf.objective_scaling)
            && valid_problem_scaling_method(conf.problem_scaling)
            && valid_integrator_type(conf.integrator)
            && valid_barrier_strategy(conf.barrier_strategy)
            && valid_inertia_strategy(conf.inertia_strategy)
            && valid_line_search_type(conf.line_search_type) && valid_print_level(conf.print_level)
            && valid_hessian_approximation(conf.hessian_approximation)
            && valid_direction_refinement_mode(conf.direction_refinement);
    }

    inline ApiStatus validate_solver_config(const SolverConfig& conf)
    {
        if (!validate_config_enums(conf)) {
            return ApiStatus::InvalidArgument;
        }
        if (conf.max_iters < 0 || conf.line_search_max_iters < 0 || conf.max_restoration_iters < 0
            || conf.inertia_max_retries < 0 || conf.newton_config.max_iters < 0) {
            return ApiStatus::InvalidArgument;
        }
        if (conf.line_search_type != LineSearchType::NONE && conf.line_search_max_iters <= 0) {
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
            conf.restoration_alpha, conf.restoration_sufficient_decrease_factor,
            conf.constraint_row_scale_min, conf.constraint_row_scale_max, conf.objective_scale_min,
            conf.objective_scale_max, conf.newton_config.tol, conf.newton_config.regularization };
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
        if (conf.reg_scale_up <= 1.0) {
            return ApiStatus::InvalidArgument;
        }
        status = validate_positive_finite_config_value(conf.reg_scale_down);
        if (status != ApiStatus::OK) {
            return status;
        }
        if (conf.reg_scale_down <= 1.0) {
            return ApiStatus::InvalidArgument;
        }
        status = validate_positive_finite_config_value(conf.min_barrier_slack);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_positive_finite_config_value(conf.warm_start_slack_init);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_open_unit_interval_config_value(conf.line_search_tau);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_open_unit_interval_config_value(conf.line_search_backtrack_factor);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_unit_interval_config_value(conf.restoration_alpha);
        if (status != ApiStatus::OK) {
            return status;
        }
        status = validate_half_open_unit_interval_config_value(
            conf.restoration_sufficient_decrease_factor);
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
