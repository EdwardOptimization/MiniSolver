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

    template <typename Enum, Enum... Values> inline bool enum_is_one_of(Enum value)
    {
        return ((value == Values) || ...);
    }

    inline bool valid_enum(Backend value)
    {
        return enum_is_one_of<Backend, Backend::CPU_SERIAL, Backend::GPU_MPX, Backend::GPU_PCR>(
            value);
    }

    inline bool valid_enum(InitializationMode value)
    {
        return enum_is_one_of<InitializationMode, InitializationMode::COLD_START,
            InitializationMode::REUSE_PRIMAL, InitializationMode::REUSE_PRIMAL_DUAL>(value);
    }

    inline bool valid_enum(WarmStartBarrierMode value)
    {
        return enum_is_one_of<WarmStartBarrierMode, WarmStartBarrierMode::RESET_TO_MU_INIT,
            WarmStartBarrierMode::REUSE_PREVIOUS_MU,
            WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP>(value);
    }

    inline bool valid_enum(WarmStartRegularizationMode value)
    {
        return enum_is_one_of<WarmStartRegularizationMode,
            WarmStartRegularizationMode::RESET_TO_REG_INIT,
            WarmStartRegularizationMode::REUSE_PREVIOUS_REG,
            WarmStartRegularizationMode::DECAY_PREVIOUS_REG>(value);
    }

    inline bool valid_enum(TerminationProfile value)
    {
        return enum_is_one_of<TerminationProfile, TerminationProfile::STRICT_KKT,
            TerminationProfile::ACCEPTABLE_NMPC, TerminationProfile::RTI_FIXED_ITERATION>(value);
    }

    inline bool valid_enum(ConstraintScalingMethod value)
    {
        return enum_is_one_of<ConstraintScalingMethod, ConstraintScalingMethod::NONE,
            ConstraintScalingMethod::ROW_INF_NORM>(value);
    }

    inline bool valid_enum(ObjectiveScalingMethod value)
    {
        return enum_is_one_of<ObjectiveScalingMethod, ObjectiveScalingMethod::NONE,
            ObjectiveScalingMethod::HESSIAN_GERSHGORIN>(value);
    }

    inline bool valid_enum(ProblemScalingMethod value)
    {
        return enum_is_one_of<ProblemScalingMethod, ProblemScalingMethod::NONE,
            ProblemScalingMethod::RUIZ_EQUILIBRATION>(value);
    }

    inline bool valid_enum(IntegratorType value)
    {
        return enum_is_one_of<IntegratorType, IntegratorType::EULER_EXPLICIT,
            IntegratorType::EULER_IMPLICIT, IntegratorType::RK2_EXPLICIT,
            IntegratorType::RK2_IMPLICIT, IntegratorType::RK4_EXPLICIT,
            IntegratorType::RK4_IMPLICIT>(value);
    }

    inline bool valid_enum(BarrierStrategy value)
    {
        return enum_is_one_of<BarrierStrategy, BarrierStrategy::MONOTONE, BarrierStrategy::ADAPTIVE,
            BarrierStrategy::MEHROTRA>(value);
    }

    inline bool valid_enum(InertiaStrategy value)
    {
        return enum_is_one_of<InertiaStrategy, InertiaStrategy::REGULARIZATION,
            InertiaStrategy::SATURATION, InertiaStrategy::IGNORE_SINGULAR>(value);
    }

    inline bool valid_enum(LineSearchType value)
    {
        return enum_is_one_of<LineSearchType, LineSearchType::MERIT, LineSearchType::FILTER,
            LineSearchType::NONE>(value);
    }

    inline bool valid_enum(PrintLevel value)
    {
        return enum_is_one_of<PrintLevel, PrintLevel::NONE, PrintLevel::WARN, PrintLevel::INFO,
            PrintLevel::ITER, PrintLevel::DEBUG>(value);
    }

    inline bool valid_enum(HessianApproximation value)
    {
        return enum_is_one_of<HessianApproximation, HessianApproximation::EXACT,
            HessianApproximation::OBJECTIVE_HESSIAN_ONLY>(value);
    }

    inline bool valid_enum(DirectionRefinementMode value)
    {
        return enum_is_one_of<DirectionRefinementMode, DirectionRefinementMode::NONE,
            DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT>(value);
    }

#define MINISOLVER_CONFIG_ENUM_FIELDS(X)                                                           \
    X(backend)                                                                                     \
    X(initialization)                                                                              \
    X(warm_start_barrier)                                                                          \
    X(warm_start_regularization)                                                                   \
    X(termination_profile)                                                                         \
    X(constraint_scaling)                                                                          \
    X(objective_scaling)                                                                           \
    X(problem_scaling)                                                                             \
    X(integrator)                                                                                  \
    X(barrier_strategy)                                                                            \
    X(inertia_strategy)                                                                            \
    X(line_search_type)                                                                            \
    X(print_level)                                                                                 \
    X(hessian_approximation)                                                                       \
    X(direction_refinement)

    inline ApiStatus validate_config_enums(const SolverConfig& conf)
    {
#define MS_VALIDATE_CONFIG_ENUM(field)                                                             \
    if (!valid_enum(conf.field)) {                                                                 \
        return ApiStatus::InvalidArgument;                                                         \
    }

        MINISOLVER_CONFIG_ENUM_FIELDS(MS_VALIDATE_CONFIG_ENUM)

#undef MS_VALIDATE_CONFIG_ENUM
        return ApiStatus::OK;
    }

    inline ApiStatus validate_solver_config(const SolverConfig& conf)
    {
        ApiStatus status = validate_config_enums(conf);
        if (status != ApiStatus::OK) {
            return status;
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

        status = validate_positive_finite_config_value(conf.mu_init);
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

#undef MINISOLVER_CONFIG_ENUM_FIELDS
