#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "minisolver/core/solver_options.h"

namespace minisolver {
namespace detail {

    // Optional marker emitted by MiniModel.py. Fused/generated paths are valid only
    // for the integrator used at code-generation time.
    template <typename, typename = void> struct has_generated_integrator : std::false_type { };

    template <typename Model>
    struct has_generated_integrator<Model, std::void_t<decltype(Model::generated_integrator)>>
        : std::true_type { };

    template <typename Model>
    static constexpr bool has_generated_integrator_v = has_generated_integrator<Model>::value;

    template <typename Model> bool generated_integrator_matches(IntegratorType integrator)
    {
        if constexpr (has_generated_integrator_v<Model>) {
            return Model::generated_integrator == integrator;
        } else {
            return true;
        }
    }

    inline bool constraint_row_scaling_active(const SolverConfig& config)
    {
        return config.constraint_scaling == ConstraintScalingMethod::ROW_INF_NORM
            || config.problem_scaling == ProblemScalingMethod::RUIZ_EQUILIBRATION;
    }

    inline bool constraint_row_scale_bounds_valid(const SolverConfig& config)
    {
        return std::isfinite(config.constraint_row_scale_min)
            && std::isfinite(config.constraint_row_scale_max)
            && config.constraint_row_scale_min > 0.0
            && config.constraint_row_scale_max >= config.constraint_row_scale_min;
    }

    inline bool constraint_scaling_plan_valid(const SolverConfig& config)
    {
        if (config.constraint_scaling == ConstraintScalingMethod::NONE) {
            return true;
        }
        if (config.constraint_scaling == ConstraintScalingMethod::ROW_INF_NORM) {
            return constraint_row_scale_bounds_valid(config);
        }
        return false;
    }

    inline bool objective_scaling_active(const SolverConfig& config)
    {
        return config.objective_scaling == ObjectiveScalingMethod::HESSIAN_GERSHGORIN
            || config.problem_scaling == ProblemScalingMethod::RUIZ_EQUILIBRATION;
    }

    inline bool objective_scale_bounds_valid(const SolverConfig& config)
    {
        return std::isfinite(config.objective_scale_min)
            && std::isfinite(config.objective_scale_max) && config.objective_scale_min > 0.0
            && config.objective_scale_max >= config.objective_scale_min;
    }

    inline bool objective_scaling_method_valid(ObjectiveScalingMethod method)
    {
        return method == ObjectiveScalingMethod::NONE
            || method == ObjectiveScalingMethod::HESSIAN_GERSHGORIN;
    }

    inline bool objective_scaling_plan_valid(const SolverConfig& config)
    {
        return objective_scaling_method_valid(config.objective_scaling)
            && (!objective_scaling_active(config) || objective_scale_bounds_valid(config));
    }

    inline bool problem_scaling_active(const SolverConfig& config)
    {
        return config.problem_scaling == ProblemScalingMethod::RUIZ_EQUILIBRATION;
    }

    inline bool problem_scaling_method_valid(ProblemScalingMethod method)
    {
        return method == ProblemScalingMethod::NONE
            || method == ProblemScalingMethod::RUIZ_EQUILIBRATION;
    }

    inline bool problem_scaling_plan_valid(const SolverConfig& config)
    {
        return problem_scaling_method_valid(config.problem_scaling)
            && (!problem_scaling_active(config)
                || (constraint_row_scale_bounds_valid(config)
                    && objective_scale_bounds_valid(config)));
    }

} // namespace detail
} // namespace minisolver
