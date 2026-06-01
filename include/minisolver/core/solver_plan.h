#pragma once

#include "minisolver/core/solver_options.h"

namespace minisolver {

struct SolverPlanInfo {
    Backend backend = Backend::CPU_SERIAL;
    LineSearchType line_search_type = LineSearchType::FILTER;
    IntegratorType integrator = IntegratorType::RUNGE_KUTTA_4;
    ConstraintScalingMethod constraint_scaling = ConstraintScalingMethod::NONE;
    ObjectiveScalingMethod objective_scaling = ObjectiveScalingMethod::NONE;
    ProblemScalingMethod problem_scaling = ProblemScalingMethod::NONE;
    bool fused_riccati_integrator_compatible = true;
    bool constraint_scaling_plan_valid = true;
    bool objective_scaling_plan_valid = true;
    bool problem_scaling_plan_valid = true;
    bool constraint_scaling_active = false;
    bool objective_scaling_active = false;
    bool problem_scaling_active = false;
    bool linear_solver_ready = false;
    bool line_search_ready = false;
};

struct SolverBuildState {
    bool dirty = true;
    SolverPlanInfo plan;
};

} // namespace minisolver
