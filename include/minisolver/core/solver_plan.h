#pragma once

#include "minisolver/core/solver_options.h"

namespace minisolver {

struct SolverPlanInfo {
    Backend backend = Backend::CPU_SERIAL;
    LineSearchType line_search_type = LineSearchType::FILTER;
    IntegratorType integrator = IntegratorType::RK4_EXPLICIT;
    bool fused_riccati_integrator_compatible = true;
    bool linear_solver_ready = false;
    bool line_search_ready = false;
};

struct SolverBuildState {
    bool dirty = true;
    SolverPlanInfo plan;
};

} // namespace minisolver
