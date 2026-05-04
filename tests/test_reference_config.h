#pragma once

#include "minisolver/core/solver_options.h"

namespace minisolver::test {

inline SolverConfig make_reference_solver_config()
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;

    // Correctness-first baseline: keep the solve route simple and avoid
    // advanced recovery/globalization heuristics.
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::MERIT;
    config.enable_corrector = false;
    config.enable_aggressive_barrier = false;
    config.direction_refinement = DirectionRefinementMode::NONE;
    config.enable_line_search_rollout = false;
    config.enable_slack_reset = false;
    config.enable_feasibility_restoration = false;
    config.enable_soc = false;

    config.max_iters = 200;
    return config;
}

} // namespace minisolver::test
