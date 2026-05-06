#pragma once

#include "minisolver/core/solver_options.h"

// SolverConfig preset profiles.
//
// These factory functions produce four canonical SolverConfig presets that
// trade off between correctness, throughput, and robustness. They intentionally
// touch only solver-strategy fields; integration / cost / model parameters
// stay at SolverConfig defaults so users can start from a profile and override
// only what their model needs.
//
//   * make_reference_config()  -- correctness-first baseline. Simple solve
//     route (MERIT line search + MONOTONE barrier, no SOC / restoration /
//     direction refinement). Used as the regression-test baseline.
//   * make_default_config()    -- production default for everyday NMPC. Same
//     as a default-constructed SolverConfig: ADAPTIVE barrier, FILTER line
//     search, restoration on, SOC off. Returned for symmetry so users can
//     name-document their intent.
//   * make_speed_config()      -- aggressive throughput: low max_iters,
//     ACCEPTABLE_NMPC termination, no SOC / restoration / direction
//     refinement. Designed for warm-started MPC loops where the upstream
//     QP is already well posed.
//   * make_robust_config()     -- maximum robustness: MEHROTRA barrier with
//     filter line search, second-order correction, feasibility restoration,
//     RUIZ_EQUILIBRATION problem scaling, defect rollout refinement, tighter
//     mu_final. Designed for one-shot solves on ill-conditioned problems.

namespace minisolver {

inline SolverConfig make_reference_config()
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;

    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::MERIT;
    config.enable_corrector = false;
    config.enable_aggressive_barrier = false;
    config.direction_refinement = DirectionRefinementMode::NONE;
    config.enable_line_search_rollout = false;
    config.enable_slack_reset = false;
    config.enable_feasibility_restoration = false;
    config.enable_soc = false;

    config.termination_profile = TerminationProfile::STRICT_KKT;
    config.max_iters = 200;
    return config;
}

inline SolverConfig make_default_config()
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    return config;
}

inline SolverConfig make_speed_config()
{
    SolverConfig config = make_default_config();

    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.max_iters = 12;
    config.enable_soc = false;
    config.enable_feasibility_restoration = false;
    config.direction_refinement = DirectionRefinementMode::NONE;
    config.enable_corrector = false;
    config.enable_aggressive_barrier = true;

    config.tol_con = 1e-3;
    config.tol_dual = 1e-3;
    config.tol_mu = 1e-4;
    config.mu_final = 1e-5;
    return config;
}

inline SolverConfig make_robust_config()
{
    SolverConfig config = make_default_config();

    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.line_search_type = LineSearchType::FILTER;
    config.enable_soc = true;
    config.enable_feasibility_restoration = true;
    config.enable_slack_reset = true;
    config.direction_refinement = DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT;
    config.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;

    config.termination_profile = TerminationProfile::STRICT_KKT;
    config.max_iters = 300;
    config.tol_con = 1e-7;
    config.tol_dual = 1e-7;
    config.tol_mu = 1e-8;
    config.mu_final = 1e-8;
    return config;
}

} // namespace minisolver
