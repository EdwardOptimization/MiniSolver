#pragma once

#include "minisolver/core/types.h"
#include <limits>

namespace minisolver {

// SolverContext will grow into the boundary between the canonical solve loop
// and strategy kernels. Start with scalar metrics only; keep runtime state
// (mu/reg/iteration) in MiniSolver until snapshot I/O and tests move over.
struct SolverMetrics {
    double last_prim_inf = 0.0;
    double last_dual_inf = 0.0;
    double last_alpha = 1.0;

    // Mehrotra predictor diagnostics, used by regression tests and debugging.
    double last_mu_aff = 0.0;
    double last_alpha_aff = 0.0;

    void reset_algorithmic()
    {
        last_prim_inf = 0.0;
        last_dual_inf = 0.0;
        last_alpha = 1.0;
        last_mu_aff = 0.0;
        last_alpha_aff = 0.0;
    }

    void reset_solve() { last_alpha = 1.0; }
};

struct StepResidualSummary {
    double barrier_mu = 0.0;
    double max_barrier_complementarity_residual = 0.0;
    double max_complementarity_gap = 0.0;
    double max_primal_inf = 0.0;
    double max_unscaled_primal_inf = 0.0;
    double avg_complementarity_gap = 0.0;
};

struct SolveState {
    double mu = 0.0;
    double reg = 0.0;
    int current_iter = 0;
    int slack_reset_consecutive_count = 0;

    void reset_algorithmic(double mu_init, double reg_init)
    {
        mu = mu_init;
        reg = reg_init;
        current_iter = 0;
        slack_reset_consecutive_count = 0;
    }
};

struct ResidualState {
    double barrier_mu = 0.0;
    double max_barrier_complementarity_residual = 0.0;
    double max_complementarity_gap = 0.0;
    double max_primal_inf = 0.0;
    double max_unscaled_primal_inf = 0.0;
    double max_dual_inf = 0.0;
    double avg_complementarity_gap = 0.0;
    double objective_cost = 0.0;

    void reset_iteration()
    {
        barrier_mu = 0.0;
        max_barrier_complementarity_residual = 0.0;
        max_complementarity_gap = 0.0;
        max_primal_inf = 0.0;
        max_unscaled_primal_inf = 0.0;
        max_dual_inf = 0.0;
        avg_complementarity_gap = 0.0;
        objective_cost = 0.0;
    }
};

struct DirectionState {
    bool solve_success = false;
    double max_dual_inf = 0.0;
    double affine_mu = 0.0;
    double affine_alpha = 0.0;

    void reset_iteration()
    {
        solve_success = false;
        max_dual_inf = 0.0;
        affine_mu = 0.0;
        affine_alpha = 0.0;
    }
};

struct DirectionResult {
    SolverStatus status = SolverStatus::UNSOLVED;
    bool solve_success = false;
    double max_dual_inf = 0.0;
};

struct IterationResult {
    SolverStatus status = SolverStatus::UNSOLVED;
    TerminationReason reason = TerminationReason::NONE;
};

struct GlobalizationState {
    double accepted_alpha = 1.0;
    bool recovered = false;

    void reset_iteration()
    {
        accepted_alpha = 1.0;
        recovered = false;
    }
};

struct GlobalizationResult {
    SolverStatus status = SolverStatus::UNSOLVED;
    double alpha = 1.0;
    bool recovered = false;
};

struct TerminationState {
    SolverStatus loop_exit_status = SolverStatus::UNSOLVED;
    bool cost_stagnated = false;
    double residual_progress_best_norm = std::numeric_limits<double>::infinity();
    double residual_progress_mu = std::numeric_limits<double>::infinity();
    int residual_stagnation_count = 0;
    bool residual_progress_feasible_mode = false;

    void reset_solve()
    {
        loop_exit_status = SolverStatus::UNSOLVED;
        cost_stagnated = false;
        residual_progress_best_norm = std::numeric_limits<double>::infinity();
        residual_progress_mu = std::numeric_limits<double>::infinity();
        residual_stagnation_count = 0;
        residual_progress_feasible_mode = false;
    }
};

struct LoopExitDecision {
    bool should_exit = false;
    SolverStatus status = SolverStatus::UNSOLVED;
    TerminationReason reason = TerminationReason::NONE;
    bool cost_stagnated = false;
};

struct PostsolveResiduals {
    bool residuals_ok = true;
    bool linear_ok = false;
    double barrier_mu = 0.0;
    double max_primal_inf = 0.0;
    double max_unscaled_primal_inf = 0.0;
    double max_dual_inf = 0.0;
    double max_barrier_complementarity_residual = 0.0;
    double max_complementarity_gap = 0.0;
};

struct SolverContext {
    SolveState solve;
    SolverMetrics metrics;
    ResidualState residual;
    DirectionState direction;
    GlobalizationState globalization;
    TerminationState termination;
    SolverInfo info;

    void reset_algorithmic(double mu_init, double reg_init)
    {
        solve.reset_algorithmic(mu_init, reg_init);
        metrics.reset_algorithmic();
        residual.reset_iteration();
        direction.reset_iteration();
        globalization.reset_iteration();
        termination.reset_solve();
        info.reset();
    }

    void reset_solve()
    {
        metrics.reset_solve();
        residual.reset_iteration();
        direction.reset_iteration();
        globalization.reset_iteration();
        termination.reset_solve();
        info.reset();
    }
};

} // namespace minisolver
