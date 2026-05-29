#pragma once

#include "minisolver/core/types.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace minisolver::detail {

struct ResidualStagnationConfigView {
    bool enabled = false;
    bool fixed_iteration_profile = false;
    bool callback_installed = false;
    int min_iters = 0;
    int window = 1;
    double rel_tol = 0.0;
    double abs_tol = 0.0;
    double tol_con = 1.0;
    double tol_dual = 1.0;
    double tol_mu = 1.0;
    double feasible_tol_scale = 1.0;
};

struct ResidualStagnationSample {
    double max_primal_inf = 0.0;
    double max_dual_inf = 0.0;
    double max_complementarity_gap = 0.0;
    double current_mu = 0.0;
    int current_iter = 0;
};

struct ResidualStagnationResult {
    SolverStatus status = SolverStatus::UNSOLVED;
    TerminationReason reason = TerminationReason::NONE;
};

// Solve-local progress state. Snapshot I/O persists config/case/status data,
// not the in-flight residual-stagnation window.
class ResidualStagnationMonitor {
public:
    void reset()
    {
        best_norm_ = std::numeric_limits<double>::infinity();
        progress_mu_ = std::numeric_limits<double>::infinity();
        stagnation_count_ = 0;
        feasible_mode_ = false;
    }

    ResidualStagnationResult update(
        const ResidualStagnationSample& sample, const ResidualStagnationConfigView& config)
    {
        ResidualStagnationResult result;
        if (!config.enabled || config.fixed_iteration_profile || config.callback_installed) {
            return result;
        }

        const double feasible_bound = config.tol_con * config.feasible_tol_scale;
        const bool feasible_mode = sample.max_primal_inf <= feasible_bound;
        const double primal_norm = sample.max_primal_inf / std::max(config.tol_con, 1.0e-300);
        const double dual_norm = sample.max_dual_inf / std::max(config.tol_dual, 1.0e-300);
        const double complementarity_norm
            = sample.max_complementarity_gap / std::max(config.tol_mu, 1.0e-300);

        if (!std::isfinite(primal_norm) || !std::isfinite(dual_norm)
            || !std::isfinite(complementarity_norm)) {
            result.status = SolverStatus::NUMERICAL_ERROR;
            result.reason = TerminationReason::NUMERICAL_ERROR;
            return result;
        }
        const double kkt_norm = std::max(primal_norm, std::max(dual_norm, complementarity_norm));

        if (feasible_mode && std::isfinite(progress_mu_) && sample.current_mu < progress_mu_) {
            reset();
        }
        progress_mu_ = sample.current_mu;

        if (std::isfinite(best_norm_) && feasible_mode != feasible_mode_) {
            reset();
            progress_mu_ = sample.current_mu;
        }
        feasible_mode_ = feasible_mode;

        // Before loose primal feasibility, only primal progress is meaningful.
        // After that point, monitor normalized KKT cleanup progress.
        const double progress_norm = feasible_mode ? kkt_norm : primal_norm;

        const bool monitor_uninitialized = !std::isfinite(best_norm_);
        const double required_progress
            = std::max(config.abs_tol, config.rel_tol * std::max(1.0, best_norm_));
        const bool improved
            = monitor_uninitialized || progress_norm + required_progress < best_norm_;
        if (improved) {
            best_norm_ = progress_norm;
            stagnation_count_ = 0;
            return result;
        }

        if (sample.current_iter >= config.min_iters) {
            ++stagnation_count_;
        }

        if (stagnation_count_ < config.window) {
            return result;
        }

        result.status = SolverStatus::INSUFFICIENT_PROGRESS;
        result.reason = TerminationReason::RESIDUAL_STAGNATION;
        return result;
    }

    double best_norm() const { return best_norm_; }
    int stagnation_count() const { return stagnation_count_; }
    bool feasible_mode() const { return feasible_mode_; }

private:
    double best_norm_ = std::numeric_limits<double>::infinity();
    double progress_mu_ = std::numeric_limits<double>::infinity();
    int stagnation_count_ = 0;
    bool feasible_mode_ = false;
};

} // namespace minisolver::detail
