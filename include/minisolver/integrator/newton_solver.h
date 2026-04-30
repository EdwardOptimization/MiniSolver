#pragma once

#include "minisolver/core/matrix_defs.h"
#include "minisolver/core/solver_options.h"

namespace minisolver {

// Stateful Newton solver for F(x) = 0.
// Workspace (F_, J_, delta_) persists across calls — zero re-initialization.
// Warm start: if the previous call converged, its solution seeds the next call.
//
// Func signature: void(const MSVec<T,N>& x, MSVec<T,N>& F, MSMat<T,N,N>& J)
template <typename T, int N>
class NewtonSolver {
public:
    // Solve F(x) = 0 starting from x (modified in-place).
    // warm_start: if true and previous call converged, replace x with
    // the previous solution before iterating.
    template <typename Func>
    bool solve(MSVec<T, N>& x, Func&& eval_func, const NewtonConfig& config = {},
               bool warm_start = true)
    {
        // Warm start: seed from previous converged solution
        if (warm_start && converged_last_)
            x = x_prev_;

        converged_last_ = false;

        for (int iter = 0; iter < config.max_iters; ++iter) {
            eval_func(x, F_, J_);

            if (MatOps::norm_inf(F_) < config.tol) {
                x_prev_ = x;
                converged_last_ = true;
                return true;
            }

            // Regularize for robustness (Levenberg-Marquardt style damping)
            MSMat<T, N, N> J_reg = J_;
            for (int i = 0; i < N; ++i)
                J_reg(i, i) += config.regularization;

            if (!MatOps::cholesky_solve(J_reg, F_, delta_))
                return false;

            x -= delta_;
        }

        if (MatOps::norm_inf(F_) < config.tol) {
            x_prev_ = x;
            converged_last_ = true;
            return true;
        }
        return false;
    }

    bool converged_last() const { return converged_last_; }
    const MSVec<T, N>& last_solution() const { return x_prev_; }

private:
    MSVec<T, N> F_;
    MSMat<T, N, N> J_;
    MSVec<T, N> delta_;
    MSVec<T, N> x_prev_;
    bool converged_last_ = false;
};

} // namespace minisolver
