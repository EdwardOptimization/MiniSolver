#pragma once
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include <limits>

namespace minisolver {

template <typename TrajArray> class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    // Solves the KKT system for the search direction (dx, du, ds, dlam)
    // affine_traj: If provided, adds Mehrotra corrector term (ds_aff * dlam_aff) to the RHS
    virtual bool solve(TrajArray& traj, int N, double mu, double reg, InertiaStrategy strategy,
        const SolverConfig& config, const TrajArray* affine_traj = nullptr)
        = 0;

    // Evaluate the dual stationarity residual on caller-provided scratch data.
    // The active solver trajectory must not be passed here; implementations are
    // allowed to overwrite search directions, gains, and bar derivatives in
    // scratch_traj while computing a fresh residual.
    virtual bool evaluate_dual_residual(TrajArray& scratch_traj, int N, double mu, double reg,
        InertiaStrategy strategy, const SolverConfig& config, double& max_dual_inf)
    {
        const bool ok = solve(scratch_traj, N, mu, reg, strategy, config);
        if (!ok) {
            max_dual_inf = std::numeric_limits<double>::infinity();
            return false;
        }

        max_dual_inf = 0.0;
        for (int k = 0; k <= N; ++k) {
            const double r_norm = MatOps::norm_inf(scratch_traj[k].r_bar);
            if (r_norm > max_dual_inf) {
                max_dual_inf = r_norm;
            }
        }
        return true;
    }

    // Overload for SOC (Second Order Correction)
    virtual bool solve_soc(TrajArray& /*traj*/, const TrajArray& /*soc_rhs_traj*/, int /*N*/,
        double /*mu*/, double /*reg*/, InertiaStrategy /*strategy*/, const SolverConfig& /*config*/)
    {
        return false;
    }

    // Optional post-solve direction refinement. Implementations may use original_system
    // as a read-only backup of the linearized system before in-place factorization.
    virtual bool refine_direction(TrajArray& /*traj*/, const TrajArray& /*original_system*/,
        int /*N*/, double /*mu*/, double /*reg*/, const SolverConfig& /*config*/)
    {
        return false;
    }
};

}
