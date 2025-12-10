#pragma once
#include "core/types.h"
#include "core/solver_options.h"

namespace minisolver {

template<typename TrajArray>
class LinearSolver {
public:
    virtual ~LinearSolver() = default;
    
    // Solves the KKT system for the search direction (dx, du, ds, dlam)
    // affine_traj: If provided, adds Mehrotra corrector term (ds_aff * dlam_aff) to the RHS
    virtual bool solve(TrajArray& traj, int N, double mu, double reg, InertiaStrategy strategy, 
                      const SolverConfig& config, const TrajArray* affine_traj = nullptr) = 0;
                      
    // Overload for SOC (Second Order Correction)
    virtual bool solve_soc(TrajArray& traj, const TrajArray& soc_rhs_traj, int N, double mu, double reg, InertiaStrategy strategy,
                          const SolverConfig& config) {
        return false;
    }

    // [NEW] Iterative Refinement
    // Solves K * dx_corr = r - K * dx
    // Uses the existing factorization (if available/stored) or re-solves.
    // For Riccati, we use the original_system (candidate buffer) to re-run the solve on residuals.
    virtual bool refine(TrajArray& traj, const TrajArray& original_system, int N, double mu, double reg, const SolverConfig& config) {
        return false; 
    }
};

}
