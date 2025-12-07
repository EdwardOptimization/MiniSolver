#pragma once
#include "algorithms/linear_solver.h"
#include "solver/riccati.h" // Reuse existing implementation functions

namespace minisolver {

// We need to pass ModelType to RiccatiSolver to access constraint metadata
// But LinearSolver base class doesn't have ModelType.
// The easiest way is to template RiccatiSolver on ModelType as well, 
// or pass metadata dynamically (but we prefer static for performance).
// However, MiniSolver<Model, N> owns the RiccatiSolver.
// So we can make RiccatiSolver<TrajArray, Model>

template<typename TrajArray, typename Model>
class RiccatiSolver : public LinearSolver<TrajArray> {
public:
    bool solve(TrajArray& traj, int N, double mu, double reg, InertiaStrategy strategy, 
               const SolverConfig& config, const TrajArray* affine_traj = nullptr) override {
        // Call the static/template function with Model type info
        return cpu_serial_solve<TrajArray, Model>(traj, N, mu, reg, strategy, config, affine_traj);
    }
    
    // SOC Implementation
    bool solve_soc(TrajArray& traj, const TrajArray& soc_rhs_traj, int N, double mu, double reg, InertiaStrategy strategy,
                   const SolverConfig& config) override {
        return cpu_serial_solve<TrajArray, Model>(traj, N, mu, reg, strategy, config, nullptr, &soc_rhs_traj);
    }

    // Iterative Refinement Implementation
    // Currently, cpu_serial_solve DOES re-factorize every time because we don't store P/L/D matrices.
    // So "Refinement" using the same function effectively re-computes the factorization (expensive).
    // But since regularization might have perturbed the system, we are solving (K + reg*I) dx = r.
    // True system is K dx* = r.
    // Residual rho = r - K dx.
    // Correction: (K + reg*I) ddx = rho.
    // dx_new = dx + ddx.
    // This reduces the error introduced by regularization.
    //
    // Implementation:
    // 1. Compute Residual rho = RHS - K * dx
    //    We need a helper to compute K * dx efficiently or just re-evaluate residuals?
    //    RHS is already stored in q_bar/r_bar (modified by barrier).
    //    Actually, we need to compute the *unregularized* KKT product.
    // 2. Solve for ddx.
    
    bool refine(TrajArray& traj, int N, double mu, double reg, const SolverConfig& config) override {
        // Placeholder: Since we don't store factorization, true IR is as expensive as a full step.
        // And we don't have a KKT-product function yet.
        // Implementing full IR requires significant infrastructure changes (KKT multiplication operator).
        // For now, return false (not implemented) or implement a simplified version later.
        return false;
    }
};

}
