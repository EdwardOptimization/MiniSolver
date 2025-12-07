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
};

}
