#pragma once
#include "core/types.h"
#include "core/solver_options.h"

namespace minisolver {

template<typename TrajArray>
class LinearSolver {
public:
    virtual ~LinearSolver() = default;
    
    // Solves the KKT system for the search direction (dx, du, ds, dlam)
    // Returns true on success, false on failure (e.g. matrix not PD)
    virtual bool solve(TrajArray& traj, int N, double mu, double reg, InertiaStrategy strategy) = 0;
};

}

