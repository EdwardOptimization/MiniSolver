#pragma once
#include "algorithms/linear_solver.h"
#include "solver/riccati.h" // Reuse existing implementation functions

namespace minisolver {

template<typename TrajArray>
class RiccatiSolver : public LinearSolver<TrajArray> {
public:
    bool solve(TrajArray& traj, int N, double mu, double reg, InertiaStrategy strategy) override {
        // Call the existing static/template function
        return cpu_serial_solve(traj, N, mu, reg, strategy);
    }
};

}

