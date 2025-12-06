#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "core/types.h"

namespace minisolver {

// Fraction-to-Boundary Rule for Interior Point Methods
// Ensures that s > 0 and lam > 0 are maintained.
// alpha = max { alpha in [0,1] | s + alpha*ds >= (1-tau)*s, lam + alpha*dlam >= (1-tau)*lam }
// which simplifies to alpha <= -tau * x / dx for all dx < 0
template<typename Knot>
double fraction_to_boundary_rule(const std::vector<Knot>& traj, double tau = 0.995) {
    double alpha_s = 1.0;
    double alpha_lam = 1.0;
    
    // We assume NC is constant per knot, derived from Knot::NC
    const int NC = Knot::NC;

    for(const auto& kp : traj) {
        for(int i=0; i<NC; ++i) {
            double s = kp.s(i);
            double ds = kp.ds(i);
            double lam = kp.lam(i);
            double dlam = kp.dlam(i);

            if (ds < 0) {
                // Limit step to avoid s becoming negative
                alpha_s = std::min(alpha_s, -tau * s / ds);
            }
            if (dlam < 0) {
                // Limit step to avoid lam becoming negative
                alpha_lam = std::min(alpha_lam, -tau * lam / dlam);
            }
        }
    }
    return std::min(alpha_s, alpha_lam);
}

}


