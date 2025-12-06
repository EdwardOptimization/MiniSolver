#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "core/types.h"

namespace minisolver {

// Fraction-to-Boundary Rule for Interior Point Methods
// Updated to accept any container type and active horizon N
template<typename TrajVector>
double fraction_to_boundary_rule(const TrajVector& traj, int N, double tau = 0.995) {
    using Knot = typename TrajVector::value_type;
    double alpha_s = 1.0;
    double alpha_lam = 1.0;
    
    const int NC = Knot::NC;

    for(int k=0; k<=N; ++k) {
        const auto& kp = traj[k];
        for(int i=0; i<NC; ++i) {
            double s = kp.s(i);
            double ds = kp.ds(i);
            double lam = kp.lam(i);
            double dlam = kp.dlam(i);

            if (ds < 0) {
                alpha_s = std::min(alpha_s, -tau * s / ds);
            }
            if (dlam < 0) {
                alpha_lam = std::min(alpha_lam, -tau * lam / dlam);
            }
        }
    }
    return std::min(alpha_s, alpha_lam);
}

}
