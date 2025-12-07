#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "core/types.h"

namespace minisolver {

// Fraction-to-Boundary Rule for Interior Point Methods
// Updated to accept any container type and active horizon N
template<typename TrajVector, typename ModelType>
double fraction_to_boundary_rule(const TrajVector& traj, int N, double tau = 0.995) {
    using Knot = typename TrajVector::value_type;
    double alpha_s = 1.0;
    double alpha_lam = 1.0;
    double alpha_soft_s = 1.0;
    double alpha_soft_dual = 1.0;
    
    const int NC = Knot::NC;

    for(int k=0; k<=N; ++k) {
        const auto& kp = traj[k];
        for(int i=0; i<NC; ++i) {
            double s = kp.s(i);
            double ds = kp.ds(i);
            double lam = kp.lam(i);
            double dlam = kp.dlam(i);

            // Lower bound for s: s + alpha * ds >= (1-tau) * s > 0
            if (ds < 0) {
                alpha_s = std::min(alpha_s, -tau * s / ds);
            }
            
            // Lower bound for lam
            if (dlam < 0) {
                alpha_lam = std::min(alpha_lam, -tau * lam / dlam);
            }
            
            double w = 0.0;
            int type = 0;
            if constexpr (NC > 0) {
                 if (i < ModelType::constraint_types.size()) {
                    type = ModelType::constraint_types[i];
                    w = ModelType::constraint_weights[i];
                 }
            }
            
            // For L1, we explicitly track soft_s and soft_dual
            if (type == 1 && w > 1e-6) {
                double ss = kp.soft_s(i);
                double dss = kp.dsoft_s(i);
                // soft_s >= 0
                if (dss < 0) {
                    alpha_soft_s = std::min(alpha_soft_s, -tau * ss / dss);
                }
                
                // Upper bound on lam implies w - lam >= 0.
                // Let nu = w - lam. dnu = -dlam.
                // nu + alpha * dnu >= 0 => w - lam - alpha * dlam >= 0.
                // If dlam > 0, alpha <= (w - lam) / dlam.
                if (dlam > 0) {
                    double gap = w - lam;
                    if (gap < 1e-9) gap = 1e-9;
                    alpha_lam = std::min(alpha_lam, tau * gap / dlam);
                }
            }
        }
    }
    return std::min({alpha_s, alpha_lam, alpha_soft_s, alpha_soft_dual});
}

}
