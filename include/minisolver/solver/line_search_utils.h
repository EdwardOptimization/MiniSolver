#pragma once
#include "minisolver/core/constraint_semantics.h"
#include "minisolver/core/types.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace minisolver {

// Fraction-to-Boundary Rule for Interior Point Methods
// Updated to accept any container type and active horizon N
template <typename TrajVector, typename ModelType>
double fraction_to_boundary_rule(
    const TrajVector& traj, int N, double tau = 0.995, const SolverConfig& config = SolverConfig())
{
    using Knot = typename TrajVector::value_type;
    double alpha_s = 1.0;
    double alpha_lam = 1.0;
    double alpha_soft_s = 1.0;

    const int NC = Knot::NC;

    for (int k = 0; k <= N; ++k) {
        const auto& kp = traj[k];
        for (int i = 0; i < NC; ++i) {
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

            // For L1 and mixed L1+L2, explicitly track soft_s and the
            // implicit soft dual z = w1 + w2*soft_s - lam.
            if (detail::active_l1_soft_constraint<ModelType>(kp, i, config)) {
                const double w = kp.l1_weight(i);
                double ss = kp.soft_s(i);
                double dss = kp.dsoft_s(i);
                // soft_s >= 0
                if (dss < 0) {
                    alpha_soft_s = std::min(alpha_soft_s, -tau * ss / dss);
                }

                const double soft_dual_floor = detail::l1_soft_dual_floor(w, config);
                double gap = detail::l1_soft_dual_gap<ModelType>(kp, i) - soft_dual_floor;
                const double dgap = detail::l1_soft_dual_direction<ModelType>(kp, i);
                if (dgap < 0.0) {
                    if (gap < 0.0) {
                        gap = 0.0;
                    }
                    alpha_lam = std::min(alpha_lam, tau * gap / (-dgap));
                }
            }
        }
    }
    return std::min({ alpha_s, alpha_lam, alpha_soft_s });
}

}
