#pragma once

#include "minisolver/core/solver_options.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace minisolver::detail {

struct InitializationKernel {
    static bool should_initialize_primal_dual(
        const SolverConfig& config, bool has_valid_primal_dual_guess)
    {
        if (config.initialization != InitializationMode::REUSE_PRIMAL_DUAL) {
            return true;
        }
        return !has_valid_primal_dual_guess;
    }

    template <typename Model, typename Knot>
    static void initialize_constraint_primal_dual(Knot& kp, int i, double mu)
    {
        double g = kp.g_val(i);
        double w = 0.0;
        int type = 0;
        if constexpr (Model::NC > 0) {
            if (static_cast<std::size_t>(i) < Model::constraint_types.size()) {
                type = Model::constraint_types[i];
                w = Model::constraint_weights[i];
            }
        }

        if (type == 1 && w > 1e-6) { // L1 Soft Constraint
            // Central Path:
            // 1) g + s - soft_s = 0
            // 2) s * lam = mu
            // 3) soft_s * (w - lam) = mu
            // Reduce to quadratic in lam: g*lam^2 - (g*w - 2*mu)*lam - mu*w = 0
            double a = g;
            double b = -(g * w - 2 * mu);
            double c = -mu * w;

            double lam_val;
            if (std::abs(a) < 1e-9) {
                lam_val = w / 2.0;
            } else {
                double delta = b * b - 4 * a * c;
                if (delta < 0) {
                    delta = 0;
                }
                lam_val = (-b + std::sqrt(delta)) / (2 * a);
            }

            lam_val = std::max(1e-8, std::min(w - 1e-8, lam_val));

            kp.lam(i) = lam_val;
            kp.s(i) = mu / lam_val;
            kp.soft_s(i) = mu / (w - lam_val);
        } else if (type == 2 && w > 1e-6) { // L2 Soft Constraint
            // Central Path:
            // 1) g + s - lam/w = 0
            // 2) s * lam = mu
            // Reduce to quadratic in lam: lam^2 - g*w*lam - mu*w = 0
            double b = -g * w;
            double c = -mu * w;
            double delta = b * b - 4 * c;
            double lam_val = (-b + std::sqrt(delta)) / 2.0;

            kp.lam(i) = std::max(1e-8, lam_val);
            kp.s(i) = mu / kp.lam(i);
        } else { // Hard Constraint
            double s_val = std::max(1e-6, -g);
            kp.s(i) = s_val;
            kp.lam(i) = mu / s_val;
        }
    }
};

} // namespace minisolver::detail
