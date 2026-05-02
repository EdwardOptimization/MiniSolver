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

struct WarmStartKernel {
    template <typename Model, typename TrajArray>
    static double average_complementarity_gap(
        const TrajArray& traj, int N, const SolverConfig& config)
    {
        double total_gap = 0.0;
        int total_pairs = 0;

        for (int k = 0; k <= N; ++k) {
            const auto& kp = traj[k];
            for (int i = 0; i < Model::NC; ++i) {
                const double s = kp.s(i);
                const double lam = kp.lam(i);
                if (std::isfinite(s) && std::isfinite(lam) && s > 0.0 && lam > 0.0) {
                    total_gap += s * lam;
                    ++total_pairs;
                }

                double w = 0.0;
                int type = 0;
                if constexpr (Model::NC > 0) {
                    if (static_cast<std::size_t>(i) < Model::constraint_types.size()) {
                        type = Model::constraint_types[i];
                        w = Model::constraint_weights[i];
                    }
                }
                if (type == 1 && w > 1e-6) {
                    const double soft_s = kp.soft_s(i);
                    const double soft_dual = w - lam;
                    if (std::isfinite(soft_s) && std::isfinite(soft_dual) && soft_s > 0.0
                        && soft_dual > config.min_barrier_slack) {
                        total_gap += soft_s * soft_dual;
                        ++total_pairs;
                    }
                }
            }
        }

        if (total_pairs == 0) {
            return config.mu_init;
        }
        return total_gap / static_cast<double>(total_pairs);
    }

    static double clamp_mu(const SolverConfig& config, double mu)
    {
        if (!std::isfinite(mu) || mu <= 0.0) {
            return config.mu_init;
        }
        const double lo = std::min(config.mu_init, config.mu_final);
        const double hi = std::max(config.mu_init, config.mu_final);
        return std::max(lo, std::min(hi, mu));
    }

    static double clamp_reg(const SolverConfig& config, double reg)
    {
        if (!std::isfinite(reg) || reg <= 0.0) {
            return config.reg_init;
        }
        const double lo = std::min(config.reg_min, config.reg_max);
        const double hi = std::max(config.reg_min, config.reg_max);
        return std::max(lo, std::min(hi, reg));
    }

    template <typename Model, typename TrajArray>
    static double select_barrier_mu(const SolverConfig& config, const TrajArray& traj, int N,
        double previous_mu, bool can_reuse_primal_dual)
    {
        if (!can_reuse_primal_dual) {
            return clamp_mu(config, config.mu_init);
        }

        switch (config.warm_start_barrier) {
        case WarmStartBarrierMode::REUSE_PREVIOUS_MU:
            return clamp_mu(config, previous_mu);
        case WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP:
            return clamp_mu(config, average_complementarity_gap<Model>(traj, N, config));
        case WarmStartBarrierMode::RESET_TO_MU_INIT:
        default:
            return clamp_mu(config, config.mu_init);
        }
    }

    static double select_regularization(const SolverConfig& config, double previous_reg)
    {
        switch (config.warm_start_regularization) {
        case WarmStartRegularizationMode::REUSE_PREVIOUS_REG:
            return clamp_reg(config, previous_reg);
        case WarmStartRegularizationMode::DECAY_PREVIOUS_REG:
            if (!std::isfinite(previous_reg) || previous_reg <= 0.0) {
                return clamp_reg(config, config.reg_init);
            }
            if (config.reg_scale_down <= 1.0 || !std::isfinite(config.reg_scale_down)) {
                return clamp_reg(config, previous_reg);
            }
            return clamp_reg(config, previous_reg / config.reg_scale_down);
        case WarmStartRegularizationMode::RESET_TO_REG_INIT:
        default:
            return clamp_reg(config, config.reg_init);
        }
    }
};

} // namespace minisolver::detail
