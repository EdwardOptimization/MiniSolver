#pragma once

#include "minisolver/core/solver_options.h"
#include <algorithm>
#include <cmath>

namespace minisolver::detail {

struct BarrierUpdateKernel {
    static double update_mu(
        const SolverConfig& config, double current_mu, double max_kkt_error, double avg_gap)
    {
        switch (config.barrier_strategy) {
        case BarrierStrategy::MONOTONE:
            if (max_kkt_error < config.barrier_tolerance_factor * current_mu) {
                return std::max(config.mu_final, current_mu * config.mu_linear_decrease_factor);
            }
            return current_mu;
        case BarrierStrategy::ADAPTIVE: {
            double target = avg_gap * config.mu_safety_margin;
            // Allow mu to hold steady if residuals are not ready for a decrease.
            return std::max(config.mu_final, std::min(current_mu, target));
        }
        case BarrierStrategy::MEHROTRA: {
            double ratio = avg_gap / current_mu;
            if (ratio > 1.0) {
                ratio = 1.0;
            }
            double sigma = std::pow(ratio, 3);
            if (sigma < 0.05) {
                sigma = 0.05;
            }
            if (sigma > 0.8) {
                sigma = 0.8;
            }
            return std::max(config.mu_final, current_mu * sigma);
        }
        }
        return current_mu;
    }

    static double mehrotra_target_mu(
        const SolverConfig& config, double mu_curr, double mu_aff, double alpha_aff)
    {
        // Aggressive Update: Use sigma^k with k >= 1.
        double sigma_base = std::pow(mu_aff / mu_curr, 3);
        double sigma = sigma_base;

        if (config.enable_aggressive_barrier) {
            if (alpha_aff > 0.9) {
                sigma = std::min(sigma, 0.01);
            }
            if (mu_curr > 1.0) {
                sigma = std::min(sigma, 0.1);
            }
        } else {
            // Mehrotra centering heuristic: conservative when affine direction is blocked.
            if (alpha_aff < 0.1) {
                sigma = std::max(sigma, 0.5);
            } else if (alpha_aff > 0.9) {
                sigma = std::min(sigma, 0.1);
            }
        }

        if (sigma > 1.0) {
            sigma = 1.0;
        }
        if (sigma < 1e-4) {
            sigma = 1e-4;
        }

        return std::max(config.mu_final, sigma * mu_curr);
    }
};

} // namespace minisolver::detail
