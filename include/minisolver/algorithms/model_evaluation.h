#pragma once

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/integrator/implicit_integrator.h"

namespace minisolver {
namespace detail {

    template <typename Model, typename Knot>
    void evaluate_model_stage(Knot& kp, const SolverConfig& config, double dt)
    {
        if (config.hessian_approximation == HessianApproximation::GAUSS_NEWTON) {
            Model::compute_cost_gn(kp);
        } else {
            Model::compute_cost_exact(kp);
        }

        detail::dispatch_compute_dynamics<Model>(kp, config.integrator, dt, config.newton_config);
        Model::compute_constraints(kp);
    }

} // namespace detail
} // namespace minisolver
