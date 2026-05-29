#pragma once

// Central registry for SolverConfig fields that need mechanical iteration.
// Keep defaults in solver_options.h for readability. The full field list is
// for snapshot I/O, equality, and layout tests; enum validation shares only
// the enum field list.

#define MINISOLVER_CONFIG_ENUM_FIELDS(X)                                                           \
    X(backend)                                                                                     \
    X(initialization)                                                                              \
    X(warm_start_barrier)                                                                          \
    X(warm_start_regularization)                                                                   \
    X(termination_profile)                                                                         \
    X(constraint_scaling)                                                                          \
    X(objective_scaling)                                                                           \
    X(problem_scaling)                                                                             \
    X(integrator)                                                                                  \
    X(barrier_strategy)                                                                            \
    X(inertia_strategy)                                                                            \
    X(line_search_type)                                                                            \
    X(print_level)                                                                                 \
    X(hessian_approximation)                                                                       \
    X(direction_refinement)

#define MINISOLVER_CONFIG_FIELDS(X_ENUM, X_INT, X_DOUBLE, X_BOOL)                                  \
    X_ENUM(backend)                                                                                \
    X_ENUM(initialization)                                                                         \
    X_ENUM(warm_start_barrier)                                                                     \
    X_ENUM(warm_start_regularization)                                                              \
    X_ENUM(termination_profile)                                                                    \
    X_ENUM(constraint_scaling)                                                                     \
    X_ENUM(objective_scaling)                                                                      \
    X_ENUM(problem_scaling)                                                                        \
    X_DOUBLE(constraint_row_scale_min)                                                             \
    X_DOUBLE(constraint_row_scale_max)                                                             \
    X_DOUBLE(objective_scale_min)                                                                  \
    X_DOUBLE(objective_scale_max)                                                                  \
    X_ENUM(integrator)                                                                             \
    X_DOUBLE(default_dt)                                                                           \
    X_INT(newton_config.max_iters)                                                                 \
    X_DOUBLE(newton_config.tol)                                                                    \
    X_DOUBLE(newton_config.regularization)                                                         \
    X_ENUM(barrier_strategy)                                                                       \
    X_DOUBLE(mu_init)                                                                              \
    X_DOUBLE(mu_final)                                                                             \
    X_DOUBLE(mu_linear_decrease_factor)                                                            \
    X_DOUBLE(barrier_tolerance_factor)                                                             \
    X_DOUBLE(mu_safety_margin)                                                                     \
    X_ENUM(inertia_strategy)                                                                       \
    X_DOUBLE(reg_init)                                                                             \
    X_DOUBLE(reg_min)                                                                              \
    X_DOUBLE(reg_max)                                                                              \
    X_DOUBLE(reg_scale_up)                                                                         \
    X_DOUBLE(reg_scale_down)                                                                       \
    X_DOUBLE(regularization_step)                                                                  \
    X_DOUBLE(singular_threshold)                                                                   \
    X_DOUBLE(huge_penalty)                                                                         \
    X_INT(linear_solve_max_attempts)                                                               \
    X_DOUBLE(tol_con)                                                                              \
    X_DOUBLE(tol_dual)                                                                             \
    X_DOUBLE(tol_mu)                                                                               \
    X_DOUBLE(tol_cost)                                                                             \
    X_DOUBLE(feasible_tol_scale)                                                                   \
    X_BOOL(enable_residual_stagnation_detection)                                                   \
    X_INT(residual_stagnation_min_iters)                                                           \
    X_INT(residual_stagnation_window)                                                              \
    X_DOUBLE(residual_stagnation_rel_tol)                                                          \
    X_DOUBLE(residual_stagnation_abs_tol)                                                          \
    X_ENUM(line_search_type)                                                                       \
    X_INT(line_search_max_iters)                                                                   \
    X_DOUBLE(line_search_tau)                                                                      \
    X_DOUBLE(line_search_backtrack_factor)                                                         \
    X_DOUBLE(filter_gamma_theta)                                                                   \
    X_DOUBLE(filter_gamma_phi)                                                                     \
    X_DOUBLE(filter_theta_max_factor)                                                              \
    X_DOUBLE(armijo_c1)                                                                            \
    X_DOUBLE(min_barrier_slack)                                                                    \
    X_DOUBLE(barrier_inf_cost)                                                                     \
    X_DOUBLE(slack_reset_trigger)                                                                  \
    X_DOUBLE(warm_start_slack_init)                                                                \
    X_DOUBLE(soc_trigger_alpha)                                                                    \
    X_DOUBLE(merit_nu_init)                                                                        \
    X_DOUBLE(eta_suff_descent)                                                                     \
    X_INT(max_restoration_iters)                                                                   \
    X_DOUBLE(restoration_mu)                                                                       \
    X_DOUBLE(restoration_reg)                                                                      \
    X_DOUBLE(restoration_alpha)                                                                    \
    X_DOUBLE(restoration_sufficient_decrease_factor)                                               \
    X_INT(max_iters)                                                                               \
    X_ENUM(print_level)                                                                            \
    X_BOOL(enable_profiling)                                                                       \
    X_ENUM(hessian_approximation)                                                                  \
    X_ENUM(direction_refinement)                                                                   \
    X_BOOL(enable_line_search_rollout)                                                             \
    X_BOOL(enable_defect_correction)                                                               \
    X_BOOL(enable_corrector)                                                                       \
    X_BOOL(enable_aggressive_barrier)                                                              \
    X_BOOL(enable_slack_reset)                                                                     \
    X_BOOL(enable_feasibility_restoration)                                                         \
    X_BOOL(enable_soc)
