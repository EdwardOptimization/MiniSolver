#pragma once

#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/solver/solver.h"
#include <memory>

namespace minisolver::test {

// Test-only friend for MiniSolver internals. Forward-declared in solver.h's
// minisolver::test namespace; defined here so production code never owns these
// private access paths.
template <typename Model, int MAX_N> struct SolverInternalAccess {
    using Solver = MiniSolver<Model, MAX_N>;

    static double& mu(Solver& s) { return s.context_.solve.mu; }
    static double& reg(Solver& s) { return s.context_.solve.reg; }

    static void apply_slack_reset(Solver& s, typename Solver::TrajArray& traj)
    {
        s.apply_slack_reset_(traj);
    }

    static double last_mu_aff(const Solver& s) { return s.context_.metrics.last_mu_aff; }
    static double last_alpha_aff(const Solver& s) { return s.context_.metrics.last_alpha_aff; }

    static double compute_fraction_to_boundary_primal(
        Solver& s, const typename Solver::TrajArray& traj)
    {
        return s.compute_fraction_to_boundary_(traj).primal;
    }

    static double compute_fraction_to_boundary_dual(
        Solver& s, const typename Solver::TrajArray& traj)
    {
        return s.compute_fraction_to_boundary_(traj).dual;
    }

    static double compute_affine_barrier_mu(Solver& s, const typename Solver::TrajArray& base,
        const typename Solver::TrajArray& aff, double alpha_primal_aff, double alpha_dual_aff)
    {
        return s.compute_affine_barrier_mu_(base, aff, alpha_primal_aff, alpha_dual_aff);
    }

    static double soft_s(const Solver& s, int stage, int idx)
    {
        return s.trajectory[stage].soft_s(idx);
    }

    static bool has_nans(Solver& s, const typename Solver::TrajArray& t) { return s.has_nans(t); }

    static bool has_valid_primal_dual_guess(Solver& s, const typename Solver::TrajArray& t)
    {
        return s.has_valid_primal_dual_guess(t);
    }

    static typename Solver::TrajArray& get_trajectory(Solver& s) { return s.trajectory.active(); }

    static StepResidualSummary evaluate_step_model(Solver& s, typename Solver::TrajArray& traj)
    {
        return s.evaluate_step_model_(traj);
    }

    static bool check_convergence(
        Solver& s, const StepResidualSummary& residuals, double max_dual_inf)
    {
        return s.check_convergence(residuals, max_dual_inf);
    }

    static void print_iteration_log(Solver& s, double alpha, bool header)
    {
        s.print_iteration_log(alpha, header);
    }

    static SolverStatus postsolve(Solver& s, SolverStatus loop_status)
    {
        return s.postsolve(loop_status);
    }

    static PostsolveResiduals refresh_postsolve_residuals(
        Solver& s, typename Solver::TrajArray& traj)
    {
        return s.refresh_postsolve_residuals_(traj);
    }

    static bool feasibility_restoration(Solver& s) { return s.feasibility_restoration(); }

    static SolverStatus step(Solver& s) { return s.execute_solve_iteration_().status; }

    static void set_linear_solver(
        Solver& s, std::unique_ptr<RiccatiSolver<typename Solver::TrajArray, Model>> solver)
    {
        s.linear_solver = std::move(solver);
    }

    static void record_line_search_diagnostics(Solver& s, const LineSearchResult& result)
    {
        s.record_line_search_diagnostics_(result);
    }

    static bool build_dirty(const Solver& s) { return s.build_state_.dirty; }

    static LineSearchType plan_line_search_type(const Solver& s)
    {
        return s.build_state_.plan.line_search_type;
    }

    static Backend plan_backend(const Solver& s) { return s.build_state_.plan.backend; }
};

} // namespace minisolver::test
