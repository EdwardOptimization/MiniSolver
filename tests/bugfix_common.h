#pragma once

#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

using namespace minisolver;

// Test-only friend for MiniSolver internals. Forward-declared in solver.h's
// minisolver::test namespace; defined here so the friend access is a pure
// test concern and production builds don't link against it.
namespace minisolver::test {
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
    static double soft_s(const Solver& s, int stage, int idx)
    {
        return s.trajectory[stage].soft_s(idx);
    }
    static bool has_nans(Solver& s, const typename Solver::TrajArray& t) { return s.has_nans(t); }
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
    static SolverStatus postsolve(Solver& s, SolverStatus loop_status)
    {
        return s.postsolve(loop_status);
    }
    static PostsolveResiduals refresh_postsolve_residuals(
        Solver& s, typename Solver::TrajArray& traj)
    {
        return s.refresh_postsolve_residuals_(traj);
    }
    static SolverStatus step(Solver& s) { return s.execute_solve_iteration_().status; }
    static void set_linear_solver(
        Solver& s, std::unique_ptr<RiccatiSolver<typename Solver::TrajArray, Model>> solver)
    {
        s.linear_solver = std::move(solver);
    }
    static bool build_dirty(const Solver& s) { return s.build_state_.dirty; }
    static LineSearchType plan_line_search_type(const Solver& s)
    {
        return s.build_state_.plan.line_search_type;
    }
    static Backend plan_backend(const Solver& s) { return s.build_state_.plan.backend; }
};
} // namespace minisolver::test

// Minimal test model: 1 state, 1 control, 1 constraint.
struct BugTestModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.u(0) - 1.0;
        kp.C(0, 0) = 0.0;
        kp.D(0, 0) = 1.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0) + 0.01 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 0.02 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 0.02;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};

// L1 soft-constraint model for testing L1-specific IPM behavior.
struct L1TestModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 100.0 };
    static constexpr std::array<int, NC> constraint_types = { 1 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0) - 5.0;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        T diff = kp.x(0) - 10.0;
        kp.cost = diff * diff + 0.01 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * diff;
        kp.r(0) = 0.02 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 0.02;
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};
