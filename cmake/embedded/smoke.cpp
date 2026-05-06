// Embedded profile smoke source.
//
// Purpose: instantiate a small but non-trivial MiniSolver template
// configuration so that we can:
//   1. Verify that MINISOLVER_EMBEDDED_PROFILE=ON compiles end-to-end without
//      <iostream> and without dynamic exception dependencies.
//   2. Provide a stable .o artifact that the binary-size budget script can
//      measure on the ARM Cortex-M cross build.
//
// The function below intentionally has external linkage and is reachable from
// a synthetic entry point so the linker (when used in --gc-sections mode by a
// downstream consumer) does not strip it.
//
// This file is *not* a test. It is the canonical anchor for the embedded size
// budget; see scripts/check_arm_size_budget.sh and the testing matrix entry
// for "Embedded ARM cross-build".

#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"

#include <array>

namespace minisolver_embedded_smoke {

struct SmokeModel {
    static constexpr int NX = 2;
    static constexpr int NU = 1;
    static constexpr int NC = 0;
    static constexpr int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x0", "x1" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "x_ref" };
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static minisolver::MSVec<T, NX> integrate(const minisolver::MSVec<T, NX>& x,
        const minisolver::MSVec<T, NU>& u, const minisolver::MSVec<T, NP>&, double dt,
        minisolver::IntegratorType)
    {
        minisolver::MSVec<T, NX> xn;
        xn(0) = x(0) + x(1) * dt;
        xn(1) = x(1) + u(0) * dt;
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        minisolver::KnotPoint<T, NX, NU, NC, NP>& kp, minisolver::IntegratorType, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.x(1) * dt;
        kp.f_resid(1) = kp.x(1) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.A(0, 1) = dt;
        kp.A(1, 0) = 0.0;
        kp.A(1, 1) = 1.0;
        kp.B(0, 0) = 0.0;
        kp.B(1, 0) = dt;
    }

    template <typename T> static void compute_constraints(minisolver::KnotPoint<T, NX, NU, NC, NP>&)
    {
    }

    template <typename T> static void compute_cost_gn(minisolver::KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff = kp.x(0) - kp.p(0);
        kp.cost = static_cast<T>(5.0) * diff * diff + static_cast<T>(0.25) * kp.u(0) * kp.u(0);
        kp.q(0) = static_cast<T>(10.0) * diff;
        kp.q(1) = static_cast<T>(0.0);
        kp.r(0) = static_cast<T>(0.5) * kp.u(0);
        kp.Q(0, 0) = 10.0;
        kp.Q(0, 1) = 0.0;
        kp.Q(1, 0) = 0.0;
        kp.Q(1, 1) = 0.0;
        kp.R(0, 0) = 0.5;
        kp.H(0, 0) = 0.0;
        kp.H(1, 0) = 0.0;
    }

    template <typename T>
    static void compute_cost_exact(minisolver::KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

// Externally visible anchor so --gc-sections does not silently drop the
// instantiation when a downstream consumer links against this object.
//
// The function is deliberately defined out-of-line and named so that
// arm-none-eabi-nm output is easy to interpret in CI logs.
extern "C" int minisolver_embedded_smoke_solve()
{
    constexpr int N = 8;
    minisolver::SolverConfig config;
    config.print_level = minisolver::PrintLevel::NONE;
    config.max_iters = 8;

    minisolver::MiniSolver<SmokeModel, 16> solver(N, minisolver::Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x0", 1.0);
    solver.set_initial_state("x1", 0.0);
    for (int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "x_ref", 0.0);
    }
    solver.rollout_dynamics();
    const minisolver::SolverStatus status = solver.solve();
    return static_cast<int>(status);
}

} // namespace minisolver_embedded_smoke
