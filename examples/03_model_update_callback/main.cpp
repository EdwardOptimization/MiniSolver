#include <array>
#include <iostream>

#include "minisolver/solver/solver.h"

using namespace minisolver;

struct CallbackDemoModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 0;
    static const int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "x_ref" };
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> xn;
        xn(0) = x(0) + dt * u(0);
        return xn;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + dt * kp.u(0);
        kp.A.setIdentity();
        kp.B.setZero();
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val.setZero();
        kp.C.setZero();
        kp.D.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T error = kp.x(0) - kp.p(0);
        kp.cost = error * error + T(0.05) * kp.u(0) * kp.u(0);
        kp.q(0) = T(2.0) * error;
        kp.r(0) = T(0.1) * kp.u(0);
        kp.Q.setZero();
        kp.Q(0, 0) = T(2.0);
        kp.R.setZero();
        kp.R(0, 0) = T(0.1);
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

using DemoSolver = MiniSolver<CallbackDemoModel, 8>;

struct MovingReference {
    int calls = 0;
    int x_ref_idx = -1;
    double base = 1.0;
    double per_iteration_shift = 0.25;
};

ApiStatus update_reference(DemoSolver& solver, void* user)
{
    auto* ref = static_cast<MovingReference*>(user);
    ++ref->calls;

    const double x_ref = ref->base + ref->per_iteration_shift * solver.get_iteration_count();
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        const ApiStatus status = solver.set_parameter(k, ref->x_ref_idx, x_ref);
        if (status != ApiStatus::OK) {
            return status;
        }
    }
    return ApiStatus::OK;
}

int main()
{
    SolverConfig config;
    config.print_level = PrintLevel::ITER;
    config.line_search_type = LineSearchType::NONE;
    config.max_iters = 4;
    config.default_dt = 0.1;

    DemoSolver solver(5, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 0.0);
    for (int k = 0; k < solver.get_horizon(); ++k) {
        solver.set_control_guess(k, "u", 0.0);
    }

    MovingReference ref;
    ref.x_ref_idx = solver.get_param_idx("x_ref");
    solver.set_model_update_callback(update_reference, &ref);

    const SolverStatus status = solver.solve();
    const SolverInfo& info = solver.get_info();

    std::cout << "status=" << status_to_string(status) << " iterations=" << info.iterations
              << " callback_calls=" << ref.calls
              << " final_x_ref=" << solver.get_parameter(0, "x_ref") << "\n";
    return 0;
}
