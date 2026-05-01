#include "minisolver/solver/solver.h"

#include <array>
#include <chrono>
#include <iostream>

using namespace minisolver;

struct MeritBenchModel {
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
        MSVec<T, NX> out;
        out(0) = x(0) + u(0) * dt;
        return out;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        kp.f_resid = integrate(kp.x, kp.u, kp.p, dt, type);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0) * kp.x(0) - 1.0;
        kp.C(0, 0) = 2.0 * kp.x(0);
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T err = kp.x(0) - 0.2;
        kp.cost = err * err + 1e-3 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * err;
        kp.r(0) = 2e-3 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-3;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct L2BenchModel {
    static const int NX = 1;
    static const int NU = 1;
    static const int NC = 1;
    static const int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = { 5.0 };
    static constexpr std::array<int, NC> constraint_types = { 2 };

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> out;
        out(0) = x(0) + u(0) * dt;
        return out;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        kp.f_resid = integrate(kp.x, kp.u, kp.p, dt, type);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.x(0) - 1.0;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T err = kp.x(0) - 2.0;
        kp.cost = err * err + 1e-3 * kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * err;
        kp.r(0) = 2e-3 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-3;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

struct BenchResult {
    int success = 0;
    double avg_us = 0.0;
    double avg_iters = 0.0;
    double avg_cost = 0.0;
};

template <typename Model> BenchResult run_case(const char* name, double x0, double u0, int repeats)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.line_search_type = LineSearchType::MERIT;
    config.max_iters = 80;
    config.tol_con = 1e-6;
    config.tol_dual = 1e-6;
    config.tol_mu = 1e-6;
    config.integrator = IntegratorType::EULER_EXPLICIT;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    BenchResult result;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        MiniSolver<Model, 8> solver(3, Backend::CPU_SERIAL, config);
        solver.set_dt(0.2);
        solver.set_initial_state("x", x0);
        for (int k = 0; k < solver.get_horizon(); ++k) {
            solver.set_control_guess(k, 0, u0);
        }
        solver.rollout_dynamics();
        const SolverStatus status = solver.solve();
        if (status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE) {
            ++result.success;
        }
        result.avg_iters += solver.get_iteration_count();
        double total_cost = 0.0;
        for (int k = 0; k <= solver.get_horizon(); ++k) {
            total_cost += solver.get_stage_cost(k);
        }
        result.avg_cost += total_cost;
    }
    const auto t1 = std::chrono::steady_clock::now();

    result.avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / repeats;
    result.avg_iters /= repeats;
    result.avg_cost /= repeats;

    std::cout << name << ",success=" << result.success << "/" << repeats
              << ",avg_us=" << result.avg_us << ",avg_iters=" << result.avg_iters
              << ",avg_cost=" << result.avg_cost << "\n";
    return result;
}

int main()
{
    constexpr int repeats = 200;
    run_case<MeritBenchModel>("hard_nonlinear", 1.8, -1.0, repeats);
    run_case<L2BenchModel>("l2_soft", 2.5, -1.0, repeats);
    return 0;
}
