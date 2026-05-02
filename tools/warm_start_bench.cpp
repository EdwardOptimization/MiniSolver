#include "minisolver/solver/solver.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace minisolver;

namespace {

struct TrackingIntegratorModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 2;
    static constexpr int NP = 1;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = { "xref" };
    static constexpr std::array<double, NC> constraint_weights = { 0.0, 0.0 };
    static constexpr std::array<int, NC> constraint_types = { 0, 0 };

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        kp.f_resid(0) = kp.x(0) + dt * kp.u(0);
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = kp.u(0) - 1.0;
        kp.g_val(1) = -kp.u(0) - 1.0;
        kp.C.setZero();
        kp.D.setZero();
        kp.D(0, 0) = 1.0;
        kp.D(1, 0) = -1.0;
    }

    template <typename T> static void compute_terminal_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.g_val(0) = -1.0;
        kp.g_val(1) = -1.0;
        kp.C.setZero();
        kp.D.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T err = kp.x(0) - kp.p(0);
        kp.cost = 5.0 * err * err + 0.02 * kp.u(0) * kp.u(0);
        kp.q(0) = 10.0 * err;
        kp.r(0) = 0.04 * kp.u(0);
        kp.Q(0, 0) = 10.0;
        kp.R(0, 0) = 0.04;
        kp.H.setZero();
    }

    template <typename T> static void compute_terminal_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T err = kp.x(0) - kp.p(0);
        kp.cost = 10.0 * err * err;
        kp.q(0) = 20.0 * err;
        kp.r.setZero();
        kp.Q(0, 0) = 20.0;
        kp.R.setZero();
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }

    template <typename T> static void compute_terminal_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_terminal_cost_gn(kp);
    }

    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost_gn(kp);
    }
};

struct StrategyCase {
    const char* name;
    InitializationMode initialization;
    BarrierStrategy barrier_update;
    WarmStartBarrierMode barrier_mode;
    WarmStartRegularizationMode regularization;
};

struct BenchStats {
    int solves = 0;
    int successes = 0;
    int total_iters_after_first = 0;
    int worst_iters_after_first = 0;
    double total_ms_after_first = 0.0;
};

constexpr int Horizon = 20;
constexpr int MaxN = 20;
constexpr int Steps = 60;
constexpr double Dt = 0.1;

SolverConfig base_config()
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 30;
    config.default_dt = Dt;
    config.line_search_type = LineSearchType::FILTER;
    config.enable_line_search_rollout = false;
    config.enable_rti = false;
    config.enable_profiling = false;
    config.barrier_strategy = BarrierStrategy::ADAPTIVE;
    config.hessian_approximation = HessianApproximation::OBJECTIVE_HESSIAN_ONLY;
    config.mu_init = 1e-1;
    config.mu_final = 1e-7;
    config.reg_init = 1e-4;
    config.reg_min = 1e-8;
    config.enable_soc = false;
    return config;
}

void initialize_guess(MiniSolver<TrackingIntegratorModel, MaxN>& solver, double x0, double xref)
{
    solver.set_initial_state({ x0 });
    solver.set_global_parameter(0, xref);
    for (int k = 0; k <= Horizon; ++k) {
        const double ratio = static_cast<double>(k) / static_cast<double>(Horizon);
        solver.set_state_guess(k, 0, x0 + ratio * (xref - x0));
        if (k < Horizon) {
            solver.set_control_guess(k, 0, 0.0);
        }
    }
}

void shift_business_guess(MiniSolver<TrackingIntegratorModel, MaxN>& solver, double measured_x)
{
    std::array<double, Horizon + 1> x {};
    std::array<double, Horizon> u {};
    for (int k = 0; k <= Horizon; ++k) {
        x[static_cast<size_t>(k)] = solver.get_state(k, 0);
        if (k < Horizon) {
            u[static_cast<size_t>(k)] = solver.get_control(k, 0);
        }
    }

    solver.set_state_guess(0, 0, measured_x);
    for (int k = 1; k <= Horizon; ++k) {
        solver.set_state_guess(k, 0, x[static_cast<size_t>(std::min(k + 1, Horizon))]);
    }
    for (int k = 0; k < Horizon; ++k) {
        solver.set_control_guess(k, 0, u[static_cast<size_t>(std::min(k + 1, Horizon - 1))]);
    }
}

BenchStats run_case(const StrategyCase& strategy)
{
    SolverConfig initial = base_config();
    initial.initialization = InitializationMode::COLD_START;
    MiniSolver<TrackingIntegratorModel, MaxN> solver(Horizon, Backend::CPU_SERIAL, initial);

    double x = 0.0;
    double xref = 2.5;
    initialize_guess(solver, x, xref);

    BenchStats stats;

    for (int step = 0; step < Steps; ++step) {
        if (step == 1) {
            SolverConfig config = base_config();
            config.initialization = strategy.initialization;
            config.barrier_strategy = strategy.barrier_update;
            config.warm_start_barrier = strategy.barrier_mode;
            config.warm_start_regularization = strategy.regularization;
            solver.set_config(config);
        }

        xref = 2.5 + 0.2 * std::sin(0.05 * static_cast<double>(step));
        solver.set_initial_state({ x });
        solver.set_global_parameter(0, xref);

        const auto t0 = std::chrono::steady_clock::now();
        const SolverStatus status = solver.solve();
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        ++stats.solves;
        const bool ok = (status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE);
        if (ok) {
            ++stats.successes;
        }

        if (step > 0) {
            const int iters = solver.get_iteration_count();
            stats.total_iters_after_first += iters;
            stats.worst_iters_after_first = std::max(stats.worst_iters_after_first, iters);
            stats.total_ms_after_first += elapsed_ms;
        }

        const double u0 = solver.get_control(0, 0);
        x += Dt * std::max(-1.0, std::min(1.0, u0));
        shift_business_guess(solver, x);
    }

    return stats;
}

} // namespace

int main()
{
    const std::array<StrategyCase, 7> cases = { {
        { "adaptive_primal_reset", InitializationMode::REUSE_PRIMAL, BarrierStrategy::ADAPTIVE,
            WarmStartBarrierMode::RESET_TO_MU_INIT,
            WarmStartRegularizationMode::RESET_TO_REG_INIT },
        { "adaptive_pd_reset_mu", InitializationMode::REUSE_PRIMAL_DUAL, BarrierStrategy::ADAPTIVE,
            WarmStartBarrierMode::RESET_TO_MU_INIT,
            WarmStartRegularizationMode::RESET_TO_REG_INIT },
        { "adaptive_pd_reuse_mu", InitializationMode::REUSE_PRIMAL_DUAL, BarrierStrategy::ADAPTIVE,
            WarmStartBarrierMode::REUSE_PREVIOUS_MU,
            WarmStartRegularizationMode::RESET_TO_REG_INIT },
        { "adaptive_pd_gap_mu", InitializationMode::REUSE_PRIMAL_DUAL, BarrierStrategy::ADAPTIVE,
            WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP,
            WarmStartRegularizationMode::RESET_TO_REG_INIT },
        { "monotone_pd_reset_mu", InitializationMode::REUSE_PRIMAL_DUAL, BarrierStrategy::MONOTONE,
            WarmStartBarrierMode::RESET_TO_MU_INIT,
            WarmStartRegularizationMode::RESET_TO_REG_INIT },
        { "monotone_pd_reuse_mu", InitializationMode::REUSE_PRIMAL_DUAL, BarrierStrategy::MONOTONE,
            WarmStartBarrierMode::REUSE_PREVIOUS_MU,
            WarmStartRegularizationMode::RESET_TO_REG_INIT },
        { "monotone_pd_gap_mu", InitializationMode::REUSE_PRIMAL_DUAL, BarrierStrategy::MONOTONE,
            WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP,
            WarmStartRegularizationMode::RESET_TO_REG_INIT },
    } };

    std::cout << "strategy,success_rate,avg_iters_after_first,worst_iters_after_first,"
                 "avg_ms_after_first\n";
    for (const auto& c : cases) {
        const BenchStats stats = run_case(c);
        const double denom = static_cast<double>(std::max(1, stats.solves - 1));
        std::cout << c.name << "," << std::fixed << std::setprecision(6)
                  << static_cast<double>(stats.successes) / static_cast<double>(stats.solves) << ","
                  << static_cast<double>(stats.total_iters_after_first) / denom << ","
                  << stats.worst_iters_after_first << "," << stats.total_ms_after_first / denom
                  << "\n";
    }
    return 0;
}
