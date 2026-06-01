#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "fusedgaussimplicitregressionmodel.h"
#include "fusedmidpointimplicitregressionmodel.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

namespace {

constexpr int kHorizon = 10;
constexpr int kMaxHorizon = 16;
constexpr int kWarmupRuns = 10;
constexpr int kMeasureRuns = 200;

SolverConfig make_config(IntegratorType integrator, bool rollout)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.integrator = integrator;
    config.default_dt = 0.08;
    config.max_iters = 80;
    config.tol_con = 1e-7;
    config.tol_dual = 1e-7;
    config.mu_final = 1e-8;
    config.tol_mu = 1e-8;
    config.tol_cost = 1e-10;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::FILTER;
    config.enable_line_search_rollout = rollout;
    config.enable_profiling = false;
    return config;
}

template <typename Model>
MiniSolver<Model, kMaxHorizon> make_initialized_solver(IntegratorType type, bool rollout)
{
    MiniSolver<Model, kMaxHorizon> solver(
        kHorizon, Backend::CPU_SERIAL, make_config(type, rollout));
    solver.set_dt(0.08);
    solver.set_initial_state("x0", 1.0);
    solver.set_initial_state("x1", -0.45);
    solver.set_initial_state("x2", 0.30);
    solver.set_initial_state("x3", -0.20);
    solver.set_initial_state("x4", 0.12);
    solver.rollout_dynamics();
    return solver;
}

struct BenchResult {
    double median_us = 0.0;
    double mean_us = 0.0;
    double cost = 0.0;
    int iterations = 0;
    SolverStatus status = SolverStatus::UNSOLVED;
};

template <typename SolverT> double total_cost(const SolverT& solver)
{
    double cost = 0.0;
    for (int k = 0; k <= kHorizon; ++k) {
        cost += solver.get_stage_cost(k);
    }
    return cost;
}

template <typename Model> BenchResult run_bench(IntegratorType type, bool rollout)
{
    std::vector<double> samples;
    samples.reserve(kMeasureRuns);

    BenchResult result;
    for (int run = 0; run < kWarmupRuns + kMeasureRuns; ++run) {
        auto solver = make_initialized_solver<Model>(type, rollout);
        auto start = std::chrono::steady_clock::now();
        const SolverStatus status = solver.solve();
        auto end = std::chrono::steady_clock::now();

        if (run >= kWarmupRuns) {
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
        if (run == kWarmupRuns + kMeasureRuns - 1) {
            result.status = status;
            result.iterations = solver.get_iteration_count();
            result.cost = total_cost(solver);
        }
    }

    std::sort(samples.begin(), samples.end());
    result.median_us = samples[samples.size() / 2];
    result.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0)
        / static_cast<double>(samples.size());
    return result;
}

template <typename Model> void run_case(const char* name, IntegratorType type)
{
    BenchResult multiple = run_bench<Model>(type, false);
    BenchResult rollout = run_bench<Model>(type, true);

    std::cout << std::left << std::setw(16) << name << std::right << std::setw(14) << std::fixed
              << std::setprecision(2) << multiple.median_us << std::setw(14) << rollout.median_us
              << std::setw(10) << std::setprecision(2) << rollout.median_us / multiple.median_us
              << std::setw(10) << multiple.iterations << std::setw(10) << rollout.iterations
              << std::setw(12) << status_to_string(multiple.status) << std::setw(12)
              << status_to_string(rollout.status) << std::setw(14) << std::scientific
              << std::setprecision(2) << std::abs(multiple.cost - rollout.cost) << "\n";
}

} // namespace

int main()
{
    std::cout << "line-search rollout benchmark\n";
    std::cout << "runs: warmup=" << kWarmupRuns << ", measured=" << kMeasureRuns << "\n";
    std::cout << std::left << std::setw(16) << "case" << std::right << std::setw(14)
              << "multiple_us" << std::setw(14) << "rollout_us" << std::setw(10) << "ratio"
              << std::setw(10) << "miters" << std::setw(10) << "riters" << std::setw(12)
              << "mstatus" << std::setw(12) << "rstatus" << std::setw(14) << "cost_diff"
              << "\n";

    run_case<FusedMidpointImplicitRegressionModel>("midpoint", IntegratorType::GAUSS_LEGENDRE_2);
    run_case<FusedGaussImplicitRegressionModel>("gauss", IntegratorType::GAUSS_LEGENDRE_4);

    return 0;
}
