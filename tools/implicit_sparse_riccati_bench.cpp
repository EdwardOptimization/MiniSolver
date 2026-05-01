#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "fusedeulerimplicitregressionmodel.h"
#include "fusedgaussimplicitregressionmodel.h"
#include "fusedmidpointimplicitregressionmodel.h"
#include "genericeulerimplicitregressionmodel.h"
#include "genericgaussimplicitregressionmodel.h"
#include "genericmidpointimplicitregressionmodel.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

namespace {

constexpr int kHorizon = 10;
constexpr int kMaxHorizon = 16;
constexpr int kWarmupRuns = 10;
constexpr int kMeasureRuns = 200;

template <typename BaseModel> struct CountingFusedModel : public BaseModel {
    inline static int fused_calls = 0;

    template <typename T>
    static void compute_fused_riccati_step(
        const MSMat<T, BaseModel::NX, BaseModel::NX>& Vxx,
        const MSVec<T, BaseModel::NX>& Vx,
        KnotPoint<T, BaseModel::NX, BaseModel::NU, BaseModel::NC, BaseModel::NP>& kp)
    {
        ++fused_calls;
        BaseModel::template compute_fused_riccati_step<T>(Vxx, Vx, kp);
    }
};

SolverConfig make_config(IntegratorType integrator)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.integrator = integrator;
    config.default_dt = 0.08;
    config.max_iters = 80;
    config.tol_con = 1e-7;
    config.tol_dual = 1e-7;
    config.tol_grad = 1e-7;
    config.mu_final = 1e-8;
    config.tol_mu = 1e-8;
    config.tol_cost = 1e-10;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::FILTER;
    config.enable_profiling = false;
    return config;
}

template <typename Model>
MiniSolver<Model, kMaxHorizon> make_initialized_solver(IntegratorType type)
{
    MiniSolver<Model, kMaxHorizon> solver(kHorizon, Backend::CPU_SERIAL, make_config(type));
    solver.set_dt(0.08);
    solver.set_initial_state("x0", 1.0);
    solver.set_initial_state("x1", -0.45);
    solver.set_initial_state("x2", 0.30);
    solver.set_initial_state("x3", -0.20);
    solver.set_initial_state("x4", 0.12);
    solver.rollout_dynamics();
    return solver;
}

template <typename SolverT>
double total_cost(const SolverT& solver)
{
    double cost = 0.0;
    for (int k = 0; k <= kHorizon; ++k)
        cost += solver.get_stage_cost(k);
    return cost;
}

struct BenchResult {
    double median_us = 0.0;
    double mean_us = 0.0;
    int iterations = 0;
    int fused_calls = 0;
    SolverStatus status = SolverStatus::UNSOLVED;
    double cost = 0.0;
};

template <typename Model>
BenchResult run_bench(IntegratorType type)
{
    std::vector<double> samples;
    samples.reserve(kMeasureRuns);

    BenchResult result;
    for (int run = 0; run < kWarmupRuns + kMeasureRuns; ++run) {
        auto solver = make_initialized_solver<Model>(type);
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

template <typename FusedBaseModel, typename GenericModel>
void run_case(const char* name, IntegratorType type)
{
    using FusedModel = CountingFusedModel<FusedBaseModel>;

    FusedModel::fused_calls = 0;
    BenchResult fused = run_bench<FusedModel>(type);
    fused.fused_calls = FusedModel::fused_calls;

    BenchResult generic = run_bench<GenericModel>(type);

    const double speedup = generic.median_us / fused.median_us;
    const double cost_diff = std::abs(fused.cost - generic.cost);

    std::cout << std::left << std::setw(18) << name
              << std::right
              << std::setw(12) << std::fixed << std::setprecision(2) << fused.median_us
              << std::setw(12) << generic.median_us
              << std::setw(10) << std::setprecision(2) << speedup
              << std::setw(10) << fused.iterations
              << std::setw(10) << generic.iterations
              << std::setw(12) << status_to_string(fused.status)
              << std::setw(12) << status_to_string(generic.status)
              << std::setw(14) << std::scientific << std::setprecision(2) << cost_diff
              << std::setw(14) << fused.fused_calls
              << "\n";
}

} // namespace

int main()
{
    std::cout << "implicit sparse Riccati benchmark\n";
    std::cout << "runs: warmup=" << kWarmupRuns << ", measured=" << kMeasureRuns << "\n";
    std::cout << std::left << std::setw(18) << "case"
              << std::right << std::setw(12) << "fused_us"
              << std::setw(12) << "generic_us"
              << std::setw(10) << "speedup"
              << std::setw(10) << "fiters"
              << std::setw(10) << "giters"
              << std::setw(12) << "fstatus"
              << std::setw(12) << "gstatus"
              << std::setw(14) << "cost_diff"
              << std::setw(14) << "fused_calls"
              << "\n";

    run_case<FusedEulerImplicitRegressionModel, GenericEulerImplicitRegressionModel>(
        "backward_euler", IntegratorType::EULER_IMPLICIT);
    run_case<FusedMidpointImplicitRegressionModel, GenericMidpointImplicitRegressionModel>(
        "midpoint", IntegratorType::RK2_IMPLICIT);
    run_case<FusedGaussImplicitRegressionModel, GenericGaussImplicitRegressionModel>(
        "gauss_legendre", IntegratorType::RK4_IMPLICIT);

    return 0;
}
