#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

namespace {

struct ScalingCase {
    std::string name;
    SolverConfig config;
};

struct ScalingResult {
    std::string name;
    int runs = 0;
    int successes = 0;
    double avg_time_ms = 0.0;
    double avg_iters = 0.0;
    double avg_primal = 0.0;
    double avg_unscaled_primal = 0.0;
    double avg_dual = 0.0;
    double avg_complementarity = 0.0;
    double avg_cost = 0.0;
};

void setup_car_obstacle_scenario(MiniSolver<CarModel, 60>& solver)
{
    const int N = solver.get_horizon();
    const double dt = solver.get_config().default_dt;
    const double target_v = 5.0;

    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("v", 0.0);

    for (int k = 0; k <= N; ++k) {
        const double t = static_cast<double>(k) * dt;

        solver.set_parameter(k, "v_ref", target_v);
        solver.set_parameter(k, "x_ref", t * target_v);
        solver.set_parameter(k, "y_ref", 0.0);

        solver.set_parameter(k, "obs_x", 10.0);
        solver.set_parameter(k, "obs_y", 0.0);
        solver.set_parameter(k, "obs_rad", 1.5);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);

        // Deliberately mixed magnitudes: this is a realistic reason to consider
        // objective/constraint normalization while keeping the model semantics fixed.
        solver.set_parameter(k, "w_pos", 100.0);
        solver.set_parameter(k, "w_vel", 0.01);
        solver.set_parameter(k, "w_theta", 0.1);
        solver.set_parameter(k, "w_acc", 0.001);
        solver.set_parameter(k, "w_steer", 10.0);

        if (k < N) {
            solver.set_control_guess(k, "acc", 0.0);
            solver.set_control_guess(k, "steer", 0.0);
        }
    }
}

double compute_total_cost(const MiniSolver<CarModel, 60>& solver)
{
    double cost = 0.0;
    for (int k = 0; k <= solver.get_horizon(); ++k) {
        cost += solver.get_stage_cost(k);
    }
    return cost;
}

bool success_status(SolverStatus status)
{
    return status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE;
}

ScalingResult run_case(const ScalingCase& c, int runs, int warmup)
{
    constexpr int N = 50;
    ScalingResult result;
    result.name = c.name;
    result.runs = runs;

    std::vector<double> times;
    times.reserve(static_cast<size_t>(runs));

    for (int i = 0; i < warmup + runs; ++i) {
        MiniSolver<CarModel, 60> solver(N, Backend::CPU_SERIAL, c.config);
        setup_car_obstacle_scenario(solver);
        solver.rollout_dynamics();

        const auto start = std::chrono::steady_clock::now();
        const SolverStatus status = solver.solve();
        const auto end = std::chrono::steady_clock::now();

        if (i < warmup) {
            continue;
        }

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        const SolverInfo& info = solver.get_info();

        times.push_back(elapsed_ms);
        result.successes += success_status(status) ? 1 : 0;
        result.avg_iters += static_cast<double>(info.iterations);
        result.avg_primal += info.primal_inf;
        result.avg_unscaled_primal += info.unscaled_primal_inf;
        result.avg_dual += info.dual_inf;
        result.avg_complementarity += info.complementarity_inf;
        result.avg_cost += compute_total_cost(solver);
    }

    const double denom = static_cast<double>(std::max(1, runs));
    result.avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / denom;
    result.avg_iters /= denom;
    result.avg_primal /= denom;
    result.avg_unscaled_primal /= denom;
    result.avg_dual /= denom;
    result.avg_complementarity /= denom;
    result.avg_cost /= denom;

    return result;
}

SolverConfig base_config()
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.integrator = IntegratorType::RUNGE_KUTTA_4;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.line_search_type = LineSearchType::FILTER;
    config.max_iters = 100;
    config.tol_con = 1e-4;
    config.tol_dual = 1e-4;
    config.tol_mu = 1e-5;
    config.enable_profiling = false;
    return config;
}

} // namespace

int main(int argc, char** argv)
{
    int runs = 50;
    int warmup = 5;
    if (argc > 1) {
        runs = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        warmup = std::max(0, std::atoi(argv[2]));
    }

    SolverConfig none = base_config();

    SolverConfig row = base_config();
    row.constraint_scaling = ConstraintScalingMethod::ROW_INF_NORM;

    SolverConfig obj = base_config();
    obj.objective_scaling = ObjectiveScalingMethod::HESSIAN_GERSHGORIN;

    SolverConfig ruiz = base_config();
    ruiz.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;

    const std::vector<ScalingCase> cases = {
        { "NONE", none },
        { "ROW_INF_NORM", row },
        { "HESSIAN_GERSHGORIN", obj },
        { "RUIZ_EQUILIBRATION", ruiz },
    };

    std::cout << "Scaling benchmark: CarModel obstacle, N=50, runs=" << runs
              << ", warmup=" << warmup << "\n";
    std::cout << std::left << std::setw(22) << "Scaling" << std::setw(12) << "Success"
              << std::setw(12) << "Time(ms)" << std::setw(10) << "Iters" << std::setw(14) << "Prim"
              << std::setw(14) << "RawPrim" << std::setw(14) << "Dual" << std::setw(14) << "Comp"
              << std::setw(14) << "Cost"
              << "\n";

    for (const auto& c : cases) {
        const ScalingResult r = run_case(c, runs, warmup);
        const double success_rate
            = 100.0 * static_cast<double>(r.successes) / static_cast<double>(std::max(1, r.runs));

        std::cout << std::left << std::setw(22) << r.name << std::fixed << std::setprecision(1)
                  << std::setw(12) << success_rate << std::setprecision(3) << std::setw(12)
                  << r.avg_time_ms << std::setw(10) << r.avg_iters << std::scientific
                  << std::setprecision(2) << std::setw(14) << r.avg_primal << std::setw(14)
                  << r.avg_unscaled_primal << std::setw(14) << r.avg_dual << std::setw(14)
                  << r.avg_complementarity << std::setw(14) << r.avg_cost << "\n";
    }

    return 0;
}
