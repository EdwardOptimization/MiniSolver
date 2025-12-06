#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>

#include "model/car_model.h"
#include "solver/solver.h"

using namespace minisolver;

struct BenchmarkResult {
    std::string name;
    int iters;
    double total_time_ms;
    double derivatives_ms;
    double linear_solve_ms;
    double line_search_ms;
    bool converged;
    double final_cost;
    double final_viol;
};

BenchmarkResult run_test(const std::string& name, SolverConfig config) {
    int N = 60;
    config.verbose = false; // Mute output for benchmark
    config.debug_mode = false;
    
    PDIPMSolver<CarModel> solver(N, Backend::CPU_SERIAL, config);
    
    // Scenario setup
    std::vector<double> dts(N);
    for(int k=0; k<N; ++k) dts[k] = (k < 20) ? 0.05 : 0.2;
    solver.set_dt(dts);
    
    double obs_x = 12.0; double obs_y = 0.0; double obs_rad = 1.5; 
    
    // Cold start setup
    double current_t = 0.0;
    for(int k=0; k<=N; ++k) {
        if(k > 0) current_t += dts[k-1];
        double x_ref = current_t * 5.0; 
        double params[] = { 5.0, x_ref, 0.0, obs_x, obs_y, obs_rad };
        if(k < N) solver.traj[k].u.setZero();
        for(int i=0; i<6; ++i) solver.traj[k].p(i) = params[i];
    }
    solver.traj[0].x.setZero(); 
    solver.rollout_dynamics();

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve();
    auto end = std::chrono::high_resolution_clock::now();
    
    BenchmarkResult res;
    res.name = name;
    res.iters = solver.current_iter;
    res.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    res.derivatives_ms = solver.timer.times["Derivatives"];
    res.linear_solve_ms = solver.timer.times["Linear Solve"];
    res.line_search_ms = solver.timer.times["Line Search"];
    
    // Check convergence
    double max_viol = 0.0;
    for(const auto& kp : solver.traj) {
        for(int i=0; i<CarModel::NC; ++i) {
            if(kp.g_val(i) > max_viol) max_viol = kp.g_val(i);
        }
    }
    res.converged = (solver.mu <= config.mu_min * 10.0) && (max_viol < 1e-3);
    
    // Compute final cost
    double cost = 0.0;
    for(const auto& kp : solver.traj) cost += kp.cost;
    res.final_cost = cost;
    res.final_viol = max_viol;
    
    return res;
}

int main() {
    std::vector<BenchmarkResult> results;
    
    // Base Config
    SolverConfig base;
    base.integrator = IntegratorType::RK4_EXPLICIT;
    base.default_dt = 0.1;
    base.max_iters = 100;
    base.tol_con = 1e-4;
    
    // 1. Monotone + Merit + Regularization (Classic)
    SolverConfig c1 = base;
    c1.barrier_strategy = BarrierStrategy::MONOTONE;
    c1.line_search_type = LineSearchType::MERIT;
    c1.inertia_strategy = InertiaStrategy::REGULARIZATION;
    results.push_back(run_test("Monotone/Merit/Reg", c1));
    
    // 2. Adaptive + Filter + IgnoreSingular (Modern)
    SolverConfig c2 = base;
    c2.barrier_strategy = BarrierStrategy::ADAPTIVE;
    c2.line_search_type = LineSearchType::FILTER;
    c2.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR;
    results.push_back(run_test("Adaptive/Filter/Ignore", c2));
    
    // 3. Mehrotra + Filter + IgnoreSingular (Fastest?)
    SolverConfig c3 = base;
    c3.barrier_strategy = BarrierStrategy::MEHROTRA;
    c3.line_search_type = LineSearchType::FILTER;
    c3.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR;
    results.push_back(run_test("Mehrotra/Filter/Ignore", c3));

    // Print Table
    std::cout << "\n================================ BENCHMARK RESULTS ================================\n";
    std::cout << std::left 
              << std::setw(25) << "Config" 
              << std::setw(8) << "Iters" 
              << std::setw(10) << "Time(ms)" 
              << std::setw(10) << "Deriv(ms)" 
              << std::setw(10) << "Solve(ms)" 
              << std::setw(10) << "LS(ms)" 
              << std::setw(12) << "Cost" 
              << std::setw(10) << "Viol"
              << "Status" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for(const auto& r : results) {
        std::cout << std::left 
                  << std::setw(25) << r.name 
                  << std::setw(8) << r.iters 
                  << std::fixed << std::setprecision(1) << std::setw(10) << r.total_time_ms
                  << std::setw(10) << r.derivatives_ms
                  << std::setw(10) << r.linear_solve_ms
                  << std::setw(10) << r.line_search_ms
                  << std::scientific << std::setprecision(2) << std::setw(12) << r.final_cost
                  << std::setw(10) << r.final_viol
                  << (r.converged ? "OK" : "FAIL") << "\n";
    }
    std::cout << "===================================================================================\n";
    
    return 0;
}

