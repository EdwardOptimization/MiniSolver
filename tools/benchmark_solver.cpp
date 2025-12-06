#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "model/car_model.h"
#include "solver/solver.h"

using namespace minisolver;

struct BenchmarkResult {
    std::string name;
    int iters;
    double time_avg;
    double time_min;
    double time_max;
    double deriv_avg;
    double solve_avg;
    double ls_avg;
    bool converged;
    double final_cost;
    double final_viol;
};

BenchmarkResult run_test(const std::string& name, SolverConfig config) {
    int N = 60;
    config.verbose = false; // Mute output for benchmark
    config.debug_mode = false;
    
    // Setup Scenario Params once
    std::vector<double> dts(N);
    for(int k=0; k<N; ++k) dts[k] = (k < 20) ? 0.05 : 0.2;
    double obs_x = 12.0; double obs_y = 0.0; double obs_rad = 1.5; 

    // --- Benchmarking Params ---
    const int NUM_RUNS = 100;
    const int WARMUP_RUNS = 10;
    
    std::vector<double> total_times;
    total_times.reserve(NUM_RUNS);
    
    double sum_deriv = 0;
    double sum_solve = 0;
    double sum_ls = 0;
    int last_iter_count = 0;
    bool last_converged = false;
    double last_cost = 0;
    double last_viol = 0;

    for(int run = 0; run < WARMUP_RUNS + NUM_RUNS; ++run) {
        // Re-initialize solver to simulate fresh start (Cold Start)
        // Use MAX_N = 100 for static allocation
        PDIPMSolver<CarModel, 100> solver(N, Backend::CPU_SERIAL, config);
        solver.set_dt(dts);
        
        // Cold start setup
        double current_t = 0.0;
        for(int k=0; k<=N; ++k) {
            if(k > 0) current_t += dts[k-1];
            double x_ref = current_t * 5.0; 
            double params[] = { 5.0, x_ref, 0.0, obs_x, obs_y, obs_rad };
            if(k < N) solver.get_traj()[k].u.setZero();
            for(int i=0; i<6; ++i) solver.get_traj()[k].p(i) = params[i];
        }
        solver.get_traj()[0].x.setZero(); 
        solver.rollout_dynamics();

        auto start = std::chrono::high_resolution_clock::now();
        solver.solve();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (run >= WARMUP_RUNS) {
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_times.push_back(ms);
            
            // Accumulate component times
            sum_deriv += solver.timer.times["Derivatives"];
            sum_solve += solver.timer.times["Linear Solve"];
            sum_ls    += solver.timer.times["Line Search"];
            
            // Capture last run stats
            if (run == WARMUP_RUNS + NUM_RUNS - 1) {
                last_iter_count = solver.current_iter;
                
                double max_viol = 0.0;
                for(const auto& kp : solver.get_traj()) {
                    for(int i=0; i<CarModel::NC; ++i) {
                        if(kp.g_val(i) > max_viol) max_viol = kp.g_val(i);
                    }
                }
                last_converged = (solver.mu <= config.mu_min * 10.0) && (max_viol < 1e-3);
                
                double cost = 0.0;
                for(const auto& kp : solver.get_traj()) cost += kp.cost;
                last_cost = cost;
                last_viol = max_viol;
            }
        }
    }
    
    BenchmarkResult res;
    res.name = name;
    res.iters = last_iter_count;
    
    double sum_total = std::accumulate(total_times.begin(), total_times.end(), 0.0);
    res.time_avg = sum_total / NUM_RUNS;
    res.time_min = *std::min_element(total_times.begin(), total_times.end());
    res.time_max = *std::max_element(total_times.begin(), total_times.end());
    
    res.deriv_avg = sum_deriv / NUM_RUNS;
    res.solve_avg = sum_solve / NUM_RUNS;
    res.ls_avg    = sum_ls / NUM_RUNS;
    
    res.converged = last_converged;
    res.final_cost = last_cost;
    res.final_viol = last_viol;
    
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
    std::cout << "\n=========================================== BENCHMARK RESULTS (Avg of 100 Runs) ===========================================\n";
    std::cout << std::left 
              << std::setw(25) << "Config" 
              << std::setw(8) << "Iters" 
              << std::setw(12) << "Time(ms)" 
              << std::setw(12) << "Min/Max"
              << std::setw(12) << "Deriv(ms)" 
              << std::setw(12) << "Solve(ms)" 
              << std::setw(12) << "LS(ms)" 
              << std::setw(12) << "Cost" 
              << std::setw(10) << "Viol"
              << "Status" << "\n";
    std::cout << std::string(130, '-') << "\n";
    
    for(const auto& r : results) {
        std::stringstream range_ss;
        range_ss << std::fixed << std::setprecision(1) << r.time_min << "/" << r.time_max;
        
        std::cout << std::left 
                  << std::setw(25) << r.name 
                  << std::setw(8) << r.iters 
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.time_avg
                  << std::setw(12) << range_ss.str()
                  << std::setw(12) << r.deriv_avg
                  << std::setw(12) << r.solve_avg
                  << std::setw(12) << r.ls_avg
                  << std::scientific << std::setprecision(2) << std::setw(12) << r.final_cost
                  << std::setw(10) << r.final_viol
                  << (r.converged ? "OK" : "FAIL") << "\n";
    }
    std::cout << "===========================================================================================================================\n";
    
    return 0;
}
