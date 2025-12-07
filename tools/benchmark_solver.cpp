#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "model/car_model.h"
#include "model/scenario.h" // [NEW] Shared Scenario Config
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
    int N = ScenarioConfig::N;
    config.print_level = PrintLevel::NONE; 
    
    std::vector<double> dts(N);
    for(int k=0; k<N; ++k) dts[k] = (k < 20) ? 0.05 : 0.2;

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
        PDIPMSolver<CarModel, 100> solver(N, Backend::CPU_SERIAL, config);
        solver.set_dt(dts);
        
        double current_t = 0.0;
        for(int k=0; k<=N; ++k) {
            if(k > 0) current_t += dts[k-1];
            double x_ref = current_t * ScenarioConfig::TARGET_V; 
            
            if(k < N) {
                solver.set_control_guess(k, "acc", 0.0);
                solver.set_control_guess(k, "steer", 0.0);
            }
            
            solver.set_parameter(k, "v_ref", ScenarioConfig::TARGET_V);
            solver.set_parameter(k, "x_ref", x_ref);
            solver.set_parameter(k, "y_ref", 0.0);
            solver.set_parameter(k, "obs_x", ScenarioConfig::OBS_X);
            solver.set_parameter(k, "obs_y", ScenarioConfig::OBS_Y);
            solver.set_parameter(k, "obs_rad", ScenarioConfig::OBS_RAD);
            solver.set_parameter(k, "L", ScenarioConfig::CAR_L);
            solver.set_parameter(k, "car_rad", ScenarioConfig::CAR_RAD);
            
            solver.set_parameter(k, "w_pos", ScenarioConfig::W_POS);
            solver.set_parameter(k, "w_vel", ScenarioConfig::W_VEL);
            solver.set_parameter(k, "w_theta", ScenarioConfig::W_THETA);
            solver.set_parameter(k, "w_acc", ScenarioConfig::W_ACC);
            solver.set_parameter(k, "w_steer", ScenarioConfig::W_STEER);
        }
        solver.set_initial_state("x", 0.0);
        solver.set_initial_state("y", 0.0);
        solver.set_initial_state("theta", 0.0);
        solver.set_initial_state("v", 0.0);
        
        solver.rollout_dynamics();

        auto start = std::chrono::high_resolution_clock::now();
        SolverStatus status = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (run >= WARMUP_RUNS) {
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_times.push_back(ms);
            
            sum_deriv += solver.timer.times["Derivatives"];
            sum_solve += solver.timer.times["Linear Solve"];
            sum_ls    += solver.timer.times["Line Search"];
            
            if (run == WARMUP_RUNS + NUM_RUNS - 1) {
                last_iter_count = solver.current_iter;
                
                double max_viol = 0.0;
                for(int k=0; k<=solver.N; ++k) {
                    for(int i=0; i<CarModel::NC; ++i) {
                        double val = solver.get_constraint_val(k, i);
                        if(val > max_viol) max_viol = val;
                    }
                }
                // Updated convergence check to use SolverStatus
                last_converged = (status == SolverStatus::SOLVED);
                
                double cost = 0.0;
                for(int k=0; k<=solver.N; ++k) cost += solver.get_stage_cost(k);
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
    base.max_iters = 200;
    base.tol_con = 1e-4;
    base.print_level = PrintLevel::NONE; // Ensure clean benchmark output
    base.reg_init = 1e-6; 
    base.reg_min = 1e-9;
    
    // Shared Robustness Settings
    base.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR; // Best performer
    base.enable_feasibility_restoration = true;
    base.enable_slack_reset = true;
    base.slack_reset_trigger = 0.1; // Trigger reset if alpha < 0.1

    // 1. Current: Mehrotra + Filter (Aggressive)
    SolverConfig c1 = base;
    c1.barrier_strategy = BarrierStrategy::MEHROTRA;
    c1.line_search_type = LineSearchType::FILTER;
    results.push_back(run_test("Mehrotra + Filter", c1));
    
    // 2. Classic: Monotone + Merit (Conservative but Robust)
    SolverConfig c2 = base;
    c2.barrier_strategy = BarrierStrategy::MONOTONE;
    c2.line_search_type = LineSearchType::MERIT;
    c2.mu_linear_decrease_factor = 0.2; // Standard reduction
    c2.barrier_tolerance_factor = 10.0;
    results.push_back(run_test("Monotone + Merit", c2));
    
    // 3. Hybrid: Adaptive + Filter (Balanced)
    SolverConfig c3 = base;
    c3.barrier_strategy = BarrierStrategy::ADAPTIVE;
    c3.line_search_type = LineSearchType::FILTER;
    c3.mu_init = 0.1;
    results.push_back(run_test("Adaptive + Filter", c3));

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
