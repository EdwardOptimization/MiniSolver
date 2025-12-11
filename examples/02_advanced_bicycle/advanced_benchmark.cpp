#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <algorithm>
#include <numeric>

// Include the newly generated extended model
#include "generated/bicycleextmodel.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

// Configuration for this benchmark
struct ExtConfig {
    static const int N = 50;
    static constexpr double TARGET_V = 10.0;
    static constexpr double OBS_X = 15.0;
    static constexpr double OBS_Y = 0.5; // Slight offset to force avoidance
    static constexpr double OBS_RAD = 1.5;
    static constexpr double CAR_RAD = 1.0;
    
    // Weights
    static constexpr double W_POS = 10.0;
    static constexpr double W_VEL = 1.0;
    static constexpr double W_THETA = 0.1;
    static constexpr double W_KAPPA = 0.1;
    static constexpr double W_A = 0.1;
    static constexpr double W_DKAPPA = 1.0;
    static constexpr double W_JERK = 1.0;
};

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
    int N = ExtConfig::N;
    config.print_level = PrintLevel::NONE; 
    
    std::vector<double> dts(N);
    // Simple uniform time steps
    for(int k=0; k<N; ++k) dts[k] = 0.05;

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

    MiniSolver<BicycleExtModel, 50> solver(N, Backend::CPU_SERIAL, config);
    for(int run = 0; run < WARMUP_RUNS + NUM_RUNS; ++run) {
        solver.reset(ResetOption::FULL);
        solver.set_dt(dts);

        double current_t = 0.0;
        for(int k=0; k<=N; ++k) {
            if(k > 0) current_t += dts[k-1];
            double x_ref = current_t * ExtConfig::TARGET_V; 
            
            // Intelligent Reference Generation (Match Debug Setup)
            double y_ref_val = 0.0;
            if (x_ref > ExtConfig::OBS_X - 10.0 && x_ref < ExtConfig::OBS_X + 10.0) {
                y_ref_val = -2.5; // Guide BELOW obstacle
            }

            // Set Parameters
            solver.set_parameter(k, "v_ref", ExtConfig::TARGET_V);
            solver.set_parameter(k, "x_ref", x_ref);
            solver.set_parameter(k, "y_ref", y_ref_val);
            
            solver.set_parameter(k, "obs_x", ExtConfig::OBS_X);
            solver.set_parameter(k, "obs_y", ExtConfig::OBS_Y);
            solver.set_parameter(k, "obs_rad", ExtConfig::OBS_RAD);
            solver.set_parameter(k, "L", 2.5); // Unused in this model dynamics but present
            solver.set_parameter(k, "car_rad", ExtConfig::CAR_RAD);
            
            solver.set_parameter(k, "w_pos", ExtConfig::W_POS);
            solver.set_parameter(k, "w_vel", ExtConfig::W_VEL);
            solver.set_parameter(k, "w_theta", ExtConfig::W_THETA);
            solver.set_parameter(k, "w_kappa", ExtConfig::W_KAPPA);
            solver.set_parameter(k, "w_a", ExtConfig::W_A);
            solver.set_parameter(k, "w_dkappa", ExtConfig::W_DKAPPA);
            solver.set_parameter(k, "w_jerk", ExtConfig::W_JERK);
            
            // Warm start controls
            if(k < N) {
                solver.set_control_guess(k, "dkappa", 0.0);
                solver.set_control_guess(k, "jerk", 0.0);
            }

            // [Fix] Provide smart state guess to avoid obstacle
            solver.set_state_guess(k, "x", x_ref);
            solver.set_state_guess(k, "y", y_ref_val);
            solver.set_state_guess(k, "v", ExtConfig::TARGET_V);
        }
        
        // Initial State: x, y, theta, kappa, v, a
        solver.set_initial_state("x", 0.0);
        solver.set_initial_state("y", 0.0);
        solver.set_initial_state("theta", 0.0);
        solver.set_initial_state("kappa", 0.0);
        solver.set_initial_state("v", 1.0); // Non-zero v to match debug
        solver.set_initial_state("a", 0.0);
        
        // solver.rollout_dynamics(); // Removed to preserve smart guess
        // solver.rollout_dynamics();

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
                    for(int i=0; i<BicycleExtModel::NC; ++i) {
                        double val = std::abs(solver.get_constraint_val(k, i) + solver.trajectory[k].s(i));
                        if(val > max_viol) max_viol = val;
                    }
                }
                last_converged = (status == SolverStatus::SOLVED || status == SolverStatus::FEASIBLE);
                
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
    base.max_iters = 300;
    base.tol_con = 0.05; // Relaxed for challenging obstacle scenario
    base.reg_init = 1e-4; 
    base.reg_min = 1e-8;

    // Robustness Settings
    base.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR;
    base.enable_feasibility_restoration = true;
    base.enable_slack_reset = true;
    base.slack_reset_trigger = 0.05;

    // 1. Adaptive + Filter (Best Performer usually)
    SolverConfig c1 = base;
    c1.barrier_strategy = BarrierStrategy::ADAPTIVE;
    c1.line_search_type = LineSearchType::FILTER;
    c1.filter_gamma_theta = 1e-6; 
    c1.filter_gamma_phi = 1e-6;
    c1.mu_init = 0.1;
    // Relax tolerances for initial feasibility
    c1.tol_con = 1e-2;
    results.push_back(run_test("ExtBicycle (Adaptive)", c1));
    
    // 2. Monotone + Merit
    SolverConfig c2 = base;
    c2.barrier_strategy = BarrierStrategy::MONOTONE;
    c2.line_search_type = LineSearchType::MERIT;
    c2.tol_con = 1e-2;
    results.push_back(run_test("ExtBicycle (Monotone)", c2));

    // 3. Mehrotra (The "Debug" Config)
    SolverConfig c3 = base;
    c3.barrier_strategy = BarrierStrategy::MEHROTRA; 
    c3.line_search_type = LineSearchType::FILTER;
    c3.mu_init = 1e-3; // [Fix] Start with smaller mu since we have good guess
    c3.tol_con = 1e-2;
    // [Fix] Enable iterative refinement for Mehrotra to handle ill-conditioning near solution
    c3.enable_iterative_refinement = true;
    c3.inertia_max_retries = 2; // Allow some retries if factorization fails
    c3.filter_gamma_theta = 1e-5; // Relax filter slightly
    results.push_back(run_test("ExtBicycle (Mehrotra)", c3));
    
    // Print Table
    std::cout << "\n========================================= EXTENDED BICYCLE (NX=6, NU=2) =========================================\n";
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
    std::cout << "=================================================================================================================\n";
    
    return 0;
}

