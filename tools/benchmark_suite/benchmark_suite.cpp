#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

// Assumes CarModel is generated in include/model/
#include "../../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

// --- Benchmark Configuration Structures ---

struct BenchmarkCase {
    std::string name;
    std::string description;
    SolverConfig config;
};

struct Result {
    std::string name;
    bool success;
    int iters;
    double time_ms;
    double cost;
    double viol;
};

// --- Scenario Setup ---
// Defines a standard obstacle avoidance and lane change scenario
void setup_scenario(MiniSolver<CarModel, 60>& solver) {
    int N = solver.N;
    double target_v = 5.0;
    
    // Initial State (Stopped at origin)
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("v", 0.0);

    // Horizon Parameters
    for(int k=0; k<=N; ++k) {
        double t = k * solver.config.default_dt;
        
        // Target: Move along X-axis at 5 m/s
        solver.set_parameter(k, "v_ref", target_v);
        solver.set_parameter(k, "x_ref", t * target_v);
        solver.set_parameter(k, "y_ref", 0.0); 
        
        // Obstacle: Placed at x=10.0 to force deviation
        solver.set_parameter(k, "obs_x", 10.0);
        solver.set_parameter(k, "obs_y", 0.0);
        solver.set_parameter(k, "obs_rad", 1.5);
        
        // Vehicle Dimensions
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        
        // Tuning Weights
        solver.set_parameter(k, "w_pos", 1.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_theta", 0.1);
        solver.set_parameter(k, "w_acc", 0.1);
        solver.set_parameter(k, "w_steer", 1.0);
        
        // Cold Start Guess (Zero controls)
        if (k < N) {
            solver.set_control_guess(k, "acc", 0.0);
            solver.set_control_guess(k, "steer", 0.0);
        }
    }
}

// --- Runner ---
Result run_case(const BenchmarkCase& test_case) {
    const int N = 50;
    const int NUM_RUNS = 100;
    const int WARMUP = 10;
    // Instantiate Solver
    MiniSolver<CarModel, 60> solver(N, Backend::CPU_SERIAL, test_case.config);

    setup_scenario(solver);
    solver.rollout_dynamics(); // Initial dynamics propagation
    
    std::vector<double> times;
    times.reserve(NUM_RUNS);
    
    Result res;
    res.name = test_case.name;
    
    for(int i=0; i < WARMUP + NUM_RUNS; ++i) {
        solver.reset(ResetOption::FULL);
        solver.set_dt(0.1);

        setup_scenario(solver);
        solver.rollout_dynamics(); // Initial dynamics propagation
        
        auto start = std::chrono::high_resolution_clock::now();
        SolverStatus status = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (i >= WARMUP) {
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            
            // Capture stats from the final run
            if (i == WARMUP + NUM_RUNS - 1) {
                res.success = (status == SolverStatus::SOLVED || status == SolverStatus::FEASIBLE);
                res.iters = solver.current_iter;
                
                // Compute Metrics
                double max_viol = 0.0;
                double total_cost = 0.0;
                auto& traj = solver.trajectory.active();
                for(int k=0; k<=N; ++k) {
                    total_cost += traj[k].cost;
                    for(int j=0; j<CarModel::NC; ++j) {
                        double v = traj[k].g_val(j); // Violations are g(x) > 0
                        if (v > max_viol) max_viol = v;
                    }
                }
                res.cost = total_cost;
                res.viol = max_viol;
            }
        }
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    res.time_ms = sum / NUM_RUNS;
    return res;
}

int main() {
    std::vector<BenchmarkCase> cases;
    
    // 1. TURBO_MPC
    // Target: Embedded MCUs, <100Hz loops.
    // Config: Euler Integration, Adaptive Barrier (Fast Drop), Loose Tolerance.
    SolverConfig c1;
    c1.integrator = IntegratorType::EULER_EXPLICIT;
    c1.barrier_strategy = BarrierStrategy::ADAPTIVE;
    c1.line_search_type = LineSearchType::FILTER;
    c1.tol_con = 1e-2;
    c1.max_iters = 20;
    c1.print_level = PrintLevel::NONE;
    cases.push_back({"TURBO_MPC", "Euler + Adaptive + Loose Tol", c1});

    // 2. BALANCED_RT
    // Target: General Robotics (UGV/UAV), 50Hz loops.
    // Config: RK2 (Midpoint), Mehrotra (Fewer Iters), Standard Tol.
    SolverConfig c2;
    c2.integrator = IntegratorType::RK2_EXPLICIT;
    c2.barrier_strategy = BarrierStrategy::MEHROTRA; 
    c2.line_search_type = LineSearchType::FILTER;
    c2.tol_con = 1e-3;
    c2.print_level = PrintLevel::NONE;
    cases.push_back({"BALANCED_RT", "RK2 + Mehrotra + Filter", c2});

    // 3. QUALITY_PLANNER (Default)
    // Target: Autonomous Driving Planning, high fidelity.
    // Config: RK4, Mehrotra, High Precision.
    SolverConfig c3;
    c3.integrator = IntegratorType::RK4_EXPLICIT;
    c3.barrier_strategy = BarrierStrategy::MEHROTRA;
    c3.line_search_type = LineSearchType::FILTER;
    c3.tol_con = 1e-4;
    c3.print_level = PrintLevel::NONE;
    cases.push_back({"QUALITY_PLANNER", "RK4 + Mehrotra + High Prec", c3});

    // 4. CLASSIC_STABLE
    // Target: Research comparison, older reliable methods.
    // Config: RK4, Monotone Barrier, Merit Function.
    SolverConfig c4;
    c4.integrator = IntegratorType::RK4_EXPLICIT;
    c4.barrier_strategy = BarrierStrategy::MONOTONE;
    c4.line_search_type = LineSearchType::MERIT;
    c4.tol_con = 1e-4;
    c4.print_level = PrintLevel::NONE;
    cases.push_back({"CLASSIC_STABLE", "RK4 + Monotone + Merit", c4});

    // 5. ROBUST_OBSTACLE
    // Target: Complex environments, narrow passages.
    // Config: SOC (Second Order Correction) enabled, Restoration enabled.
    SolverConfig c5 = c3; // Base on Quality
    c5.enable_soc = true;
    c5.enable_feasibility_restoration = true;
    c5.enable_slack_reset = true;
    cases.push_back({"ROBUST_OBS", "Quality + SOC + Restoration", c5});

    // 6. HIGH_PRECISION
    // Target: Ground Truth generation, offline trajectory optimization.
    // Config: Exact Hessian, Very tight tolerance.
    SolverConfig c6 = c3;
    c6.hessian_approximation = HessianApproximation::EXACT;
    c6.tol_con = 1e-6;
    c6.tol_dual = 1e-6;
    cases.push_back({"HIGH_PRECISION", "RK4 + Exact Hessian + 1e-6", c6});

    // 7. AGGRESSIVE_RACING
    // Target: Racing scenarios where approximate feasibility is acceptable for speed.
    // Config: Aggressive Barrier reduction, Ignore Singularities.
    SolverConfig c7 = c3;
    c7.enable_aggressive_barrier = true;
    c7.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR;
    cases.push_back({"AGGRESSIVE", "Mehrotra + Aggressive Mu", c7});

    // 8. SQP_RTI
    // Target: >1000Hz Control Loops.
    // Config: Single Iteration per call.
    SolverConfig c8;
    c8.integrator = IntegratorType::EULER_EXPLICIT;
    c8.max_iters = 1;
    c8.enable_rti = true;
    c8.print_level = PrintLevel::NONE;
    cases.push_back({"SQP_RTI", "Euler + Single Iteration", c8});

    // --- Report Generation ---
    std::cout << "\n==================================================================================================\n";
    std::cout << std::left << std::setw(18) << "Archetype" 
              << std::setw(35) << "Configuration Summary" 
              << std::setw(12) << "Time(ms)" 
              << std::setw(8) << "Iters" 
              << std::setw(10) << "Status" 
              << std::setw(12) << "Cost"
              << std::setw(10) << "MaxViol" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    for(const auto& c : cases) {
        Result r = run_case(c);
        
        std::string status_str = r.success ? "OK" : "FAIL";
        if (c.config.enable_rti) status_str = "RTI"; // RTI always "completes"
        
        std::cout << std::left << std::setw(18) << r.name 
                  << std::setw(35) << c.description 
                  << std::fixed << std::setprecision(3) << std::setw(12) << r.time_ms 
                  << std::setw(8) << r.iters 
                  << std::setw(10) << status_str 
                  << std::scientific << std::setprecision(2) << std::setw(12) << r.cost
                  << std::scientific << std::setprecision(1) << std::setw(10) << r.viol << "\n";
    }
    std::cout << "==================================================================================================\n";

    return 0;
}