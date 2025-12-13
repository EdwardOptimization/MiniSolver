#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "generated/linesearchbenchmarkmodel.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

// Simple result structure for benchmark
struct BenchmarkResult {
    std::string name;
    bool success;
    int iters;
    double time_ms;
    double cost;
    double viol;
};

void setup_scenario(MiniSolver<LineSearchBenchmarkModel, 100>& solver) {
    int N = solver.N;

    // Rosenbrock classic start point is usually around (-1.2, 1).
    // We set up a trajectory optimization problem that traverses the Rosenbrock valley.
    // Start: (-1.0, -0.5)
    // Target (implicit in cost): (1.0, 1.0)
    solver.set_initial_state("x", -1.0);
    solver.set_initial_state("y", -0.5);

    // Parameters for Rosenbrock function: f = (a-x)^2 + b(y-x^2)^2
    // Standard values: a=1, b=100
    double a_param = 1.0;
    double b_param = 100.0;
    double w_u = 1e-3; // Small regularization on control

    for(int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "a_param", a_param);
        solver.set_parameter(k, "b_param", b_param);
        solver.set_parameter(k, "w_u", w_u);
    }

    // Initial guess: constant velocity towards target
    // Target is roughly (1, 1). Delta = (2, 1.5).
    // 40 steps, dt=0.05 -> T=2.0s.
    // Avg vel = (1, 0.75)
    for(int k = 0; k < N; ++k) {
        solver.set_control_guess(k, "vx", 1.0);
        solver.set_control_guess(k, "vy", 0.75);
    }
}

void benchmark_simd_modes() {
    std::cout << "=== Line Search SIMD Benchmark ===\n";
    std::cout << "Problem: Rosenbrock Valley Trajectory Optimization\n";
    std::cout << "This problem features a narrow, curved valley that requires precise line search.\n\n";

    // Test configurations
    std::vector<std::pair<std::string, SimdMode>> configs = {
        {"DISABLED", SimdMode::DISABLED},
        {"ALL_ITERATIONS", SimdMode::ALL_ITERATIONS},
        {"SKIP_FIRST", SimdMode::SKIP_FIRST}
    };

    std::vector<BenchmarkResult> results;

    // Run multiple trials for statistical significance
    const int num_trials = 5; 

    // First test if the solver can solve the problem at all
    std::cout << "Testing basic solvability...\n";
    {
        SolverConfig basic_config;
        basic_config.integrator = IntegratorType::RK4_EXPLICIT;
        basic_config.barrier_strategy = BarrierStrategy::MEHROTRA;
        basic_config.line_search_type = LineSearchType::FILTER; // Filter is usually robust
        basic_config.tol_con = 1e-3;
        basic_config.max_iters = 100;
        basic_config.print_level = PrintLevel::INFO;

        MiniSolver<LineSearchBenchmarkModel, 100> test_solver(100, Backend::CPU_SERIAL, basic_config);
        
        // Set time steps (T=2.0s)
        std::vector<double> dts(100, 0.02);
        test_solver.set_dt(dts);
        
        setup_scenario(test_solver);

        auto start = std::chrono::high_resolution_clock::now();
        SolverStatus status = test_solver.solve();
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        std::cout << "Basic test: " << (status == SolverStatus::SOLVED ? "SUCCESS" : "FAILED")
                  << " in " << time_ms << " ms\n\n";
    }

    for(auto& [name, mode] : configs) {
        std::cout << "Testing SIMD mode: " << name << "\n";

        std::vector<double> times;
        std::vector<int> iters;

        for(int trial = 0; trial < num_trials; ++trial) {
            SolverConfig config;
            config.line_search_simd_mode = mode;
            config.integrator = IntegratorType::RK4_EXPLICIT;
            config.barrier_strategy = BarrierStrategy::MEHROTRA;
            config.line_search_type = LineSearchType::FILTER; // Use Filter method
            config.tol_con = 1e-3;
            config.max_iters = 200; 
            config.line_search_max_iters = 20; // Allow deep line searches
            config.print_level = PrintLevel::NONE;

            MiniSolver<LineSearchBenchmarkModel, 100> solver(100, Backend::CPU_SERIAL, config);

            std::vector<double> dts(100, 0.02);
            solver.set_dt(dts);

            setup_scenario(solver);

            auto start = std::chrono::high_resolution_clock::now();
            SolverStatus status = solver.solve();
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

            if(status == SolverStatus::SOLVED) {
                times.push_back(time_ms);
                iters.push_back(1); 
            }
        }

        if(!times.empty()) {
            double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            results.push_back({name, true, 0, avg_time, 0.0, 0.0});

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  ✓ Average solve time: " << avg_time << " ms\n";
            std::cout << "  ✓ Success rate: " << times.size() << "/" << num_trials << "\n\n";
        } else {
            results.push_back({name, false, 0, 0.0, 0.0, 0.0});
            std::cout << "  ✗ Failed to solve\n\n";
        }
    }

    // Print comparison table
    std::cout << "=== Performance Comparison ===\n";
    std::cout << std::left << std::setw(15) << "Mode"
              << std::right << std::setw(8) << "Time(ms)"
              << std::setw(8) << "Success"
              << std::setw(10) << "Trials" << "\n";
    std::cout << std::string(41, '-') << "\n";

    for(auto& result : results) {
        if(result.success) {
            std::cout << std::left << std::setw(15) << result.name
                      << std::right << std::fixed << std::setprecision(1)
                      << std::setw(8) << result.time_ms
                      << std::setw(8) << "YES"
                      << std::setw(10) << num_trials << "\n";
        } else {
            std::cout << std::left << std::setw(15) << result.name
                      << std::right << std::setw(8) << "FAILED"
                      << std::setw(8) << "NO"
                      << std::setw(10) << "0" << "\n";
        }
    }

    // Calculate speedup
    auto baseline = std::find_if(results.begin(), results.end(),
                                [](const BenchmarkResult& r){ return r.name == "DISABLED"; });
    if(baseline != results.end() && baseline->success) {
        std::cout << "\n=== Speedup Analysis ===\n";
        for(auto& result : results) {
            if(result.success && result.name != "DISABLED") {
                double speedup = baseline->time_ms / result.time_ms;
                std::cout << result.name << " speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
            }
        }
    }
}

int main(int /*argc*/, char** /*argv*/) {
    srand(42); 

    SimdLevel level = SimdDetector::detect_capability();
    std::cout << "Detected SIMD capability: ";
    switch(level) {
        case SimdLevel::NONE: std::cout << "NONE"; break;
        case SimdLevel::AVX2: std::cout << "AVX2"; break;
        case SimdLevel::AVX512: std::cout << "AVX512"; break;
    }
    std::cout << std::endl << std::endl;

    benchmark_simd_modes();

    return 0;
}
