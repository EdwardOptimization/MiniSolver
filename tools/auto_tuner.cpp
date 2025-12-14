#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cmath>
#include <functional>

#include "../examples/01_car_tutorial/generated/car_model.h"

#include "minisolver/solver/solver.h"

using namespace minisolver;

// --- Tuning Search Space ---

struct TunableParams {
    // Categorical
    IntegratorType integrator;
    BarrierStrategy barrier;
    LineSearchType ls_type;
    InertiaStrategy inertia;
    bool enable_restoration;
    bool enable_slack_reset;

    // Continuous (Log Scale)
    double mu_init;
    double reg_init;
    double tol_con;
    double slack_reset_trigger;

    // Identification
    int id;
};

std::string integrator_to_string(IntegratorType t) {
    switch(t) {
        case IntegratorType::EULER_EXPLICIT: return "EULER";
        case IntegratorType::RK2_EXPLICIT: return "RK2";
        case IntegratorType::RK4_EXPLICIT: return "RK4";
        case IntegratorType::EULER_IMPLICIT: return "EULER_IMP";
        case IntegratorType::RK2_IMPLICIT: return "RK2_IMP";
        default: return "UNKNOWN";
    }
}

std::string barrier_to_string(BarrierStrategy t) {
    switch(t) {
        case BarrierStrategy::MONOTONE: return "MONOTONE";
        case BarrierStrategy::MEHROTRA: return "MEHROTRA";
        case BarrierStrategy::ADAPTIVE: return "ADAPTIVE";
        default: return "UNKNOWN";
    }
}

std::string ls_to_string(LineSearchType t) {
    return (t == LineSearchType::FILTER) ? "FILTER" : "MERIT";
}

std::string inertia_to_string(InertiaStrategy t) {
    return (t == InertiaStrategy::IGNORE_SINGULAR) ? "IGNORE" : "REG";
}

void print_config(const TunableParams& p) {
    std::cout << "  Integrator:  " << integrator_to_string(p.integrator) << "\n";
    std::cout << "  Barrier:     " << barrier_to_string(p.barrier) << "\n";
    std::cout << "  LineSearch:  " << ls_to_string(p.ls_type) << "\n";
    std::cout << "  Inertia:     " << inertia_to_string(p.inertia) << "\n";
    std::cout << "  Restoration: " << (p.enable_restoration ? "ON" : "OFF") << "\n";
    std::cout << "  SlackReset:  " << (p.enable_slack_reset ? "ON" : "OFF") << "\n";
    std::cout << "  Mu Init:     " << p.mu_init << "\n";
    std::cout << "  Reg Init:    " << p.reg_init << "\n";
    std::cout << "  Tol Con:     " << p.tol_con << "\n";
}

SolverConfig to_solver_config(const TunableParams& p) {
    SolverConfig c;
    c.integrator = p.integrator;
    c.barrier_strategy = p.barrier;
    c.line_search_type = p.ls_type;
    c.inertia_strategy = p.inertia;
    c.enable_feasibility_restoration = p.enable_restoration;
    c.enable_slack_reset = p.enable_slack_reset;
    c.mu_init = p.mu_init;
    c.reg_init = p.reg_init;
    c.tol_con = p.tol_con;
    c.slack_reset_trigger = p.slack_reset_trigger;
    
    // Fixed / Derived
    c.print_level = PrintLevel::NONE;
    c.max_iters = 100;
    c.default_dt = 0.1;
    
    // Reasonable defaults for others
    c.mu_final = 1e-6;
    c.reg_min = 1e-9;
    
    return c;
}

// --- Random Generation ---

class ConfigGenerator {
    std::mt19937 rng;
public:
    ConfigGenerator() : rng(std::random_device{}()) {}

    TunableParams generate(int id) {
        TunableParams p;
        p.id = id;
        
        // Random Enums
        std::uniform_int_distribution<int> dist_int_3(0, 2); // 3 options
        std::uniform_int_distribution<int> dist_int_2(0, 1); // 2 options
        std::uniform_int_distribution<int> dist_bool(0, 1);

        // Integrator: RK4, RK2, EULER (0,1,2 in enum order roughly? Check SolverOptions.h)
        // Order: EULER_EXPLICIT=0, EULER_IMPLICIT=1, RK2_EXPLICIT=2, RK2_IMPLICIT=3, RK4_EXPLICIT=4...
        // Let's pick explicitly
        int r_int = std::uniform_int_distribution<int>(0, 2)(rng);
        if(r_int == 0) p.integrator = IntegratorType::EULER_EXPLICIT;
        else if(r_int == 1) p.integrator = IntegratorType::RK2_EXPLICIT;
        else p.integrator = IntegratorType::RK4_EXPLICIT;

        int r_bar = dist_int_3(rng);
        if(r_bar == 0) p.barrier = BarrierStrategy::MONOTONE;
        else if(r_bar == 1) p.barrier = BarrierStrategy::MEHROTRA;
        else p.barrier = BarrierStrategy::ADAPTIVE;

        p.ls_type = (dist_int_2(rng) == 0) ? LineSearchType::MERIT : LineSearchType::FILTER;
        p.inertia = (dist_int_2(rng) == 0) ? InertiaStrategy::REGULARIZATION : InertiaStrategy::IGNORE_SINGULAR;
        
        p.enable_restoration = dist_bool(rng);
        p.enable_slack_reset = dist_bool(rng);

        // Continuous
        std::uniform_real_distribution<double> dist_01(0.0, 1.0);
        
        // Log uniform mu_init: 1e-2 to 10.0
        p.mu_init = std::pow(10.0, -2.0 + dist_01(rng) * 3.0); 

        // Log uniform reg_init: 1e-6 to 1e-2
        p.reg_init = std::pow(10.0, -6.0 + dist_01(rng) * 4.0);
        
        // Tol Con: 1e-3 to 1e-5 (Tighter is harder)
        p.tol_con = std::pow(10.0, -5.0 + dist_01(rng) * 2.0);

        p.slack_reset_trigger = 0.01 + dist_01(rng) * 0.1; // 0.01 to 0.11

        return p;
    }
};

// --- Evaluation ---

struct EvalResult {
    bool success;
    double avg_time;
    double score; // Lower is better
};

EvalResult evaluate_config(const TunableParams& params) {
    SolverConfig config = to_solver_config(params);
    
    // Define Scenarios
    struct ScenarioDef {
        std::string name;
        std::function<void(MiniSolver<CarModel, 100>&)> setup;
    };
    
    std::vector<ScenarioDef> scenarios;
    
    // 1. Standard Benchmark
    scenarios.push_back({"Standard", [&](MiniSolver<CarModel, 100>& s){
        for(int k=0; k<=s.N; ++k) {
             double t = k * 0.1; // Approx
             s.set_parameter(k, "v_ref", 5.0);
             s.set_parameter(k, "x_ref", t*5.0);
             s.set_parameter(k, "y_ref", 0.0);
             s.set_parameter(k, "obs_x", 12.0);
             s.set_parameter(k, "obs_y", 0.0);
             s.set_parameter(k, "obs_rad", 1.5);
             s.set_parameter(k, "L", 2.5);
             s.set_parameter(k, "car_rad", 1.0);
             s.set_parameter(k, "w_pos", 1.0);
             s.set_parameter(k, "w_vel", 1.0);
             s.set_parameter(k, "w_theta", 0.1);
             s.set_parameter(k, "w_acc", 0.1);
             s.set_parameter(k, "w_steer", 1.0);
        }
        s.set_initial_state("x", 0.0);
        s.set_initial_state("y", 0.0);
        s.set_initial_state("theta", 0.0);
        s.set_initial_state("v", 0.0);
    }});

    // 2. Hard Obstacle (Larger, closer)
    scenarios.push_back({"HardObs", [&](MiniSolver<CarModel, 100>& s){
        for(int k=0; k<=s.N; ++k) {
             double t = k * 0.1;
             s.set_parameter(k, "v_ref", 5.0);
             s.set_parameter(k, "x_ref", t*5.0);
             s.set_parameter(k, "y_ref", 0.0);
             s.set_parameter(k, "obs_x", 10.0); // Closer
             s.set_parameter(k, "obs_y", 0.1);  // Slightly offset
             s.set_parameter(k, "obs_rad", 1.8); // Slightly smaller than 2.0
             s.set_parameter(k, "L", 2.5);
             s.set_parameter(k, "car_rad", 1.0);
             // Higher penalty on constraints implicitly via barrier
             s.set_parameter(k, "w_pos", 1.0);
             s.set_parameter(k, "w_vel", 1.0);
             s.set_parameter(k, "w_theta", 0.1);
             s.set_parameter(k, "w_acc", 0.1);
             s.set_parameter(k, "w_steer", 1.0);
        }
        s.set_initial_state("x", 0.0);
        s.set_initial_state("y", 0.0);
        s.set_initial_state("theta", 0.0);
        s.set_initial_state("v", 0.0);
    }});
    
    // 3. Bad Initial Guess (Cold start with zero inputs is already "bad", but let's try non-zero bad inputs)
    // Actually, solver.rollout_dynamics() overwrites state guesses, so we only control control guesses.
    scenarios.push_back({"BadGuess", [&](MiniSolver<CarModel, 100>& s){
        for(int k=0; k<=s.N; ++k) {
             s.set_parameter(k, "v_ref", 5.0);
             s.set_parameter(k, "x_ref", k*0.1*5.0);
             s.set_parameter(k, "y_ref", 0.0);
             s.set_parameter(k, "obs_x", 12.0);
             s.set_parameter(k, "obs_y", 0.0);
             s.set_parameter(k, "obs_rad", 1.5);
             s.set_parameter(k, "L", 2.5);
             s.set_parameter(k, "car_rad", 1.0);
             s.set_parameter(k, "w_pos", 1.0);
             s.set_parameter(k, "w_vel", 1.0);
             s.set_parameter(k, "w_theta", 0.1);
             s.set_parameter(k, "w_acc", 0.1);
             s.set_parameter(k, "w_steer", 1.0);
             
             if(k < s.N) {
                 // Initialize with "Turn Left" guess which is wrong for obstacle at (12,0)
                 s.set_control_guess(k, "steer", 0.3); // Reduced from 0.5
                 s.set_control_guess(k, "acc", 1.0);
             }
        }
        s.set_initial_state("x", 0.0);
        s.set_initial_state("y", 0.0);
        s.set_initial_state("theta", 0.0);
        s.set_initial_state("v", 0.0);
    }});

    double total_time = 0.0;
    int fails = 0;
    
    for(auto& sc : scenarios) {
        MiniSolver<CarModel, 100> solver(60, Backend::CPU_SERIAL, config);
        // Setup
        sc.setup(solver);
        solver.rollout_dynamics(); // Initial rollout
        
        auto start = std::chrono::high_resolution_clock::now();
        SolverStatus status = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if(status != SolverStatus::OPTIMAL && status != SolverStatus::FEASIBLE) {
            fails++;
            total_time += 1000.0; // Penalty time
        } else {
            total_time += ms;
        }
    }
    
    EvalResult res;
    res.success = (fails == 0);
    res.avg_time = total_time / scenarios.size();
    
    // Score: Time + Penalty
    res.score = res.avg_time + (fails * 5000.0);
    
    return res;
}

int main(int argc, char** argv) {
    int samples = 50;
    if(argc > 1) samples = std::atoi(argv[1]);
    
    std::cout << ">> Auto-Tuning MiniSolver (" << samples << " samples)...\n";
    
    ConfigGenerator gen;
    TunableParams best_params;
    EvalResult best_res;
    best_res.score = 1e9;
    
    for(int i=0; i<samples; ++i) {
        TunableParams p = gen.generate(i);
        EvalResult res = evaluate_config(p);
        
        if (res.score < best_res.score) {
            best_res = res;
            best_params = p;
            std::cout << "[NEW BEST] ID: " << i << " Score: " << res.score 
                      << " AvgTime: " << res.avg_time << "ms Success: " << res.success << "\n";
            print_config(p);
            std::cout << "------------------------------------------------\n";
        }
        
        if(i % 10 == 0) std::cout << "." << std::flush;
    }
    
    std::cout << "\n\n>> TUNING COMPLETE.\n";
    std::cout << ">> BEST CONFIGURATION:\n";
    print_config(best_params);
    std::cout << ">> Final Score: " << best_res.score << "\n";
    
    return 0;
}

