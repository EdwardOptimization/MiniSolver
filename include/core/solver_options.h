#pragma once

namespace minisolver {

enum class IntegratorType {
    EULER_EXPLICIT,
    EULER_IMPLICIT,
    RK2_EXPLICIT,
    RK2_IMPLICIT,
    RK4_EXPLICIT,
    RK4_IMPLICIT
};

enum class BarrierStrategy {
    MONOTONE, 
    ADAPTIVE, 
    MEHROTRA  
};

enum class InertiaStrategy {
    REGULARIZATION, 
    SATURATION,     
    IGNORE_SINGULAR 
};

enum class LineSearchType {
    MERIT,  
    FILTER  
};

// [NEW] Print Levels
enum class PrintLevel {
    NONE,   // Silent
    INFO,   // Start/End summary only
    ITER,   // One line per iteration
    DEBUG   // Detailed internal state
};

struct SolverConfig {
    // --- Integration ---
    IntegratorType integrator = IntegratorType::EULER_EXPLICIT;
    double default_dt = 0.1;

    // --- Barrier Strategy ---
    BarrierStrategy barrier_strategy = BarrierStrategy::MONOTONE; 
    
    double mu_init = 1e-1;      
    double mu_min = 1e-6;       
    double mu_linear_decrease_factor = 0.2; 
    double barrier_tolerance_factor = 10.0; 
    double mu_safety_margin = 0.1; 

    // --- Regularization ---
    InertiaStrategy inertia_strategy = InertiaStrategy::REGULARIZATION;
    double reg_init = 1e-6;     
    double reg_min = 1e-9;
    double reg_max = 1e9;
    double reg_scale_up = 10.0;
    double reg_scale_down = 2.0;

    // --- Convergence Tolerances ---
    double tol_grad = 1e-4;     
    double tol_con = 1e-4;      
    double tol_mu = 1e-5;       

    // --- Line Search & Robustness ---
    LineSearchType line_search_type = LineSearchType::MERIT;
    int line_search_max_iters = 10;
    double line_search_tau = 0.995; 
    
    bool enable_slack_reset = true; 
    double slack_reset_trigger = 0.05;

    bool enable_soc = true;             
    double merit_nu_init = 1000.0;      
    double eta_suff_descent = 1e-4;     
    
    bool enable_feasibility_restoration = true;

    // --- General ---
    int max_iters = 20;
    PrintLevel print_level = PrintLevel::ITER; // Replaces verbose/debug_mode
};

}
