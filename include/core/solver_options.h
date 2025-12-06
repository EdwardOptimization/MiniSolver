#pragma once

namespace roboopt {

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

struct SolverConfig {
    // --- Integration ---
    IntegratorType integrator = IntegratorType::EULER_EXPLICIT;
    double default_dt = 0.1;

    // --- Barrier Strategy ---
    BarrierStrategy barrier_strategy = BarrierStrategy::MONOTONE; 
    
    double mu_init = 1e-1;      
    double mu_min = 1e-6;       
    
    // For MONOTONE:
    double mu_linear_decrease_factor = 0.2; 
    double barrier_tolerance_factor = 10.0; 
    
    // For ADAPTIVE/MEHROTRA:
    double mu_safety_margin = 0.1; 

    // --- Regularization ---
    double reg_init = 1e-6;     
    double reg_min = 1e-6;
    double reg_max = 1e9;
    double reg_scale_up = 10.0;
    double reg_scale_down = 2.0;

    // --- Convergence Tolerances ---
    double tol_grad = 1e-4;     
    double tol_con = 1e-4;      
    double tol_mu = 1e-5;       

    // --- Line Search ---
    int line_search_max_iters = 10;
    double line_search_tau = 0.995; 

    // --- General ---
    int max_iters = 20;
    bool verbose = true;
    bool debug_mode = false; // <--- NEW: Enable detailed per-iteration debug prints
};

}
