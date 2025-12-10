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
    // In the future: FACTORIZATION_MODIFY
};

enum class LineSearchType {
    MERIT,  
    FILTER  
};

enum class HessianApproximation {
    EXACT,          // Full Hessian (Objective + Constraints)
    GAUSS_NEWTON    // Gauss-Newton (Objective J^T J only), ignore constraint curvature
};

// Print Levels
enum class PrintLevel {
    NONE,   // Silent
    WARN,   // Warnings and Errors only
    INFO,   // Start/End summary only
    ITER,   // One line per iteration
    DEBUG   // Detailed internal state
};

enum class Backend {
    CPU_SERIAL,
    GPU_MPX,
    GPU_PCR
};

struct SolverConfig {
    Backend backend = Backend::CPU_SERIAL;

    // --- Integration ---
    // RK4 is a good balance for general nonlinear problems
    IntegratorType integrator = IntegratorType::RK4_EXPLICIT;
    double default_dt = 0.1;

    // --- Barrier Strategy ---
    // ADAPTIVE is generally the most robust and fast for general nonlinear problems
    BarrierStrategy barrier_strategy = BarrierStrategy::ADAPTIVE; 
    
    double mu_init = 1e-1;      
    double mu_min = 1e-6;       // Tighter tolerance for high precision
    double mu_linear_decrease_factor = 0.2; 
    double barrier_tolerance_factor = 10.0; 
    double mu_safety_margin = 0.1; 

    // --- Regularization ---
    InertiaStrategy inertia_strategy = InertiaStrategy::REGULARIZATION;
    double reg_init = 1e-4;     // Slightly higher init to be safe
    double reg_min = 1e-8;
    double reg_max = 1e9;
    double reg_scale_up = 100.0; // Aggressive scaling to recover quickly
    double reg_scale_down = 2.0;
    
    // Inertia Tuning
    double singular_threshold = 1e-4; // For IGNORE_SINGULAR
    double huge_penalty = 1e9;        // Penalty for frozen directions
    int inertia_max_retries = 5;

    // --- Convergence Tolerances ---
    double tol_grad = 1e-4;     
    double tol_con = 1e-4;      
    double tol_dual = 1e-4; // [NEW] Dual Infeasibility Tolerance
    double tol_mu = 1e-5;       
    // [NEW] Objective Stagnation Tolerance
    // Stops the solver if the cost improvement between iterations is smaller than this value,
    // provided the solution is feasible.
    double tol_cost = 1e-6;

    // --- Line Search & Robustness ---
    // Filter is generally more robust than Merit without parameter tuning
    LineSearchType line_search_type = LineSearchType::FILTER;
    int line_search_max_iters = 10;
    double line_search_tau = 0.995; // Fraction to boundary
    double line_search_backtrack_factor = 0.5; 
    
    // Filter Method Parameters
    double filter_gamma_theta = 1e-5; 
    double filter_gamma_phi = 1e-5;   
    
    // Barrier Numerical Safety
    double min_barrier_slack = 1e-12; 
    double barrier_inf_cost = 1e9;    
    
    // Watchdog / Heuristics
    double slack_reset_trigger = 1e-3; // Only reset if step is VERY small
    double warm_start_slack_init = 1e-2; 

    // Globalization
    double soc_trigger_alpha = 0.5; 
    double merit_nu_init = 1000.0;      
    double eta_suff_descent = 1e-4;     
    
    // Restoration
    int max_restoration_iters = 5; 
    double restoration_mu = 1e-1;  
    double restoration_reg = 1e-2; 
    double restoration_alpha = 0.8; 

    // --- General ---
    int max_iters = 30; // Give it enough room
    PrintLevel print_level = PrintLevel::ITER; 

    // --- Advanced Features ---
    HessianApproximation hessian_approximation = HessianApproximation::GAUSS_NEWTON; // Default to GN for speed
    
    bool enable_iterative_refinement = false;
    int max_refinement_steps = 1;
    
    // SQP-RTI Mode
    bool enable_rti = false; // [NEW] If true, solve() performs only 1 SQP iteration (or config.max_iters)
    // and reuses linearization if possible (requires more state storage, currently partial support via warm_start) 
    
    // Line Search Logic
    // PURE IPM: Disable rollout by default. Trust the linearization.
    bool enable_line_search_rollout = false; 
    
    // Riccati Logic
    bool enable_defect_correction = true; 
    
    // Mehrotra Logic
    bool enable_corrector = true; 
    bool enable_aggressive_barrier = true; // [NEW] Allow aggressive mu reduction based on step size
    
    // Feasibility Logic (Heuristics)
    // Disabled by default for PURE IPM behavior. Enable only if needed.
    bool enable_slack_reset = true; // Enable by default for robustness
    bool enable_feasibility_restoration = true; // Enable by default
    bool enable_soc = true; // Enable SOC by default for robust handling of nonlinearities 
};

}
