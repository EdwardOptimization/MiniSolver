#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <fstream>

#include "generated/bicycleextmodel.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

// Debug configuration
struct ExtConfig {
    static const int N = 80;
    static constexpr double TARGET_V = 10.0;
    static constexpr double OBS_X = 15.0;
    static constexpr double OBS_Y = 0.5;
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

template<int MAX_N>
void print_traj_summary(const MiniSolver<BicycleExtModel, MAX_N>& solver) {
    std::cout << "\n--- Trajectory Summary ---\n";
    std::cout << std::left << std::setw(5) << "k"
              << std::setw(10) << "x"
              << std::setw(10) << "y"
              << std::setw(10) << "v"
              << std::setw(10) << "a"
              << std::setw(10) << "kappa"
              << std::setw(10) << "dk"
              << std::setw(10) << "jerk"
              << "\n";
              
    for(int k=0; k<=solver.N; k+=5) { // Print every 5th step
        auto x = solver.get_state(k);
        auto u = solver.get_control(k);
        
        std::cout << std::left << std::setw(5) << k
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << x[0]
                  << std::setw(10) << x[1]
                  << std::setw(10) << x[4]
                  << std::setw(10) << x[5]
                  << std::setw(10) << x[3]
                  << std::setw(10) << (k<solver.N ? u[0] : 0.0)
                  << std::setw(10) << (k<solver.N ? u[1] : 0.0)
                  << "\n";
    }
}

int main() {
    int N = ExtConfig::N;
    
    // --- Configuration ---
    SolverConfig config;
    config.integrator = IntegratorType::RK4_EXPLICIT; 
    config.default_dt = 0.05; 
    
    // Enable debug printing
    config.print_level = PrintLevel::DEBUG; 
    
    // Robustness strategy
    config.barrier_strategy = BarrierStrategy::ADAPTIVE; 
    config.line_search_type = LineSearchType::FILTER; 
    config.inertia_strategy = InertiaStrategy::REGULARIZATION;
    
    config.line_search_tau = 0.9; 
    
    config.reg_scale_down = 10.0; 
    
    config.max_iters = 150;
    config.tol_con = 1e-4; // Strict tolerance for simple problem
    config.mu_init = 0.1;  // Small mu for simple problem
    config.reg_init = 1e-6; 
    config.reg_min = 1e-9;
    
    config.enable_feasibility_restoration = true;
    config.enable_slack_reset = true;

    std::cout << ">> Initializing ExtBicycle MiniSolver (N=" << N << ")...\n";
    
    MiniSolver<BicycleExtModel, 120> solver(N, Backend::CPU_SERIAL, config);
    
    // Setup time steps
    std::vector<double> dts(N, 0.05);
    solver.set_dt(dts);

    // Setup Problem (Parameters & Initial Guess)
    double current_t = 0.0;
    for(int k=0; k<=N; ++k) {
        if(k > 0) current_t += dts[k-1];
        double x_ref = current_t * ExtConfig::TARGET_V; 
        
        // Smart reference for obstacle avoidance
        double y_ref_val = 0.0;
        if (x_ref > ExtConfig::OBS_X - 10.0 && x_ref < ExtConfig::OBS_X + 10.0) {
           y_ref_val = -2.5; 
        }

        solver.set_parameter(k, "v_ref", ExtConfig::TARGET_V);
        solver.set_parameter(k, "x_ref", x_ref);
        solver.set_parameter(k, "y_ref", y_ref_val);
        
        solver.set_parameter(k, "obs_x", ExtConfig::OBS_X);
        solver.set_parameter(k, "obs_y", ExtConfig::OBS_Y);
        solver.set_parameter(k, "obs_rad", ExtConfig::OBS_RAD);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", ExtConfig::CAR_RAD);
        
        // Ramp up weights to avoid initial shock
        double w_pos_val = (k < 10) ? 1.0 : 10.0;
        
        solver.set_parameter(k, "w_pos", w_pos_val);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_theta", 0.5); // Increase heading weight to keep direction
        solver.set_parameter(k, "w_kappa", 0.1);
        solver.set_parameter(k, "w_a", 0.1);
        solver.set_parameter(k, "w_dkappa", 1.0); // Stronger penalty on steering rate
        solver.set_parameter(k, "w_jerk", 1.0);   // Stronger penalty on jerk
        
        // Zero initialization for controls
        if(k < N) {
            solver.set_control_guess(k, "dkappa", 0.0);
            solver.set_control_guess(k, "jerk", 0.0);
        }
        
        // Simple linear guess
        double y_guess = y_ref_val;
        
        solver.set_state_guess(k, "x", x_ref);
        solver.set_state_guess(k, "y", y_guess);
        solver.set_state_guess(k, "v", ExtConfig::TARGET_V);
    }
    
    // Set Initial State
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("kappa", 0.0);
    solver.set_initial_state("v", 5.0); 
    solver.set_initial_state("a", 0.0);
    
    // Important: Don't just rollout dynamics from x0, 
    // because that would overwrite our smart state guess with a purely open-loop one 
    // which might be bad if controls are zero.
    // However, MiniSolver relies on "defect" (gap) minimization.
    // Let's trust the solver to close the gaps if we give it a good state guess.
    // BUT, MiniSolver implementation usually expects a consistent-ish trajectory or handles gaps.
    // The `rollout_dynamics` function FORCES the trajectory to be feasible w.r.t dynamics 
    // based on the current controls. 
    // If we want to keep the state guess, we should NOT call rollout_dynamics immediately, 
    // OR we should find controls that match the state guess (inverse dynamics).
    // For now, let's try WITHOUT rollout_dynamics to see if the warm-started states help.
    // Wait, if defects are large, first iteration might have huge steps.
    // Let's call rollout_dynamics() but with the non-zero control guess we set.
    // solver.rollout_dynamics();  


    std::cout << ">> Solving...\n";
    SolverStatus status = solver.solve(); 
    std::cout << ">> Final Status: " << status_to_string(status) << "\n";
    
    print_traj_summary(solver);

    std::cout << "\n>> Testing Warm Start (Shift & Solve again)...\n";
    // 1. Shift
    solver.shift_trajectory();
    
    solver.is_warm_started = true;
    
    // 2. Update Initial State (Simulate vehicle moving forward)
    // We take the state from k=1 of the previous solution as the new x0
    auto x1 = solver.get_state(0); // Since we shifted, index 0 now holds the old index 1
    solver.set_initial_state(x1);
    
    // 3. Update Reference (Shift time window)
    for(int k=0; k<=N; ++k) {
         // In a real MPC, we would sample the reference at t + k*dt
         // Here we just keep the reference consistent with the previous run's relative time?
         // No, reference should be updated.
         // Let's just update x_ref for t_new
         double t_new = current_t + 0.05 + (k * 0.05);
         solver.set_parameter(k, "x_ref", t_new * ExtConfig::TARGET_V);
    }
    
    // 4. Solve again
    // Reduce max iters because warm start should be fast
    solver.config.max_iters = 20; 
    
    SolverStatus status2 = solver.solve();
    std::cout << ">> Warm Start Status: " << status_to_string(status2) << "\n";
    print_traj_summary(solver);

    return 0;
}

