#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <fstream>

#include "model/car_model.h"
#include "solver/solver.h"

using namespace minisolver;

// Template to accept any PDIPMSolver<Model, MAX_N>
template<typename SolverType>
void save_trajectory_csv(const std::string& filename, 
                         const SolverType& solver, 
                         const std::vector<double>& dts,
                         double obs_x, double obs_y, double obs_rad) 
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    file << "t,x,y,theta,v,acc,steer,g_obs,obs_x,obs_y,obs_rad\n";

    double current_t = 0.0;
    // Iterate only up to valid horizon N
    // Use get_state and get_constraint_val instead of get_traj
    
    for(int k=0; k <= solver.N; ++k) {
        std::vector<double> x = solver.get_state(k);
        std::vector<double> u = solver.get_control(k); // might be empty at N
        double g_obs = solver.get_constraint_val(k, 4); // Obstacle constraint
        
        if(k > 0 && k-1 < dts.size()) current_t += dts[k-1];

        // Safe access
        double u0 = (k < solver.N) ? u[0] : 0.0;
        double u1 = (k < solver.N) ? u[1] : 0.0;

        file << current_t << ","
             << x[0] << "," << x[1] << "," << x[2] << "," << x[3] << ","
             << u0 << "," << u1 << ","
             << g_obs << "," 
             << obs_x << "," << obs_y << "," << obs_rad << "\n";
    }
    file.close();
    std::cout << ">> Trajectory saved to " << filename << "\n";
}

int main(int argc, char** argv) {
    int N = 60;
    Backend mode = Backend::CPU_SERIAL; 

    // --- Configuration ---
    SolverConfig config;
    config.integrator = IntegratorType::RK4_EXPLICIT; 
    config.default_dt = 0.1; 

    config.barrier_strategy = BarrierStrategy::MONOTONE; // Revert to robust strategy
    
    // [NEW] Advanced Features
    config.line_search_type = LineSearchType::FILTER;  
    config.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR; 
    config.enable_feasibility_restoration = true; 
    
    // Trigger Slack Reset more easily to escape bad initial guess
    config.slack_reset_trigger = 0.1; 
    config.mu_init = 100.0;
    config.mu_min = 1e-6;   
    config.mu_linear_decrease_factor = 0.2; 
    config.reg_init = 1e-4; 
    config.reg_min = 1e-6; 
    config.tol_con = 1e-4;
    config.max_iters = 60;  
    config.print_level = PrintLevel::DEBUG; 

    std::cout << ">> Initializing PDIPM Solver (N=" << N << ")...\n";
    std::cout << ">> Features: Filter LS, Inertia(Ignore), Feasibility Restoration\n";
    
    // Instantiate with MAX_N = 100
    PDIPMSolver<CarModel, 100> solver(N, mode, config);

    std::vector<double> dts(N);
    for(int k=0; k<N; ++k) dts[k] = (k < 20) ? 0.05 : 0.2;
    solver.set_dt(dts);

    // Scenario
    double obs_x = 12.0; double obs_y = 0.0; double obs_rad = 1.5; 
    
    // Initialize Trajectory (Cold Start)
    double current_t = 0.0;
    for(int k=0; k<=N; ++k) {
        if(k > 0) current_t += dts[k-1];
        double x_ref = current_t * 5.0; 
        
        if(k < N) {
            solver.set_control_guess(k, "acc", 0.0);
            solver.set_control_guess(k, "steer", 0.0);
        }
        
        solver.set_parameter(k, "v_ref", 5.0);
        solver.set_parameter(k, "x_ref", x_ref);
        solver.set_parameter(k, "y_ref", 0.0);
        solver.set_parameter(k, "obs_x", obs_x);
        solver.set_parameter(k, "obs_y", obs_y);
        solver.set_parameter(k, "obs_rad", obs_rad);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);
        solver.set_parameter(k, "w_pos", 1.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "w_theta", 0.1);
        solver.set_parameter(k, "w_acc", 0.1);
        solver.set_parameter(k, "w_steer", 1.0);
    }
    
    // Set Initial State
    solver.set_initial_state("x", 0.0);
    solver.set_initial_state("y", 0.0);
    solver.set_initial_state("theta", 0.0);
    solver.set_initial_state("v", 0.0);
    solver.rollout_dynamics();

    std::cout << ">> Solving (Cold Start)...\n";
    solver.solve(); 

    save_trajectory_csv("trajectory.csv", solver, dts, obs_x, obs_y, obs_rad);
    return 0;
}
