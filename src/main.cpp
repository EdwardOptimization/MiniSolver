#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <fstream>

#include "model/car_model.h"
#include "solver/solver.h"

using namespace minisolver;

void save_trajectory_csv(const std::string& filename, 
                         const PDIPMSolver<CarModel>& solver, 
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
    for(size_t k=0; k < solver.traj.size(); ++k) {
        const auto& kp = solver.traj[k];
        if(k > 0 && k-1 < dts.size()) current_t += dts[k-1];

        file << current_t << ","
             << kp.x(0) << "," << kp.x(1) << "," << kp.x(2) << "," << kp.x(3) << ","
             << kp.u(0) << "," << kp.u(1) << ","
             << kp.g_val(4) << "," 
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

    config.barrier_strategy = BarrierStrategy::MONOTONE; 
    
    // [NEW] Advanced Features
    config.line_search_type = LineSearchType::FILTER;  
    config.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR; 
    config.enable_feasibility_restoration = true; 
    
    config.mu_init = 0.1;
    config.mu_min = 1e-6;   
    config.mu_linear_decrease_factor = 0.2; 
    config.reg_init = 1e-6; 
    config.reg_min = 1e-9;
    config.tol_con = 1e-4;
    config.max_iters = 60;  
    config.debug_mode = true; 
    config.verbose = true; // Use internal logging

    std::cout << ">> Initializing PDIPM Solver (N=" << N << ")...\n";
    std::cout << ">> Features: Filter LS, Inertia(Ignore), Feasibility Restoration\n";
    PDIPMSolver<CarModel> solver(N, mode, config);

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
        double params[] = { 5.0, x_ref, 0.0, obs_x, obs_y, obs_rad };
        if(k < N) solver.traj[k].u.setZero();
        for(int i=0; i<6; ++i) solver.traj[k].p(i) = params[i];
    }

    solver.traj[0].x.setZero(); 
    solver.rollout_dynamics();

    std::cout << ">> Solving (Cold Start)...\n";
    solver.solve(); // Use the simplified high-level interface

    save_trajectory_csv("trajectory.csv", solver, dts, obs_x, obs_y, obs_rad);
    return 0;
}
