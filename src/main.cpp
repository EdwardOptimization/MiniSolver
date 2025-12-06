#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <fstream>

#include "model/car_model.h"
#include "solver/solver.h"

using namespace roboopt;

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

struct IterationMetrics {
    double cost = 0.0;
    double prim_inf = 0.0; 
    double dual_inf = 0.0; 
    double compl_inf = 0.0; 
};

IterationMetrics compute_metrics(const PDIPMSolver<CarModel>& solver) {
    IterationMetrics m;
    
    // 1. Total Cost
    for(const auto& kp : solver.traj) {
        m.cost += kp.cost;
    }

    // 2. Primal Infeasibility
    for(const auto& kp : solver.traj) {
        for(int i=0; i<CarModel::NC; ++i) {
            double viol = std::abs(kp.g_val(i) + kp.s(i));
            if(viol > m.prim_inf) m.prim_inf = viol;
        }
    }

    // 3. Dual Infeasibility (Max Norm of Lagrangian Gradient)
    for(const auto& kp : solver.traj) {
        double g_norm = kp.q_bar.lpNorm<Eigen::Infinity>(); 
        double r_norm = kp.r_bar.lpNorm<Eigen::Infinity>(); 
        m.dual_inf = std::max(m.dual_inf, std::max(g_norm, r_norm));
    }

    // 4. Complementarity
    for(const auto& kp : solver.traj) {
        for(int i=0; i<CarModel::NC; ++i) {
            double comp = std::abs(kp.s(i) * kp.lam(i)); // deviation from 0
            if(comp > m.compl_inf) m.compl_inf = comp;
        }
    }

    return m;
}

int main(int argc, char** argv) {
    int N = 60;
    Backend mode = Backend::CPU_SERIAL; 

    // --- Configuration ---
    SolverConfig config;
    config.integrator = IntegratorType::RK4_EXPLICIT; 
    config.default_dt = 0.1; 

    config.barrier_strategy = BarrierStrategy::MONOTONE; 
    config.mu_init = 0.1;
    config.mu_min = 1e-6;   
    config.mu_linear_decrease_factor = 0.2; 
    config.reg_init = 1e-6; 
    config.reg_min = 1e-9;
    config.tol_con = 1e-4;
    config.max_iters = 60;  
    config.debug_mode = true; // Enable detailed diagnostics

    std::cout << ">> Initializing PDIPM Solver (N=" << N << ")...\n";
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
    std::cout << "Iter |   Cost   |  Log(Mu) |  Log(Reg)|  PrimInf |  DualInf | ComplInf \n";
    std::cout << "-----------------------------------------------------------------------\n";

    for(int iter=0; iter<config.max_iters; ++iter) {
        solver.step();
        IterationMetrics m = compute_metrics(solver);
        
        std::cout << std::setw(4) << iter << " | "
                  << std::scientific << std::setprecision(2) 
                  << m.cost << " | " 
                  << std::log10(solver.mu) << " | "
                  << std::log10(solver.reg) << " | "
                  << m.prim_inf << " | "
                  << m.dual_inf << " | "
                  << m.compl_inf << std::endl;

        if(solver.check_convergence(m.prim_inf)) {
            std::cout << ">> Converged.\n";
            break;
        }
    }

    // --- Warm Start Demo ---
    // Simulate a moving obstacle or slightly shifted target
    std::cout << "\n>> Testing Warm Start (Shifted Scenario)...\n";
    // Shift obstacle slightly
    double new_obs_x = 12.5; 
    
    // Save current solution
    auto warm_traj = solver.traj; 
    
    // Create new solver instance (clean slate)
    PDIPMSolver<CarModel> solver2(N, mode, config);
    solver2.set_dt(dts);
    
    // Update params
    for(int k=0; k<=N; ++k) {
        double params[] = { 5.0, 0.0, 0.0, new_obs_x, obs_y, obs_rad }; // x_ref needs recalculation but simplicity first
        // Fix params correctly
        double t_ref = (k<20)? k*0.05 : 1.0 + (k-20)*0.2;
        params[1] = t_ref * 5.0; 
        for(int i=0; i<6; ++i) solver2.traj[k].p(i) = params[i];
    }
    solver2.traj[0].x.setZero();

    // Apply Warm Start
    solver2.warm_start(warm_traj);
    
    std::cout << ">> Solving (Warm Start)...\n";
    std::cout << "Iter |   Cost   |  Log(Mu) |  Log(Reg)|  PrimInf |  DualInf | ComplInf \n";
    std::cout << "-----------------------------------------------------------------------\n";

    for(int iter=0; iter<config.max_iters; ++iter) {
        solver2.step();
        IterationMetrics m = compute_metrics(solver2);
        
        std::cout << std::setw(4) << iter << " | "
                  << std::scientific << std::setprecision(2) 
                  << m.cost << " | " 
                  << std::log10(solver2.mu) << " | "
                  << std::log10(solver2.reg) << " | "
                  << m.prim_inf << " | "
                  << m.dual_inf << " | "
                  << m.compl_inf << std::endl;

        if(solver2.check_convergence(m.prim_inf)) {
            std::cout << ">> Converged.\n";
            break;
        }
    }

    save_trajectory_csv("trajectory.csv", solver2, dts, new_obs_x, obs_y, obs_rad);
    return 0;
}
