#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

#include "model/car_model.h"
#include "solver/solver.h"

using namespace roboopt;

// --- Visualization Helper ---
void plot_trajectory(const std::vector<double>& X, const std::vector<double>& Y,
                     double obs_x, double obs_y)
{
    const int width = 60;
    const int height = 20;
    char grid[height][width];

    // 1. Clear Grid
    for(int i=0; i<height; ++i)
        for(int j=0; j<width; ++j)
            grid[i][j] = ' ';

    double min_x = 0, max_x = 25;
    double min_y = -4, max_y = 4;

    auto to_grid_x = [&](double x) { return (int)((x - min_x) / (max_x - min_x) * (width - 1)); };
    auto to_grid_y = [&](double y) { return (int)((1.0 - (y - min_y) / (max_y - min_y)) * (height - 1)); };

    // 2. Draw Reference Line (y=0)
    int y_zero = to_grid_y(0);
    if(y_zero >= 0 && y_zero < height) {
        for(int j=0; j<width; ++j) grid[y_zero][j] = '-';
    }

    // 3. Draw Obstacle [O]
    int ox = to_grid_x(obs_x);
    int oy = to_grid_y(obs_y);
    if(ox >= 0 && ox < width && oy >= 0 && oy < height) {
        grid[oy][ox] = 'O';
        if(ox+1 < width) grid[oy][ox+1] = 'O';
        if(ox-1 >= 0)    grid[oy][ox-1] = 'O';
    }

    // 4. Draw Car Path [*] (Overwrites Reference)
    for(size_t k=0; k<X.size(); ++k) {
        int gx = to_grid_x(X[k]);
        int gy = to_grid_y(Y[k]);
        if(gx >= 0 && gx < width && gy >= 0 && gy < height) {
            grid[gy][gx] = '*';
        }
    }

    std::cout << "\n--- Trajectory Plot ---\n";
    std::cout << "Legend: [O] Obstacle, [*] Car\n";
    std::cout << "+" << std::string(width, '-') << "+\n";
    for(int i=0; i<height; ++i) {
        std::cout << "|";
        for(int j=0; j<width; ++j) std::cout << grid[i][j];
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(width, '-') << "+\n";
}

int main(int argc, char** argv) {
    int N = 60;
    Backend mode = Backend::CPU_SERIAL; 

    std::cout << ">> Initializing Primal-Dual Interior Point Solver (N=" << N << ")...\n";
    PDIPMSolver<CarModel> solver(N, mode);

    // --- Scenario Definition ---
    double obs_x = 12.0;
    double obs_y = 0.0;
    double obs_weight = 200.0; // Lower weight than before, relying on dynamics and cost balance

    // Initialize Trajectory
    for(int k=0; k<=N; ++k) {
        double t = k * 0.1;
        double x_ref = t * 5.0; 
        double y_ref = 0.0;
        double v_target = 5.0;

        // p = [v_target, x_ref, y_ref, obs_x, obs_y, obs_weight]
        double params[] = { v_target, x_ref, y_ref, obs_x, obs_y, obs_weight };

        // Initial Control Guess: Drive straight
        if(k < N) {
            solver.traj[k].u(0) = 0.0;
            solver.traj[k].u(1) = 0.0;
        }

        for(int i=0; i<6; ++i) solver.traj[k].p(i) = params[i];
    }

    // Initial Rollout to get consistent x
    solver.traj[0].x.setZero(); // Start at 0,0
    solver.rollout_dynamics();

    std::cout << ">> Solving...\n";

    // Solver Loop
    for(int iter=0; iter<20; ++iter) {
        solver.step();

        // Optional: Print progress
        double max_constraint_viol = 0.0;
        for(const auto& kp : solver.traj) {
            // Check violation: g(x,u) <= 0
            // Since we use slack s, check primal residual: g + s = 0
            for(int i=0; i<CarModel::NC; ++i) {
                // If s > 0, then g = -s < 0 (Satisfied)
                // Real violation is if g > 0.
                if(kp.g_val(i) > max_constraint_viol) max_constraint_viol = kp.g_val(i);
            }
        }

        std::cout << "Iter " << std::setw(2) << iter 
                  << " | Mu: " << std::scientific << solver.mu 
                  << " | Max Con: " << max_constraint_viol << std::endl;
        
        if(solver.mu < 1e-5 && max_constraint_viol < 1e-3) {
            std::cout << ">> Converged.\n";
            break;
        }
    }

    // Extract Results
    std::vector<double> path_x, path_y;
    for(const auto& kp : solver.traj) {
        path_x.push_back(kp.x(0));
        path_y.push_back(kp.x(1));
    }

    // Plot
    plot_trajectory(path_x, path_y, obs_x, obs_y);
    std::cout << ">> Final X Position: " << path_x.back() << std::endl;

    return 0;
}
