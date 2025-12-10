#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

// Include the Model you want to replay (e.g. CarModel)
#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/solver/solver.h"
#include "minisolver/core/serializer.h"

using namespace minisolver;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./replay_solver <case_file.dat>\n";
        return 1;
    }
    
    std::string filename = argv[1];
    
    // We assume MAX_N = 100 is enough for logged cases
    // If your production uses larger N, increase this.
    constexpr int MAX_N = 100;
    
    // We assume the model is CarModel. 
    // If you log different models, you need a different replay tool or compile switch.
    using Model = CarModel;
    
    std::cout << ">> Replaying Case: " << filename << "\n";
    
    // 1. Create Solver (Config will be overwritten by load)
    MiniSolver<Model, MAX_N> solver(10, Backend::CPU_SERIAL); // Initial N doesn't matter
    
    // 2. Load Case
    if (!SolverSerializer<Model, MAX_N>::load_case(filename, solver)) {
        std::cerr << ">> Failed to load case.\n";
        return 1;
    }
    
    // 3. Solve
    std::cout << ">> Config Loaded:\n";
    std::cout << "   N: " << solver.N << "\n";
    std::cout << "   Integrator: " << (int)solver.config.integrator << "\n";
    std::cout << "   Barrier: " << (int)solver.config.barrier_strategy << "\n";
    
    solver.config.print_level = PrintLevel::ITER; // Force verbose for replay
    
    std::cout << ">> Starting Solve...\n";
    SolverStatus status = solver.solve();
    
    std::cout << ">> Replay Finished. Status: " << status_to_string(status) << "\n";
    
    return 0;
}

