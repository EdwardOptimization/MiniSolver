# MiniSolver: High-Performance Nonlinear MPC Solver

MiniSolver is a blazing fast, header-only C++17 library for solving Nonlinear Model Predictive Control (NMPC) problems. It is designed for embedded robotics applications where performance, robustness, and zero dynamic memory allocation are critical.

## 🚀 Key Features

*   **⚡ Blazing Fast**: Solves a 60-step nonlinear kinematic bicycle model with obstacle avoidance in **< 2ms** on a standard CPU.
*   **🧠 Auto-Differentiation**: Comes with a Python DSL (`MiniModel`) based on SymPy to automatically generate highly optimized, C++17 compatible model code with analytical derivatives (Jacobians/Hessians).
*   **🛡️ Robust Algorithms**: Implements state-of-the-art Interior Point Method (IPM) techniques:
    *   **Filter Line Search**: For global convergence without merit function tuning parameters.
    *   **Inertia Correction**: Handles non-convexity by dynamically regularizing the Hessian.
    *   **Feasibility Restoration**: Automatically recovers from infeasible warm starts or bad steps.
    *   **Watchdog / Slack Reset**: Heuristic strategies to escape local minima or stuck iterations.
*   **💾 Zero-Malloc**: Uses `std::array` and static templates (`MAX_N`) to ensure **zero dynamic memory allocation** during the solve phase, making it hard real-time safe.
*   **🔌 Matrix Abstraction**: Built on a flexible Matrix Abstraction Layer (MAL), allowing you to swap the backend (currently Eigen3) with custom embedded math libraries.

## 📊 Benchmark

On an Intel Core i7 (Single Thread):

| Metric | Time |
| :--- | :--- |
| **Derivatives** | ~0.2 ms |
| **Linear Solve (Riccati)** | ~0.8 ms |
| **Line Search** | ~0.5 ms |
| **Total Solve (Cold Start)** | **~1.7 ms** |

## 🛠️ Quick Start

### Prerequisites
*   CMake >= 3.10
*   Eigen3
*   Python 3 (for code generation) + SymPy (`pip install sympy`)
*   C++17 Compiler (GCC/Clang)

### Build & Run Demo

```bash
# 1. Generate C++ Model from Python DSL
python3 tools/car_model_gen.py

# 2. Build Project
mkdir build && cd build
cmake ..
make -j4

# 3. Run Demo
./MiniSolverApp
```

Or simply run the all-in-one script:
```bash
./run_demo.sh
```

This will solve a collision avoidance problem and generate a `trajectory_plot.png` visualization.

## 📝 Defining Your Own OCP

MiniSolver separates model definition from the solver core. You define your OCP in Python, and we generate the fast C++ code.

Create a file `my_model.py`:

```python
from MiniModel import OptimalControlModel
import sympy as sp

model = OptimalControlModel(name="DroneModel")

# 1. Define Variables
px = model.state("px")
py = model.state("py")
vx = model.state("vx")
vy = model.state("vy")
thrust = model.control("thrust")
theta = model.control("theta")

# 2. Define Parameters (References, Obstacles, Weights)
target_x = model.parameter("target_x")

# 3. Dynamics (x_dot = f(x,u))
model.set_dynamics(px, vx)
model.set_dynamics(py, vy)
model.set_dynamics(vx, thrust * sp.cos(theta))
model.set_dynamics(vy, thrust * sp.sin(theta))

# 4. Objective (Least Squares style)
model.minimize( 10.0 * (px - target_x)**2 )
model.minimize( 0.1 * thrust**2 )

# 5. Constraints (g(x,u) <= 0)
model.subject_to( thrust - 10.0 <= 0 ) # Max thrust
model.subject_to( 1.0 - (px**2 + py**2) <= 0 ) # Keep away from origin (r > 1)

# 6. Generate C++ Header
model.generate("include/model")
```

Then in your C++ code:

```cpp
#include "model/drone_model.h"
#include "solver/solver.h"

using namespace minisolver;

int main() {
    PDIPMSolver<DroneModel, 100> solver(50, Backend::CPU_SERIAL); // N=50
    
    // Set initial state
    solver.set_initial_state("px", 0.0);
    
    // Solve
    solver.solve();
}
```

## 📂 Project Structure

*   `include/core/`: Basic types (`KnotPoint`), configuration, and matrix abstraction.
*   `include/algorithms/`: Independent algorithm implementations (Riccati, Line Search).
*   `include/solver/`: The main `PDIPMSolver` orchestrator.
*   `include/model/`: Generated model headers.
*   `tools/`: Python DSL (`MiniModel.py`) and benchmark tools.

## 🤝 License
MIT
