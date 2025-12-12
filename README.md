

# MiniSolver: High-Performance Embedded NMPC Library

![C++17](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Embedded-lightgrey)
![Zero-Malloc](https://img.shields.io/badge/Memory-Zero--Malloc-brightgreen)

**MiniSolver** is a professional-grade, header-only C++17 library for solving Nonlinear Model Predictive Control (NMPC) problems.

Engineered specifically for **embedded robotics** and **autonomous driving**, it combines the flexibility of Python-based symbolic modeling with the raw performance of highly optimized, template-generated C++ code. It explicitly guarantees **Zero Dynamic Memory Allocation (Zero-Malloc)** during the solve phase, making it deterministic and safe for hard real-time systems.

---

## 🚀 Key Features

### ⚡ Blazing Fast Performance
* **Riccati Recursion**: Utilizes a specialized block-tridiagonal linear solver ($O(N)$ complexity) tailored for optimal control structures.
* **SQP-RTI Support**: Real-Time Iteration (SQP-RTI) mode allows for **>1 kHz** control loops by performing a single quadratic programming sub-step per control tick.
* **Analytical Derivatives**: Uses SymPy to generate flattened, algebraically simplified C++ code for Jacobians and Hessians at compile-time, eliminating runtime overhead.

### 🛡️ Embedded Safety & Robustness
* **Zero-Malloc Guarantee**: All memory is allocated on the stack (or `.bss`) via `std::array` and C++ templates (`MAX_N`). No `new`/`malloc` calls occur during the `solve()` loop.
* **Robust Interior Point Method (IPM)**:
    * **Mehrotra Predictor-Corrector**: Drastically reduces iteration counts by utilizing higher-order corrections.
    * **Filter Line Search**: Ensures global convergence without the tedious tuning of merit function parameters.
    * **Feasibility Restoration**: Automatically triggers a minimum-norm recovery phase if the solver encounters infeasible warm starts.

### 🔧 Advanced Solver Capabilities
* **Second Order Correction (SOC)**: Handles highly nonlinear constraints (e.g., tight obstacle avoidance) by curving the search step.
* **Native Soft Constraints**: Supports L1 (Exact) and L2 (Quadratic) soft constraints via **Dual Regularization**, allowing for relaxation without increasing the problem dimensions (slack variables are handled implicitly).
* **Iterative Refinement**: High-precision mode that uses full-precision residuals to correct regularization errors in the linear solver.

---

## 📊 Performance Benchmarks

Benchmarks performed on an Intel Core i7 (Single Thread) for a **60-step Kinematic Bicycle Model** with obstacle avoidance.

**Note:** When using Fused Riccati (default), ensure the integrator type used in C++ matches the one used during Python generation. The fused kernel is specialized for a specific integrator's Jacobian structure.

| Archetype | Configuration | Avg Time | Use Case |
| :--- | :--- | :--- | :--- |
| **TURBO_MPC** | Euler + Adaptive Barrier | **~0.8 ms** | Microcontrollers (MCU), Racing Drones |
| **BALANCED_RT** | RK2 + Mehrotra | **~1.2 ms** | General UGV Navigation |
| **QUALITY_PLANNER** | RK4 + Mehrotra + Filter | **~1.8 ms** | Autonomous Driving Trajectory Planner |
| **SQP-RTI** | Euler + Single Iteration | **~0.2 ms** | High-Frequency Control (>1kHz) |

---

## 🛠️ Quick Start

### Prerequisites
* **CMake** >= 3.10
* **C++17 Compiler** (GCC/Clang)
* **Python 3** + **SymPy** (`pip install sympy`)
* *(Optional)* **Eigen3** (Default backend, can be swapped for built-in `MiniMatrix`)

### 1. Define Your Model (Python)
Define your Optimal Control Problem (OCP) using the Python DSL. This generates the optimized C++ headers.

```python
# my_model.py
from minisolver.MiniModel import OptimalControlModel
import sympy as sp

model = OptimalControlModel(name="DroneModel")

# 1. Define Variables
px, py, vz = model.state("px", "py", "vz")
thrust = model.control("thrust")

# 2. Dynamics (f(x,u))
model.set_dynamics(px, vx) # ... assume vx defined
model.set_dynamics(py, vy)
model.set_dynamics(vz, thrust - 9.81)

# 3. Objective (Least Squares)
model.minimize( 10.0 * (px - 0.0)**2 ) # Target x=0
model.minimize( 0.1 * thrust**2 )      # Regularization

# 4. Constraints
model.subject_to( thrust <= 20.0 )     # Hard Constraint
# Soft Constraint (L1 Penalty via Dual Regularization)
model.subject_to( vz <= 5.0, weight=100.0, loss='L1' )

# 5. Generate C++ Code
model.generate("include/model")
````

### 2\. Solve in C++

Include the generated header and the solver.

```cpp
#include "model/drone_model.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

int main() {
    // Instantiate solver with Max Horizon N=100
    MiniSolver<DroneModel, 100> solver(50, Backend::CPU_SERIAL);
    
    // Set Initial Condition
    solver.set_initial_state("px", -10.0);
    
    // Configure for Robustness
    solver.config.barrier_strategy = BarrierStrategy::MEHROTRA;
    solver.config.enable_soc = true;
    
    // Solve
    SolverStatus status = solver.solve();
    
    // Retrieve Optimal Control
    std::vector<double> u_opt = solver.get_control(0);
}
```

### 3\. Build & Run

MiniSolver includes a one-click build script that handles dependency checking, code generation, and compilation.

```bash
./build.sh
```

-----

## 📂 Project Structure

  * **`include/core/`**: Core types (`KnotPoint`), memory-safe containers (`Trajectory`), and Matrix Abstraction Layer (`MiniMatrix`/`Eigen`).
  * **`include/solver/`**: The main `MiniSolver` class orchestrating the IPM loop.
  * **`include/algorithms/`**: Numerical engines:
      * `RiccatiSolver`: Block-tridiagonal linear system solver.
      * `LineSearch`: Filter and Merit function strategies.
  * **`python/minisolver/`**: The `MiniModel` DSL and C++ code generator.
  * **`tools/`**:
      * `auto_tuner.cpp`: Monte-Carlo search for optimal solver configurations.
      * `replay_solver.cpp`: Debugging tool to replay serialized solver states.
      * `benchmark_suite/`: comprehensive performance testing.

## 🤝 License & Citation

**MiniSolver** is licensed under the **Apache 2.0 License**.

If you use this software in academic work, please refer to `CITATION.cff`.

*Maintained by Edward Qu.*
