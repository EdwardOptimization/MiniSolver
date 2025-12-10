# MiniSolver Benchmark & Configuration Guide

This document details the configuration archetypes used in the `benchmark_suite`. It serves as a guide for selecting the optimal solver settings for your specific application requirements.

## 1. Solver Archetypes

We have defined 8 distinct configuration sets ("Archetypes") representing common use cases in robotics and optimal control.

### 🏎️ 1. TURBO_MPC
* **Target Application**: Embedded Microcontrollers (MCU), high-frequency loops (>100Hz), resource-constrained drones/robots.
* **Key Settings**:
    * `Integrator = EULER_EXPLICIT`: Lowest computational cost per step.
    * `BarrierStrategy = ADAPTIVE`: Rapidly decreases barrier parameter $\mu$ when far from constraints.
    * `Tol = 1e-2`: Accepts "good enough" solutions to save iterations.
* **Pros**: Extremely fast (~1ms).
* **Cons**: Lower physical fidelity; potential integration drift over long horizons.

### ⚖️ 2. BALANCED_RT
* **Target Application**: General-purpose UGV/UAV navigation, Real-Time planning (50Hz).
* **Key Settings**:
    * `Integrator = RK2_EXPLICIT` (Midpoint): Good balance of accuracy vs. speed.
    * `BarrierStrategy = MEHROTRA`: Predictor-Corrector logic reduces total iteration count significantly.
* **Pros**: Stable convergence with moderate CPU usage. Recommended starting point.

### 🎯 3. QUALITY_PLANNER (Default)
* **Target Application**: Autonomous Driving trajectory planners, dynamic maneuvers.
* **Key Settings**:
    * `Integrator = RK4_EXPLICIT`: High physical fidelity.
    * `BarrierStrategy = MEHROTRA`: State-of-the-art convergence rate.
    * `Tol = 1e-4`: Standard engineering tolerance.
* **Pros**: High-quality smooth trajectories; reliable constraint handling.

### 🏛️ 4. CLASSIC_STABLE
* **Target Application**: Academic research baselines; scenarios where modern heuristics might oscillate.
* **Key Settings**:
    * `BarrierStrategy = MONOTONE`: Linearly decreases $\mu$ (Fiacco-McCormick).
    * `LineSearch = MERIT`: Uses L1-Penalty Merit function instead of Filter.
* **Pros**: Theoretically well-understood; very predictable behavior.
* **Cons**: Generally slower; Merit parameters may need manual tuning.

### 🛡️ 5. ROBUST_OBSTACLE
* **Target Application**: Cluttered environments, narrow passages, bad initial guesses.
* **Key Settings**:
    * `Enable SOC` (Second Order Correction): Curves the search step to fit nonlinear constraint boundaries (e.g., circular obstacles).
    * `Feasibility Restoration`: Automatically triggers a minimum-norm recovery phase if the solver gets stuck.
* **Pros**: Extremely robust against "getting stuck" on boundaries.
* **Cons**: SOC steps add computational overhead per iteration.

### 🔬 6. HIGH_PRECISION
* **Target Application**: Offline trajectory generation, Ground Truth computation, space applications.
* **Key Settings**:
    * `Hessian = EXACT`: Computes full Lagrange Hessian (including constraint curvature).
    * `Tol = 1e-6`: Tight tolerance for primal and dual variables.
* **Pros**: Quadratic convergence near solution; mathematically exact.
* **Cons**: Expensive derivatives; Hessian may become indefinite (requires Inertia Correction).

### ⚡ 7. AGGRESSIVE_RACING
* **Target Application**: Racing/Agile maneuvers where speed is prioritized over strict optimality.
* **Key Settings**:
    * `Aggressive Barrier`: Forces $\mu$ to drop faster if the step size is large.
    * `Inertia = IGNORE_SINGULAR`: Skips expensive regularization in some cases.
* **Pros**: Can reduce iteration count by 20-30%.
* **Cons**: Higher risk of numerical instability if the problem is ill-conditioned.

### ⏱️ 8. SQP-RTI (Real-Time Iteration)
* **Target Application**: Ultra-high frequency control (>1000Hz).
* **Key Settings**:
    * `MaxIters = 1`: Performs exactly one Quadratic Programming (QP) sub-problem per control tick.
    * `WarmStart`: Relies heavily on the solution from the previous timestep.
* **Pros**: Deterministic execution time (microsecond scale).
* **Cons**: Output is suboptimal; accuracy depends entirely on the feedback rate.

---

## 2. Technical Component Guide

### Integrators
| Type | Order | Cost | Use Case |
| :--- | :---: | :--- | :--- |
| **Euler** | 1 | 1x | Simple dynamics, short horizons, MCUs. |
| **RK2** | 2 | 2x | General robotics, good trade-off. |
| **RK4** | 4 | 4x | Complex dynamics (vehicles, drones), long horizons. |

### Barrier Strategies
* **Monotone**: Conservative. Good for debugging.
* **Adaptive**: Heuristic-based. Fast for loose tolerances.
* **Mehrotra**: Advanced Predictor-Corrector. Solves 2 linear systems per step but drastically reduces iteration count. **Recommended for most cases.**

### Hessian Approximation
* **Gauss-Newton (GN)**: $H \approx J^T J$. Convex by definition. Fast and stable. Ignores constraint curvature.
* **Exact**: $H = \nabla^2 \mathcal{L}$. Captures all curvature. Necessary for high precision but requires regularization if non-convex.

### Globalization (Line Search)
* **Merit**: Requires descent on $\phi(\alpha) = Cost + \nu \cdot Violation$. Sensitive to $\nu$.
* **Filter**: Accepts steps that improve *either* Cost *or* Violation. Parameter-free and generally more robust for nonlinear constraints.

---

## 3. Quick Selection Flowchart

1.  **Is hardware extremely limited (e.g., Arduino/STM32)?**
    * Yes $\rightarrow$ **TURBO_MPC**
2.  **Is the environment highly cluttered / nonlinear?**
    * Yes $\rightarrow$ **ROBUST_OBSTACLE**
3.  **Do you need >1kHz feedback?**
    * Yes $\rightarrow$ **SQP_RTI**
4.  **Are you generating ground truth data?**
    * Yes $\rightarrow$ **HIGH_PRECISION**
5.  **Default / Unsure?**
    * $\rightarrow$ **QUALITY_PLANNER** or **BALANCED_RT**