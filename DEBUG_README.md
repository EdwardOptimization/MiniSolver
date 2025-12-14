# Developer Guide & Internals

This guide documents the architectural decisions (ADRs), internal mechanisms, and debugging strategies for **MiniSolver**. It is intended for those modifying the core solver or diagnosing complex convergence issues.

---

## 🏗️ Architecture Decisions (ADR)

### 1. Python DSL vs. C++ Templates
**Decision**: Use Python (`MiniModel.py`) to generate C++ code instead of pure C++ templates or operator overloading (e.g., CppAD).
**Reasoning**:
* **Compile-Time Derivatives**: SymPy calculates analytical Jacobians and Hessians offline. The result is flat, scalar C++ code that compilers (GCC/Clang) can optimize heavily via Common Subexpression Elimination (CSE).
* **Readability**: Generated code explicitly shows the math, making it easier to audit than deeply nested template expression trees.

### 2. Memory Model: Zero-Malloc & Double Buffering
**Decision**: The solver must never allocate dynamic memory after initialization.
**Implementation**:
* **Static Arrays**: All data is stored in `std::array<KnotPoint, MAX_N>` structures sized at compile time.
* **Double Buffering**: The `Trajectory` class maintains two buffers: `memory_A` (active) and `memory_B` (candidate).
* **Pointer Swapping**: During Line Search, we generate trial steps in the `candidate` buffer. If accepted, we simply swap the `active_ptr` and `candidate_ptr` pointers. This avoids expensive `memcpy` operations (~200KB per step).

### 3. Zero-Copy Iterative Refinement
**Challenge**: Iterative Refinement (IR) requires preserving the original linear system $(A, b)$ to compute residuals $r = b - Ax$, but the Riccati solver factorizes matrices in-place (destroying $A$).
**Solution**: We repurpose the **Candidate Buffer** (normally used for Line Search trials) as a "Backup Buffer" during the linear solve phase.
1.  Copy `active` trajectory (linearization point) to `candidate`.
2.  Run Riccati factorization on `active`.
3.  Compute residuals using the preserved matrices in `candidate`.
4.  Apply correction $\delta x$ to `active`.

### 4. Matrix Abstraction Layer (MAL)
**Decision**: Decouple the solver logic from the linear algebra library.
**Implementation**: `include/minisolver/core/matrix_defs.h` defines a unified API.
* **`USE_EIGEN`**: Links against Eigen3 for SIMD optimizations (Desktop/Linux).
* **`USE_CUSTOM_MATRIX`**: Uses `MiniMatrix.h`, a self-contained, template-based linear algebra class. This enables compilation on bare-metal systems (e.g., STM32) with **zero external dependencies**.

### 5. Sparsity Handling: Fused Kernels
**Approach**: **Symbolic Code Generation**. Instead of a generic sparse matrix format (CSR/CSC), we generate specific C++ code for every matrix operation in the Riccati recursion.
* **Result**: Zero storage overhead for sparsity patterns. The "sparsity" is baked into the instruction stream.

---

## 🪤 Known "Gotchas" (Post-Mortem Analysis)

### 1. The "Ghost Cost" Bug (Parameter Synchronization)
* **Symptom**: Solver converges for one step, then the cost becomes zero or garbage in subsequent steps or iterations.
* **Cause**: When swapping buffers (`active` <-> `candidate`), we initially forgot to copy the user-set **Parameters** (`p`, e.g., reference trajectories) to the candidate buffer. The candidate buffer contained default-initialized parameters (zeros).
* **Fix**: The `Trajectory::prepare_candidate()` method must explicitly sync parameters or ensure parameters are global.

### 2. Vanishing Gradients inside Obstacles
* **Symptom**: Solver gets stuck when the initial guess is exactly inside a circular obstacle.
* **Cause**: Constraint form $R^2 - (x^2 + y^2) \le 0$. At $(0,0)$, the gradient $\nabla g = [-2x, -2y]$ is zero.
* **Fix**: We reformulated the constraint as a distance field: $R - \sqrt{x^2 + y^2 + \epsilon} \le 0$. This ensures the gradient is a unit vector pointing outward, providing valid search directions even at the center.

### 3. Dual Variable Inconsistency in Restoration
* **Symptom**: After `Feasibility Restoration` succeeds, the main IPM loop immediately diverges.
* **Cause**: The restoration phase solves a *different* optimization problem (min-norm correction). The Lagrange multipliers (Duals) from this phase are not valid for the original OCP constraints.
* **Fix**: After restoration, we perform a "Dual Reset" using the complementarity condition $\lambda_i = \mu / s_i$ to re-initialize valid duals for the original problem.

### 4. Feasible Stagnation (The "Rollout Trap")
* **Symptom**: Solver performs `rollout_dynamics()` (making primal constraint violation $0$) but then fails to reduce Cost, taking tiny steps (`Alpha ~ 0`) and hitting max iterations.
* **Cause**: When the initial guess is feasible but far from optimal (e.g., a straight line vs. a curve), **Adaptive Barrier** strategies may reduce $\mu$ too aggressively or misjudge the search direction. The **Filter** mechanism, having no "infeasibility" to trade off, degrades into a strict descent method, rejecting steps that would slightly violate dynamics to improve cost.
* **Fix**: For highly nonlinear warm-starts:
    1.  Use **Mehrotra Predictor-Corrector** (it handles nonlinearity better via higher-order corrections).
    2.  Use **Monotone Barrier** (more conservative/stable).
    3.  Increase `mu_init` (e.g., `0.1`) to keep the barrier "soft" initially.

---

## 🛠️ Debugging Strategies

### 1. High-Verbosity Logging
Use the `MINISOLVER_LOG_LEVEL` macro to inspect internal KKT residuals without overhead in production.

```cpp
// In CMake or Compilation flags
add_definitions(-DMINISOLVER_LOG_LEVEL=4)
````

  * **Log Level 4 (DEBUG)**: Prints `MinSlack`, `Alpha` (step size), and specific constraint violation indices.
  * **Interpretation**:
      * High `Reg` (e.g., $10^9$): Hessian is singular. Check if your cost function is convex or if the system is uncontrollable.
      * Tiny `Alpha` ($< 10^{-4}$): The line search is blocked by a constraint boundary (Step size too small).

### 2\. The Replay Tool (Serializer)

If the solver crashes in production, capture the state using the Serializer.

```cpp
// In your application code, before solve():
SolverSerializer<CarModel, 100>::save_case("crash_dump.dat", solver);
```

Then, debug locally using the replay tool:

```bash
./build/replay_solver crash_dump.dat
```

This loads the exact configuration, trajectory guess, and parameters from the crash site.

### 3\. Auto-Tuner

If convergence is unstable, use the `auto_tuner` tool to perform a Monte-Carlo search over the configuration space (Integrators, Barrier Strategies, Line Search types).

```bash
./build/auto_tuner 100  # Run 100 trials
```

It outputs the `SolverConfig` that achieved the highest success rate and lowest runtime.

-----

## 🔮 Future Roadmap

* [x] **Iterative Refinement**: Implemented to recover precision from regularization errors.
* [x] **Slack Reset & Restoration**: Implemented for robust feasibility handling.
* [x] **Fused Riccati**: Implemented via SymPy codegen.
* [ ] **GPU Parallelization**: `gpu_ops.cu` exists as a placeholder. Parallel Scan (MPX/PCR) logic needs to be fully implemented for $N > 100$.
* [ ] **Implicit Integrators**: Better Newton-based implicit solvers for stiff systems.

-----

*MiniSolver Team - Internal Use Only*
