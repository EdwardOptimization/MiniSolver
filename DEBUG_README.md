# Developer Guide & Debugging Notes

This document is for contributors and maintainers of **MiniSolver**. It documents the architectural decisions, "gotchas", and lessons learned during development.

---

## 🏗️ Architecture Decisions (ADR)

### 1. Python DSL + C++ CodeGen
**Why?**
*   **Auto-Differentiation**: Writing analytical derivatives for RK4 expansion manually is error-prone. Finite Differences are slow and inaccurate. SymPy handles this perfectly at compile-time.
*   **Performance**: The generated code is flat C++, often using CSE (Common Subexpression Elimination), which compilers optimize heavily (`-O3`).
*   **Flexibility**: Users can define dynamics in Python without touching C++ templates.

### 2. Zero-Malloc (Static Allocation)
**Why?**
*   **Embedded Safety**: `malloc` is non-deterministic. We use `std::array<Knot, MAX_N>` to allocate everything on the stack (or BSS).
*   **Cache Locality**: Contiguous memory layout improves prefetching.

### 3. Double Buffering (Pointer Swapping)
**Why?**
*   In Line Search, we generate trial trajectories. Copying the entire `std::vector` or `std::array` (60 knots * 2KB) is expensive (~120KB memcpy).
*   We use two buffers (`traj_memory_A`, `traj_memory_B`) and swap `traj_ptr` and `candidate_ptr`.

### 4. Zero-Malloc Iterative Refinement
**Problem**: Iterative Refinement (IR) requires saving a copy of the original linear system (A, b) to compute residuals $r = b - Ax$. Riccati solves in-place, destroying A and b. Allocating a copy violates "Zero-Malloc".
**Solution**: We utilize the **Candidate Buffer** (normally used for Line Search trial steps) as temporary storage for the backup.
1.  **Backup**: Before `solve()`, copy `active` trajectory (containing A, b) to `candidate`.
2.  **Solve**: Run Riccati on `active` (A, b -> L, D, x).
3.  **Refine**: 
    *   Compute residual using `candidate` (A, b) and `active` (x).
    *   Store residual in `candidate` (overwrite b).
    *   Run Riccati on `candidate` to find correction $\delta x$.
    *   Update `active`: $x \leftarrow x + \delta x$.
**Benefit**: High precision and robustness against regularization errors without extra memory.

---

## 🪤 The "Gotchas" (Post-Mortem Analysis)

### 1. The "Ghost Cost" Bug (Parameter Sync)
*   **Symptom**: Solver runs fine for 1 iteration, then Cost becomes meaningless or zero, convergence fails.
*   **Cause**: We implemented Double Buffering for Line Search (`swap(active, candidate)`). We copied `x, u, s, lam` to the candidate, but **forgot to copy Parameters (`p`)**.
*   **Result**: When the pointer swapped, the new active trajectory had zeroed parameters (default initialized).
*   **Fix**: Explicitly copy `p` in the Line Search loop, or ensure static parameters are synced across all buffers.

### 2. Variable Shadowing in CodeGen
*   **Symptom**: Compilation error `declaration of 'T x' shadows a parameter` or silent logic errors.
*   **Cause**: The user defined a state named `"x"`. The generated C++ function signature was `dynamics(const MSVec& x, ...)`. Inside, we unpacked `T x = x(0);`.
*   **Fix**: `MiniModel.py` now generates function arguments with `_in` suffix (`x_in`, `u_in`) and uses `tmp_` prefix for CSE variables to strictly isolate namespaces.

### 3. The "Vanishing Gradient" of Obstacles
*   **Symptom**: Solver gets stuck when initializing inside an obstacle. Gradients are tiny or zero.
*   **Cause**: Constraint $R^2 - (x^2 + y^2) \le 0$. At the center $(0,0)$, the gradient is $2x = 0$.
*   **Fix**: We rewrote the constraint as **Linear Distance**: $R - \sqrt{x^2 + y^2 + \epsilon} \le 0$. The gradient is now normalized (unit vector), providing a constant "push-out" force even at the center.

### 4. Template Hell with Eigen Alignment
*   **Symptom**: `std::vector<Knot>` crashes or fails to compile when passing to functions.
*   **Cause**: Eigen fixed-size vectorization requires 16/32-byte alignment. Standard `vector` doesn't guarantee this without `aligned_allocator`.
*   **Fix**:
    1.  Added `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` to `KnotPoint`.
    2.  Refactored helper functions to be templates `template<typename TrajVector>`, accepting any container type (vector, array, span) to decouple from the allocator type.

### 5. "Why compute Cost if we overwrite it?" (Feasibility Restoration)
*   **Observation**: In `feasibility_restoration`, we call `compute()` (which calculates Cost), but immediately `setIdentity(Q)`.
*   **Reason**: `compute()` calculates *everything* (Dynamics A, B and Cost Q, R). We need A, B for the restoration step, but we must **replace** the original Cost with a Minimum-Norm objective ($\min \|dx\|^2$) to find the closest feasible point.
*   **Optimization**: We split `compute` into `compute_dynamics`, `compute_constraints`, `compute_cost` to allow selective computation, saving FLOPs.

---

## 🛠️ Debugging Strategies

### 1. Enable PrintLevel::DEBUG
In `main.cpp`:
```cpp
config.print_level = PrintLevel::DEBUG;
```
Look for:
*   `MinSlack`: If this hits `1e-9` while `PrimInf` is large, you are blocked by a constraint boundary.
*   `Alpha`: If small (< 0.01) repeatedly, the Line Search is blocking progress (shape of the barrier is too steep). Trigger **Slack Reset**.
*   `Reg`: If large, the Hessian is non-convex or singular.

### 2. Analyzing Logs
*   **DualInf Explosions** (`1e10`): Inconsistent gradients. Check if `codegen` derivative logic matches the integration scheme (RK4 vs Euler). MiniSolver uses SymPy to guarantee this consistency.
*   **PrimInf Stagnation**: Local infeasible minimum. Use **Feasibility Restoration**.

### 3. Visualizing
Run `./run_demo.sh` and open `trajectory_plot.png`.
*   If the trajectory "teleports", check if `dt` matches the dynamics timescale.
*   If it vibrates around the obstacle, increase `reg_min` or `mu_min`.

---

## 🔮 Future Development

1.  **Sparse Riccati**: Currently we treat 4x4 matrices as dense. For larger systems (N=1000), exploiting sparsity in $A, B$ becomes crucial.
2.  **GPU Backend**: The `include/core/gpu_types.h` and `src/cuda/` files are placeholders for a Parallel Scan implementation of Riccati (O(log N)).
3.  **SQP-RTI**: Real-time iteration scheme (one QP per time step) for >1kHz control.

---
*Maintained by the MiniSolver Team.*
