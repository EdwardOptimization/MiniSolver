

# MiniSolver: High-Performance Embedded NMPC Library

![C++17](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Embedded-lightgrey)
![Zero-Malloc](https://img.shields.io/badge/Memory-Zero--Malloc-brightgreen)

**MiniSolver** is a professional-grade, header-only C++17 library for solving Nonlinear Model Predictive Control (NMPC) problems.

Engineered specifically for **embedded robotics** and **autonomous driving**, it combines the flexibility of Python-based symbolic modeling with the raw performance of highly optimized, template-generated C++ code. The default real-time configuration is designed for **Zero Dynamic Memory Allocation (Zero-Malloc)** during the solve phase; profiling and iteration logging are opt-in diagnostics and should stay disabled in hard real-time loops.

---

## 🚀 Key Features

### ⚡ Blazing Fast Performance
* **Riccati Recursion**: Utilizes a specialized block-tridiagonal linear solver ($O(N)$ complexity) tailored for optimal control structures.
* **SQP-RTI Support**: Real-Time Iteration (SQP-RTI) mode allows for **>1 kHz** control loops by performing a single quadratic programming sub-step per control tick.
* **Analytical Derivatives**: Uses SymPy to generate flattened, algebraically simplified C++ code for Jacobians and Hessians at compile-time, including true Gauss-Newton Hessians for explicit least-squares residuals.
* **🔥 Fused Riccati Kernels**: Unlike solvers that use generic matrix libraries, MiniSolver uses Python (SymPy) to symbolically fuse the Riccati backward pass (`Q + A'PA`) into a single, flattened C++ function. This eliminates all loop overhead and explicitly bypasses multiplication by zero, achieving **perfect sparsity exploitation** for small-to-medium systems ($N_x < 20$).

### 🛡️ Embedded Safety & Robustness
* **Zero-Malloc Solve Path**: The default solver configuration performs no `new`/`malloc` calls during `solve()`. Keep profiling and iteration logging disabled for hard real-time use.
* **Robust Interior Point Method (IPM)**:
    * **Mehrotra Predictor-Corrector**: Drastically reduces iteration counts by utilizing higher-order corrections.
    * **Filter Line Search**: Ensures global convergence without the tedious tuning of merit function parameters. The H-type filter history uses Pareto-frontier pruning so that strictly improving (θ, φ) sequences collapse to a single entry instead of inflating the legacy 1024-slot circular buffer; per-search and per-solve counters (`SolverInfo::filter_entries_pruned_total`, `filter_redundant_inserts_total`, `filter_max_history_size`) expose the realized pruning work.
    * **Feasibility Restoration**: Automatically triggers a minimum-norm recovery phase if the solver encounters infeasible warm starts. The quadratic-penalty rho can run in `RestorationPenaltyMode::FIXED` (legacy hardcoded value) or `VIOLATION_ADAPTIVE` (clamped multi-scale rescaling per restoration sub-iteration), with realized rho extremes surfaced in `SolverInfo::restoration_rho_min_used` / `restoration_rho_max_used`.
* **Riccati Inertia-Correction Diagnostics**: `SolverInfo::riccati_indefinite_blocks` and `SolverInfo::riccati_max_diagonal_perturbation` always surface what the existing SPD fallbacks (regularization escalation, small-Nu freeze, saturation/ignore-singular sweeps) did. The small-Nu freeze fallback additionally always flips `degraded_step` (and bumps `degraded_riccati_freeze_count`) regardless of mode. Set `riccati_robust_mode = RiccatiRobustMode::INERTIA_AWARE_DIAGNOSTICS` to *also* flip `degraded_step` for the non-freeze paths (general-path SPD retry, saturation, ignore-singular), so monitoring code can gate downstream control actions on the local QP staying cleanly SPD without any inertia correction.

### 🔧 Advanced Solver Capabilities
* **Second Order Correction (SOC)**: Handles highly nonlinear constraints (e.g., tight obstacle avoidance) by curving the search step.
* **Native Soft Constraints**: Supports L1 (Exact) and L2 (Quadratic) soft constraints via **Dual Regularization**, allowing for relaxation without increasing the problem dimensions (slack variables are handled implicitly).
* **Defect Rollout Refinement**: Optional correction pass for linearized dynamics defects after the Riccati solve. Configure it with `direction_refinement = DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT`; it is not a full KKT iterative-refinement method. For nonlinear unconstrained or weakly active problems, opt into `DirectionRefinementMode::FULL_KKT_ITERATIVE_REFINEMENT` to iterate the dynamics-defect rollout up to `direction_refinement_max_passes` times or until the rollout defect drops below `direction_refinement_tol`. The mode auto-degrades to a single primal pass whenever active inequality duals are detected, preserving the OD-005 dual-consistency contract; per-iteration counters and the last observed defect are exposed via `SolverInfo::direction_refinement_passes` and `SolverInfo::direction_refinement_last_defect`.

---

## 📊 Performance Benchmarks

Benchmarks performed on an Intel Core i7 (Single Thread) for a **Kinematic Bicycle Model with Obstacle Avoidance** ($N_x=6, N_u=2, N=50$).

| Strategy | Integrator | Line Search | Avg Time | Iterations | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TURBO** | Euler | Filter | **~0.8 ms** | 10-15 | Approx |
| **ROBUST (Rec)** | RK4 | **Filter** | **~2.4 ms** | 23 | **Optimal** |
| **STABLE** | RK4 | Merit | ~3.0 ms | 46 | Optimal |
| **ADAPTIVE** | RK4 | Filter | ~25.0 ms | 300* | Feasible |

*> Adaptive strategy may stagnate on feasible but high-cost solutions in scenarios with bad initial guesses (e.g., straight-line initialization).*

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
from minisolver.MiniModel import Dot, OptimalControlModel
import sympy as sp

model = OptimalControlModel(name="DroneModel")

# 1. Define Variables
px, py, vx, vy, vz = model.state("px", "py", "vx", "vy", "vz")
thrust = model.control("thrust")

# 2. Dynamics (f(x,u))
model.subject_to(Dot(px) == vx)
model.subject_to(Dot(py) == vy)
model.subject_to(Dot(vz) == thrust - 9.81)

# 3. Objective
model.add_residual(px - 0.0, weight=20.0) # 0.5 * 20 * (px - 0)^2
model.add_residual(thrust, weight=0.2)    # 0.5 * 0.2 * thrust^2

# 4. Constraints
model.subject_to( thrust <= 20.0 )     # Hard Constraint
# Soft Constraint (L1 Penalty via Dual Regularization)
model.subject_to( vz <= 5.0, weight=100.0, loss='L1' )

# 5. Generate C++ Code
model.generate("include/model")
```

### 2. Solve in C++

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
    SolverConfig config = solver.get_config();
    config.barrier_strategy = BarrierStrategy::MEHROTRA;
    config.enable_soc = true;
    solver.set_config(config);
    
    // Solve
    SolverStatus status = solver.solve();
    
    // Retrieve Optimal Control
    std::vector<double> u_opt = solver.get_control(0);
}
```

### 3. Build & Run

MiniSolver includes a one-click build script that handles dependency checking, code generation, and compilation.

```bash
./build.sh
```

## 📦 Embedded Deployment

1.  Generate your model headers on a PC: `python3 generate_model.py`
2.  Copy the `minisolver/` include folder and your generated headers to your MCU project.
3.  Define `USE_CUSTOM_MATRIX` to remove `Eigen3` dependency.
4.  Compile with `-O3`. **No external libraries required.**

### Embedded Logger Profile

For hard real-time / MCU targets where `<iostream>` is unwanted:

* CMake option `MINISOLVER_DISABLE_STREAM_LOGGER=ON` drops `<iostream>` from
  `minisolver/core/logger.h`, removes the `std::cout`/`std::cerr` fallback, and
  defaults `LoggerConfig::silent_fallback` to `true`. Install a `LogCallback`
  if you need messages (for example to route into a UART buffer).
* Runtime knob `LoggerConfig::silent_fallback` (default `false`) drops messages
  with no callback installed, even at error level. Use this when you want host
  builds to behave like the embedded profile without a recompile.
* Compile-time `MINISOLVER_LOG_LEVEL=MLOG_LEVEL_NONE` removes every `MLOG_*`
  call entirely.

### Embedded Build Profile (`MINISOLVER_EMBEDDED_PROFILE`)

When you want a one-flag MCU build instead of opting in to each knob
individually, configure with `-DMINISOLVER_EMBEDDED_PROFILE=ON`. The bundle:

* Forces `USE_CUSTOM_MATRIX=ON` and `USE_EIGEN=OFF` (no Eigen dependency).
* Forces `MINISOLVER_DISABLE_STREAM_LOGGER=ON` and adds the compile definition
  `MINISOLVER_LOG_LEVEL=0` so every `MLOG_*` call is removed.
* Defines `MINISOLVER_EMBEDDED_PROFILE` so model code can guard
  host-only diagnostics.
* Disables `MINISOLVER_BUILD_TESTS`, `MINISOLVER_BUILD_EXAMPLES`,
  `MINISOLVER_BUILD_TOOLS`, and `MINISOLVER_FETCH_DEPS` so the build never
  pulls non-embedded dependencies.

The profile also enables a single OBJECT target,
`minisolver_embedded_smoke`, which instantiates a minimal MiniSolver template
configuration. The CI job `embedded-arm-cortex-m4` cross-compiles this target
with `cmake/toolchains/arm-cortex-m4.cmake` and runs
`scripts/check_arm_size_budget.sh build_arm` to enforce a 256 KiB budget on
the resulting `.o` file. Tightening or loosening that budget should be
documented in `docs/testing/testing-matrix.md` in the same change.

### API Error Contract

Public setters return `ApiStatus`. `ApiStatus::OK` means the value was
accepted; any other value means the solver state was not mutated. Checked
scalar getters use `(int stage, int idx, double& out)` overloads and return
`ApiStatus` so production code can distinguish "value retrieved" from "stage
out of range".

### SolverConfig Preset Profiles

`include/minisolver/core/solver_config_profiles.h` exposes four named factory
functions so callers can document intent at the call site instead of hand-
configuring strategy fields:

```cpp
SolverConfig cfg = minisolver::make_default_config();   // production default
SolverConfig cfg = minisolver::make_reference_config(); // correctness baseline
SolverConfig cfg = minisolver::make_speed_config();     // throughput-oriented
SolverConfig cfg = minisolver::make_robust_config();    // ill-conditioned NMPC
```

* `make_reference_config` -- correctness-first baseline (MERIT line search,
  MONOTONE barrier, no SOC / restoration / direction refinement). Used as the
  regression-test reference.
* `make_default_config` -- default-constructed `SolverConfig` (FILTER line
  search, ADAPTIVE barrier, restoration on, SOC off, no direction refinement).
* `make_speed_config` -- low `max_iters`, `ACCEPTABLE_NMPC` termination,
  aggressive barrier, no SOC / restoration. Designed for warm-started MPC
  loops.
* `make_robust_config` -- MEHROTRA + filter + SOC + restoration +
  RUIZ_EQUILIBRATION problem scaling + dynamics-defect rollout refinement +
  tighter `mu_final`. Designed for one-shot solves on poorly-scaled problems.

Profiles only override solver-strategy fields; integration / cost / model
parameters stay at `SolverConfig` defaults so callers can layer their own
overrides on top.

### Coordinate-Scaling Hint

For NMPC problems whose control coordinates span several orders of magnitude,
opt in to the coordinate-scaling hint:

```cpp
SolverConfig cfg = solver.get_config();
cfg.coordinate_scaling = CoordinateScalingMethod::USER_SUPPLIED;
solver.set_config(cfg);

solver.set_control_scale("u_force", 1.0);  // already O(1)
solver.set_control_scale("u_steer", 0.1);  // small-magnitude control
```

The hint is consumed exclusively by the dual-stationarity termination metric:
`dual_inf = max_i |r_bar_i| * control_scale_i`, where `r_bar` is the Riccati-
projected control residual. It never rescales primal variables, the search
direction, dynamics Jacobians, or the Riccati recursion.

The API is intentionally control-only. Dual stationarity is reported via the
inf-norm of `r_bar`; state stationarity is eliminated by the Riccati
substitution and parameters are not optimisation variables, so neither has a
residual to weight. Earlier revisions exposed `set_state_scale` /
`set_parameter_scale` setters that stored values without affecting `dual_inf`
while `coordinate_scaling_active` reported `true`. They have been removed
until a state/parameter-aware termination metric exists.

`SolverInfo::coordinate_scaling_active` is `true` only when the strategy is
`USER_SUPPLIED` *and* at least one control scale differs from `1.0`. Setters
validate each scale against `config.coordinate_scale_min/max` (defaults
`[1e-6, 1e6]`), require finite values, and return `ApiStatus`.

For full state/control/parameter equilibration (which would also rescale
`dx/du`, Jacobians, Hessians, warm-start deltas, and Riccati blocks), see the
deferred Stage 5 section in `docs/architecture/scaling-normalization-design.md`.

### Warm-Start Barrier and Regularization Reuse

Two orthogonal opt-ins control how the previous solve's barrier parameter and
regularization carry into the next `solve()` call:

```cpp
SolverConfig cfg = solver.get_config();
cfg.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
cfg.warm_start_barrier = WarmStartBarrierMode::REUSE_PREVIOUS_MU;
cfg.warm_start_regularization = WarmStartRegularizationMode::DECAY_PREVIOUS_REG;
solver.set_config(cfg);
```

* `WarmStartBarrierMode` -- `RESET_TO_MU_INIT` (default), `REUSE_PREVIOUS_MU`,
  or `FROM_COMPLEMENTARITY_GAP`. The latter two require
  `InitializationMode::REUSE_PRIMAL_DUAL` and a valid stored slack/dual
  iterate; the solver transparently falls back to `mu_init` if the iterate
  fails validation.
* `WarmStartRegularizationMode` -- `RESET_TO_REG_INIT` (default),
  `REUSE_PREVIOUS_REG`, or `DECAY_PREVIOUS_REG`. Decay divides the previous
  regularization by `config.reg_scale_down` once per solve; reuse keeps it
  unchanged. Both are clamped to `[reg_min, reg_max]`.

Benchmark evidence (custom MiniMatrix backend, double-integrator tracking
problem, 60 neighbouring solves with horizon 20; output of
`tools/warm_start_bench`):

```text
strategy                              avg_iters  worst_iters  avg_ms
adaptive_primal_reset                 6.49       11           0.0077
adaptive_pd_reset_mu                  3.53       7            0.0043
adaptive_pd_reuse_mu                  3.53       7            0.0043
adaptive_pd_reuse_mu_decay_reg        3.53       7            0.0043
monotone_pd_reset_mu                  8.27       12           0.0091
monotone_pd_reuse_mu                  3.53       7            0.0042
monotone_pd_reuse_mu_decay_reg        3.53       7            0.0041
```

Takeaways: switching from `REUSE_PRIMAL` to `REUSE_PRIMAL_DUAL` cuts the
average iteration count by roughly 46% on this problem; reusing mu adds an
extra 57% on top of that under the `MONOTONE` barrier strategy. Regularization
reuse and decay are neutral on this well-conditioned case but remain
documented opt-ins for poorly-scaled problems where escalation cost dominates.
The `WarmStartReuseTest` regression suite locks the cumulative-iteration win
of `REUSE_PREVIOUS_MU + DECAY_PREVIOUS_REG` over `RESET_TO_MU_INIT +
RESET_TO_REG_INIT` on a small tracking model so future changes cannot silently
regress the warm-start contract.

### RTI-lite Warm Start

For repeated MPC solves where state deltas between control cycles are small,
opt into the RTI-lite mode to reuse the previous primal-dual iterate:

```cpp
SolverConfig cfg = solver.get_config();
cfg.enable_rti_lite = true;
cfg.rti_lite_max_linearization_age = 3;     // up to 3 reuses *and* up to 3 SQP/IPM iters per reuse
cfg.rti_lite_max_state_delta = 0.5;          // L2 state-delta gate (model units)
solver.set_config(cfg);
```

When the gates pass (previous solve was acceptable, state delta < threshold,
linearization age < budget), the next `solve()` call runs at most
`rti_lite_max_linearization_age` SQP/IPM iterations with `ACCEPTABLE_NMPC`
termination. The same `rti_lite_max_linearization_age` knob also caps the
number of consecutive reused solves before a full refresh is forced, so the
total "stale-linearization work" budget is bounded by a single integer. If
any gate fails the solver falls back to a full solve and resets the
linearization age. `SolverInfo::rti_lite_reused_linearization` and
`SolverInfo::rti_lite_linearization_age` report which path ran.
`set_config()` always invalidates the RTI-lite history so a strategy change
does not silently warm-start from an incompatible seed.

### MiniMatrix vs Eigen Microbenchmark

`tools/mini_matrix_vs_eigen_bench` measures the inner kernels exercised by
the Riccati backward pass (GEMM and LDLT factorization) at the small fixed
sizes typical of NMPC. Build it once per backend:

```bash
cmake -S . -B .build_custom -DUSE_CUSTOM_MATRIX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .build_custom --target mini_matrix_vs_eigen_bench
./.build_custom/mini_matrix_vs_eigen_bench

cmake -S . -B .build_eigen -DUSE_EIGEN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .build_eigen --target mini_matrix_vs_eigen_bench
./.build_eigen/mini_matrix_vs_eigen_bench
```

Reference numbers from a 2026-05-06 host run (release, native arch
disabled, single thread; ns/iter, lower is better):

```text
kernel             4x4    6x6    8x8    12x12
GEMM (MiniMatrix)  2.1   33.3   45.4   147.1
GEMM (Eigen3)      2.1   19.7   87.9   212.5
LDLT (MiniMatrix)  2.2   23.7   40.2   101.7
LDLT (Eigen3)     38.5   74.6  121.0   251.8
```

Read with care: 4x4 GEMM ties because both backends fully unroll. Eigen
beats MiniMatrix at 6x6 GEMM (the SIMD path kicks in earlier than the
template-unroll budget covers); MiniMatrix already wins at 8x8/12x12
GEMM by 1.4-2x because the unrolled `StaticFor` path keeps register
pressure low. LDLT factorization is uniformly faster on MiniMatrix
because Eigen's pivoting adds setup cost that does not pay off at these
sizes. The bench is the *evidence anchor* for any future kernel tuning,
including widening `MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK` or adding
SIMD to `matmul`; speculative changes without a measurable win on this
table will not land.

### Solve-Time Allocation Discipline

The default `SolverConfig` is allocation-free during `solve()`. Two flags can
break that contract; they are intentionally opt-in for diagnostics:

* `SolverConfig::print_level >= PrintLevel::ITER` enables the iteration log,
  which currently uses `std::stringstream` for header/row formatting.
* `SolverConfig::enable_profiling = true` records timing samples through
  dynamically-sized buffers.

`test_memory` enforces zero-malloc with both flags off across the supported
line-search and integrator combinations.

-----

## 📂 Project Structure

  * **`include/minisolver/core/`**: Core types (`KnotPoint`), memory-safe containers (`Trajectory`), and solver configuration/state.
  * **`include/minisolver/matrix/`**: Fixed-size matrix abstraction layer (`MiniMatrix`/`Eigen`).
  * **`include/minisolver/solver/`**: The main `MiniSolver` class orchestrating the IPM loop.
  * **`include/minisolver/algorithms/`**: Numerical engines:
      * `RiccatiSolver`: Block-tridiagonal linear system solver.
      * `LineSearch`: Filter and Merit function strategies.
  * **`python/minisolver/`**: The `MiniModel` DSL and C++ code generator.
  * **`examples/`**: Runnable generated-model examples, including the car tutorial and advanced bicycle case.
  * **`tools/`**:
      * `auto_tuner.cpp`: Monte-Carlo search for optimal solver configurations.
      * `replay_solver.cpp`: Debugging tool to replay solver snapshot states.
      * `benchmark_suite/`: MiniSolver configuration benchmark suite.
  * **`docs/`**: Design notes, ADRs, review ledgers, and testing plans.

## 🧭 Development Workflow

MiniSolver is developed with a multi-agent engineering workflow. AI coding and
review agents are used for implementation, independent review, benchmark
investigation, debugging, and documentation research. Final architecture
decisions, validation criteria, and releases are maintained by Edward Qu.

Non-trivial solver changes are expected to be evidence-driven: reproduce the
issue, add a focused test or benchmark, make the smallest defensible change, and
record validation results before trusting the change.

## 🤝 License & Citation

**MiniSolver** is licensed under the **Apache 2.0 License**.

If you use this software in academic work, please refer to `CITATION.cff`.

*Maintained by Edward Qu.*
