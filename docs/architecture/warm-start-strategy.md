# Warm-Start Strategy

Date: 2026-05-02

Status: design accepted for implementation

Related:

- `include/minisolver/algorithms/initialization.h`
- `include/minisolver/core/solver_options.h`
- `include/minisolver/solver/solver.h`

## Goal

MiniSolver should support repeated MPC solves where adjacent problem instances
are close enough that a full cold-start interior-point solve is unnecessary.
The target behavior is:

```text
same solver instance + shifted business trajectory + small parameter/state change
  -> reuse valid primal-dual state
  -> preserve a barrier parameter consistent with complementarity
  -> converge in a small number of full SQP/IPM iterations
```

This is distinct from `enable_rti=true`. RTI intentionally exits after a fixed
single iteration. Warm-start should make ordinary `solve()` converge quickly
when the previous solution is already a good initial guess.

## Evidence From Mature Systems

| System | Behavior | MiniSolver takeaway |
| --- | --- | --- |
| Ipopt | Warm start is explicit (`warm_start_init_point`) and uses separate interior-push options for bounds, slacks, and multipliers. It also has same-structure and target-`mu` concepts. Barrier update can use average complementarity in adaptive modes. | Do not blindly trust user-provided primal-dual variables. Repair/push them inside the interior and make `mu` explicit. |
| CasADi `nlpsol` | Inputs include primal and multiplier guesses (`x0`, `lam_x0`, `lam_g0`); outputs can be fed back as the next input. | Treat warm-start state as explicit data flow, not hidden magic. |
| acados / HPIPM | QP warm-start has levels: cold, warm, hot, very hot. HPIPM initializes or clips inequality slacks/multipliers (`t`, `lam`) depending on the level and `t0_init`. | Provide strategy levels and clipping/repair rather than a single boolean. |
| FORCESPRO PDIP | Cold/centered/primal warm starts initialize slacks and multipliers so complementarity is tied to `mu0`; primal warm-start computes slacks from residuals. | If only primal is reused, rebuild slacks/duals on the central path. |
| FORCESPRO SQP | SQP solvers store the previous solution internally and use it on subsequent calls unless reinitialized. | A persistent solver object should make MPC-style reuse easy, while still allowing explicit reinitialization. |
| do-mpc | The full MPC solution vector is also the next initial guess (`opt_x_num`). | High-level MPC code commonly handles trajectory reuse at the application layer. |
| OSQP | Repeated solves can automatically reuse primal and dual variables; matrix factorization can be cached when sparsity is unchanged. | Same-structure reuse and iterate reuse are separate concerns. MiniSolver already keeps fixed workspaces; warm-start should focus on iterate/barrier state. |

Primary references:

- Ipopt options: https://coin-or.github.io/Ipopt/OPTIONS.html
- CasADi NLP solver interface: https://web.casadi.org/api/html/d4/d89/group__nlpsol.html
- acados Python options: https://docs.acados.org/python_interface/index.html
- FORCESPRO solver options: https://forces-feature.embotech.com/Documentation/solver_options/index.html
- FORCESPRO SQP high-level interface: https://forces-6-3-0.embotech.com/Documentation/high_level_interface/index.html
- do-mpc MPC API: https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPC.html
- OSQP Python interface: https://osqp.org/docs/interfaces/python.html

## Design Boundaries

Warm-start has three independent layers:

| Layer | Data | Owner |
| --- | --- | --- |
| Primal guess | `x/u` trajectories and parameters | User/business code, via existing setters |
| Primal-dual guess | `s/lam/soft_s` interior-point variables | Solver, with user override allowed through existing setters |
| Algorithmic state | `mu`, `reg`, filter/merit history | Solver warm-start kernel |

MiniSolver should not reintroduce built-in horizon shifting. Users should shift
the business trajectory explicitly. The solver should make the shifted iterate
usable for a fast warm-started IPM solve.

## Initial Strategy Set

The first implementation should be internal and configuration-driven, not a
public OOP strategy framework.

```cpp
enum class WarmStartBarrierMode {
    RESET_TO_MU_INIT,
    REUSE_PREVIOUS_MU,
    FROM_COMPLEMENTARITY_GAP
};

enum class WarmStartRegularizationMode {
    RESET_TO_REG_INIT,
    REUSE_PREVIOUS_REG,
    DECAY_PREVIOUS_REG
};
```

Default config must remain conservative:

```cpp
initialization = COLD_START;
warm_start_barrier = RESET_TO_MU_INIT;
warm_start_regularization = RESET_TO_REG_INIT;
```

Recommended MPC candidate:

```cpp
initialization = REUSE_PRIMAL_DUAL;
warm_start_barrier = FROM_COMPLEMENTARITY_GAP;
warm_start_regularization = RESET_TO_REG_INIT;
```

The reason to prefer `FROM_COMPLEMENTARITY_GAP` over `REUSE_PREVIOUS_MU` is that
it binds the barrier problem to the actual stored `s/lam/soft_s` state. If the
user or business logic edits the primal-dual guess, blindly reusing the previous
`mu` can make complementarity residuals inconsistent.

`reg` is different from `mu`: it is a numerical recovery state, not a barrier
path state. The default and recommended MPC strategy should reset it. Reuse and
decay modes are kept for expert tuning and benchmarking.

## WarmStartKernel Contract

`WarmStartKernel` should run during `presolve()` before model-based slack/dual
initialization:

1. Detect whether `REUSE_PRIMAL_DUAL` is requested and the stored primal-dual
   state is finite/interior-valid.
2. If primal-dual reuse is invalid, fall back to model-based primal-dual
   initialization and `mu_init`.
3. If primal-dual reuse is valid, select `mu` by the configured barrier mode.
4. Select `reg` by the configured regularization mode.
5. Reset solve counters, metrics, and line-search/filter history.
6. Leave trajectory shifting outside the solver.

`FROM_COMPLEMENTARITY_GAP` should compute the average complementarity over:

```text
hard pair: s_i * lam_i
L1 soft pair: soft_s_i * (w_i - lam_i)
```

Then clamp the result to a safe range:

```text
mu_final <= mu <= mu_init
```

If the average gap is not finite or no valid complementarity pair exists, fall
back to `mu_init`.

## Validation Plan

Required unit tests:

- Default behavior still resets `mu` to `mu_init` and `reg` to `reg_init`.
- `REUSE_PREVIOUS_MU` preserves a finite previous `mu`, clamped to
  `[mu_final, mu_init]`.
- `FROM_COMPLEMENTARITY_GAP` sets `mu` from stored `s*lam` and L1
  `soft_s*(w-lam)`.
- Invalid primal-dual guesses fall back to model-based initialization and
  `mu_init`.
- Regularization reset/reuse/decay modes are explicit and captured in snapshots.

Required benchmark:

- Repeated same-structure MPC-like solves with small state/reference changes.
- Compare:
  - cold/default reset.
  - `REUSE_PRIMAL_DUAL + RESET_TO_MU_INIT`.
  - `REUSE_PRIMAL_DUAL + REUSE_PREVIOUS_MU`.
  - `REUSE_PRIMAL_DUAL + FROM_COMPLEMENTARITY_GAP`.
- Report success rate, mean iterations after the first solve, median time, and
  worst iteration count.

Initial local result on `tools/warm_start_bench.cpp`:

```text
strategy,success_rate,avg_iters_after_first,worst_iters_after_first,avg_ms_after_first
adaptive_primal_reset,1.000000,7.050847,11,0.007727
adaptive_pd_reset_mu,1.000000,3.525424,7,0.004012
adaptive_pd_reuse_mu,1.000000,3.525424,7,0.003996
adaptive_pd_gap_mu,1.000000,3.525424,7,0.004207
monotone_pd_reset_mu,1.000000,9.423729,12,0.009993
monotone_pd_reuse_mu,1.000000,3.525424,7,0.004651
monotone_pd_gap_mu,1.000000,3.525424,7,0.004181
```

Interpretation:

- Reusing primal-dual state matters more than the exact barrier warm-start mode
  when the solver uses the default adaptive barrier update; adaptive `mu` already
  pulls reset `mu_init` down toward the current complementarity gap before the
  direction solve.
- With monotone barrier update, preserving a gap-consistent `mu` matters: reset
  `mu_init` roughly doubles the iteration count in this benchmark.
- Recommended general MPC preset remains:

```cpp
initialization = REUSE_PRIMAL_DUAL;
warm_start_barrier = FROM_COMPLEMENTARITY_GAP;
warm_start_regularization = RESET_TO_REG_INIT;
barrier_strategy = ADAPTIVE;
```

This preset is as fast as previous-`mu` reuse in the benchmark, but it is safer
when application code edits slacks/duals or restores a snapshot iterate because
`mu` is recomputed from the actual complementarity state.
