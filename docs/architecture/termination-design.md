# Termination Design

MiniSolver termination is a solver contract, not a small condition tweak. The
current preview implementation has a stable core contract: strict `OPTIMAL` uses
true complementarity instead of the internal barrier target, fixed-iteration RTI
behavior is selected through `SolverConfig`, `ACCEPTABLE_NMPC` has explicit
primal-feasible early exits, residual/cost stagnation exits are separated from
infeasibility claims, and `SolverInfo` records the final solution quality, loop
exit status, termination reason, and residual snapshot.

## Design Question

Define how MiniSolver decides:

- whether the current iterate is `OPTIMAL` or only `FEASIBLE`;
- why the solve loop stopped when it is not optimal;
- which residual snapshot is authoritative;
- how internal IPM state such as `mu` relates to final KKT quality.

Out of scope for this pass:

- changing globalization, filter, SOC, or restoration behavior;
- adding public strategy/plugin objects;
- adding infeasibility or unboundedness certificates before the solver can
  actually produce them.

## References

Mature solvers separate solution quality from termination cause and expose the
residuals that justify the status.

- Ipopt successful termination requires a scaled NLP error plus absolute
  thresholds on dual infeasibility, constraint violation, and complementarity.
  It also has a separate acceptable-level termination path and resource-limit
  exits.
  Source: https://sources.debian.org/src/coinor-ipopt/3.11.9-2.3/Ipopt/doc/options.tex/
- acados exposes NLP residual channels `res_stat`, `res_eq`, `res_ineq`, and
  `res_comp`, plus solver/QP status and iteration statistics.
  Source: https://docs.acados.org/python_interface/index.html
- OSQP separates status values such as solved, solved inaccurate, primal/dual
  infeasible, maximum iterations, time limit, and numerical/non-convex errors;
  its convergence check is residual-based.
  Sources: https://osqp.org/docs/interfaces/status_values.html and
  https://osqp.org/docs/solver/index.html
- Clarabel returns status codes such as `SOLVED`, `ALMOST_SOLVED`,
  infeasibility certificates, max-iteration/time, numerical error, and
  insufficient progress.
  Source: https://clarabel.org/stable/julia/getting_started_jl/
- CasADi's Ipopt interface records both `return_status` and iteration metrics
  such as infeasibility, `mu`, step lengths, and objective values.
  Source: https://web.casadi.org/api/html/dd/d36/ipopt__interface_8hpp_source.html

## Current MiniSolver State

Public surface:

- `SolverStatus solve()` is the primary result.
- `const SolverInfo& get_info() const` exposes the last solve's structured
  diagnostics without changing `solve()` return semantics.
- `SolverStatus` now distinguishes quality (`OPTIMAL`, `FEASIBLE`) from several
  exit causes (`MAX_ITER`, `STEP_TOO_SMALL`, `INSUFFICIENT_PROGRESS`,
  `LINEAR_SOLVE_FAILED`, `RESTORATION_FAILED`, `INVALID_INPUT`,
  `NUMERICAL_ERROR`).
- `TerminationReason` records why the loop stopped, e.g. strict convergence,
  fixed-iteration profile, cost stagnation, line-search failure, max iteration,
  or postsolve infeasibility.

Internal residuals:

- `StepResidualSummary::barrier_mu` snapshots the internal barrier target used
  for that derivative evaluation.
- `max_barrier_complementarity_residual` currently means
  `max |s * lambda - mu|`, including the L1 soft secondary pair.
- `avg_complementarity_gap` is computed for barrier updates.
- postsolve refresh recomputes primal and dual residuals before classifying
  final quality.

Resolved first-pass semantic gap:

- `TerminationKernel::check_convergence()` uses primal infeasibility, dual
  infeasibility, and true complementarity gap.
- `max_barrier_complementarity_residual` remains available as centrality
  diagnostics and for barrier scheduling, but it is no longer the strict
  `OPTIMAL` certificate.
- `mu_final` remains the lower bound for internal barrier scheduling.

## Recommended Contract

Keep `solve()` returning one `SolverStatus`, but make termination internally
produce two concepts:

- `SolutionQuality`: quality of the final iterate.
- `TerminationReason`: why the loop stopped.

`SolverStatus` remains the compact public summary:

- return `OPTIMAL` only when fresh residuals satisfy the strict quality criteria;
- return `FEASIBLE` when primal feasibility is acceptable but strict quality is
  not met;
- otherwise return the loop termination reason, not a guessed infeasibility
  claim.

`SolverInfo` is a fixed-size diagnostic object, not a plugin or strategy layer.
It intentionally has no dynamic containers so reading it does not compromise the
zero-malloc solve contract. Selected fields include:

```cpp
struct SolverInfo {
    SolverStatus status;
    SolverStatus loop_status;
    TerminationReason termination_reason;
    int iterations;

    double primal_inf;
    double dual_inf;
    double complementarity_inf;
    double barrier_centrality_inf;
    double mu;
    double alpha;

    bool linear_ok;
    bool line_search_failed;
    bool restoration_used;
    bool degraded_step;
    int regularization_escalation_count;
    int soc_attempt_count;
    int restoration_attempt_count;
};
```

See `include/minisolver/core/types.h` for the authoritative field list.

This is intentionally an info object, not a public OOP strategy layer.

## Loop Exit Precedence

The solve loop treats termination as a single explicit decision after each
iteration. Current precedence is:

| Priority | Loop condition | Loop status | Reason |
| --- | --- | --- | --- |
| 1 | Explicit step result, e.g. numerical error, linear solve failure, tiny step, restoration failure, invalid input, or residual stagnation | step result status | concrete step reason |
| 2 | `RTI_FIXED_ITERATION` after any non-fatal step | `UNSOLVED` until postsolve verifies quality | `FIXED_ITERATION` |
| 3 | Strict KKT candidate from fresh same-iterate residuals | `OPTIMAL` | `CONVERGED` |
| 4 | Accepted `ACCEPTABLE_NMPC` primal-feasible shortcut | `FEASIBLE` | `PRIMAL_FEASIBLE` |
| 5 | Cost stagnation, only when no model-update callback is installed | `INSUFFICIENT_PROGRESS` | `COST_STAGNATION` |
| 6 | Iteration budget exhausted | `MAX_ITER` | `MAX_ITERATIONS` |

Postsolve still refreshes residuals before returning the final status. A loop
exit reason describes why the iteration loop stopped; final `SolverStatus`
describes the quality that postsolve could prove for the returned iterate.
Residual stagnation is always a loop-level `INSUFFICIENT_PROGRESS` exit with
`RESIDUAL_STAGNATION`; if the stalled iterate is primal-acceptable, postsolve
may still upgrade the final status to `FEASIBLE`.

## Residual Definitions

MiniSolver should distinguish these metrics:

| Metric | Meaning | Used for |
| --- | --- | --- |
| `primal_inf` | max dynamics defect and true constraint violation | `OPTIMAL`, `FEASIBLE`, restoration decisions |
| `dual_inf` | stationarity / Lagrangian-gradient residual | `OPTIMAL` |
| `complementarity_inf` | max true complementarity gap, e.g. `s * lambda` and `soft_s * (w - lambda)` | `OPTIMAL` |
| `barrier_centrality_inf` | max centrality residual `|s * lambda - mu|` | barrier update, debugging, centering diagnostics |
| `avg_complementarity_gap` | average true complementarity gap | adaptive barrier / Mehrotra target updates |
| `mu` | internal barrier target | algorithm state, warm start, barrier schedule |

This avoids the current ambiguity where `tol_mu` can mean either barrier target
progress or true KKT complementarity.

## Options

### Option A: Keep Current Strict Barrier Semantics

`OPTIMAL` requires `mu <= mu_final` and `|s * lambda - mu|` small.

Pros:

- smallest code change;
- conservative along the current central path;
- keeps existing tests stable.

Cons:

- `mu_final` becomes part of final solution quality, not just an algorithmic
  target;
- users can set loose KKT tolerances and still fail to reach `OPTIMAL`;
- cannot clearly explain `tol_mu` as true complementarity tolerance.

### Option B: KKT Quality Semantics

`OPTIMAL` requires `primal_inf <= tol_con`, `dual_inf <= tol_dual`, and
`complementarity_inf <= tol_mu`. `mu` remains an internal barrier target.

Pros:

- matches the solver-quality model used by mature NLP/IPM solvers;
- separates algorithm state from final certificate quality;
- makes `SolverInfo` and benchmark reporting easier to interpret.

Cons:

- requires adding true complementarity metrics to both step and postsolve
  residual snapshots;
- needs regression tests for cases where centrality and true complementarity
  disagree;
- may change existing status outcomes.

### Option C: Two-Level Quality

Strict `OPTIMAL` uses KKT quality; `FEASIBLE` or a future acceptable status uses
relaxed residuals and/or cost stagnation.

Pros:

- matches Ipopt-style desired vs acceptable termination;
- practical for NMPC where a useful feasible iterate is often enough;
- keeps `solve()` compact while `SolverInfo` carries details.

Cons:

- needs careful naming to avoid making `FEASIBLE` sound optimal;
- requires documenting exact acceptable thresholds.

## Recommendation

Use Option B for strict `OPTIMAL`, plus Option C for practical NMPC exits. The
first implementation pass follows this recommendation.

Implemented:

- `StepResidualSummary` and `PostsolveResiduals` carry `max_complementarity_gap`
  separately from `max_barrier_complementarity_residual`.
- `TerminationKernel` consumes an internal `TerminationSnapshot`.
- `SolverConfig::termination_profile` selects the internal termination mode.
- `TerminationProfile::RTI_FIXED_ITERATION` is the single RTI-style fixed
  iteration configuration entry point.
- `TerminationProfile::ACCEPTABLE_NMPC` has active primal-only loop exits for
  real-time NMPC:
  - Before the direction solve, a `REUSE_PRIMAL_DUAL` warm start may return
    `FEASIBLE` when the freshly evaluated primal residual already satisfies
    `primal_inf <= tol_con`. This shortcut is intentionally not used for cold
    starts, so a primal-feasible but unoptimized initial rollout still gets at
    least one direction-solve attempt. It is also disabled when a model-update
    callback is installed, because callback-updated objectives, references, or
    local models may require at least one direction solve even if primal
    feasibility is already satisfied.
  - After an accepted globalization step, MiniSolver refreshes primal feasibility
    on the accepted trajectory and returns `FEASIBLE` when `primal_inf <= tol_con`.
  These exits do not use `feasible_tol_scale`, stale dual residuals, or stale
  complementarity; postsolve still refreshes the final residual snapshot.
- Residual and cost stagnation monitors are disabled when a model-update callback
  is installed, because callbacks may change references, parameters, constraints,
  or local approximations between iterations, making cross-iteration residual/cost
  comparisons unreliable.
- Residual stagnation is evaluated on the current accepted iterate after fresh
  same-iterate primal/dual residuals are available and before applying another
  trial step. It is a current-iterate progress monitor, not a post-step progress
  test.
- Loop-exit decisions are centralized into one `LoopExitDecision` after each
  iteration. `run_solve_loop_()` now remains orchestration code: execute one
  iteration, ask for the loop-exit decision, commit the decision, then let
  postsolve verify the final solution quality.
- `SolverInfo` and `TerminationReason` expose the final residual snapshot and
  loop stop reason through `get_info()`.

Still deferred:

- acceptable/reduced-accuracy status beyond the existing `FEASIBLE` summary.
- scale-aware termination thresholds.

## Kernel Boundary Decision

Do not introduce separate public termination strategy objects for
`STRICT_KKT`, `ACCEPTABLE_NMPC`, and `RTI_FIXED_ITERATION` now. The current
boundary is intentionally smaller:

- `TerminationKernel` owns pure residual-quality predicates such as strict KKT
  convergence, accepted-step primal feasibility, postsolve quality
  classification, and cost-stagnation checks.
- `MiniSolver::run_solve_loop_()` owns loop-budget and loop-exit ordering,
  including fixed-iteration RTI behavior and fatal-failure precedence.
- Users select built-in behavior only through `SolverConfig::termination_profile`.
  Do not add a public OOP/plugin termination framework without a concrete
  external extension use case.

This keeps the hot loop predictable and avoids turning three stable built-in
profiles into premature strategy plumbing. It also preserves the important
semantic distinction that RTI fixed iteration is a budget/loop-exit policy, not
a residual convergence predicate.

## Current Refactor Boundary

The current state is intentionally a small explicit state machine, not a public
termination framework. Further refactoring is deferred unless a new behavior
creates a real maintenance problem. In particular:

- Keep `execute_solve_iteration_()` responsible for producing one
  `IterationResult` from the current iterate.
- Keep `run_solve_loop_()` responsible for orchestration only: run an iteration,
  consume one `LoopExitDecision`, and record the loop exit.
- Keep `postsolve()` responsible for fresh residual verification and final
  quality classification.
- Do not split `STRICT_KKT`, `ACCEPTABLE_NMPC`, or `RTI_FIXED_ITERATION` into
  public strategy objects. Users should continue to choose behavior through
  `SolverConfig`.
- Do not add presolve infeasibility detection or local infeasibility heuristics
  as part of termination cleanup. MiniSolver should not return infeasibility
  claims without evidence it can actually justify.

This boundary is acceptable for the preview solver. The next useful validation
step is a replay/benchmark corpus that exercises the profiles under continuous
NMPC-style solves, not more termination abstraction.

Future upgrade path:

- Introduce an internal `TerminationPlan` or `TerminationDecision` phase only
  when termination rules grow beyond the current lightweight predicates.
- Introduce `PostsolveVerdict` only if postsolve grows additional independent
  final-verdict paths; do not add it just for naming symmetry.
- Resolve that plan from `SolverConfig` at construction, `set_config()`, or the
  existing solve build boundary, not through virtual dispatch in the per-stage
  hot path.
- Keep the public API config-first unless users need stable third-party
  termination extensions.

Upgrade triggers:

- `ACCEPTABLE_NMPC` needs Ipopt-style consecutive acceptable iterations or
  multiple reduced-accuracy thresholds.
- RTI needs more than one level, time-budget exits, or adaptive iteration
  budgets.
- scale-aware termination adds both internal/scaled and physical/unscaled
  acceptance criteria.
- watchdog, trust-region, or recovery policies need to share a single
  structured termination decision with globalization/restoration.

## Implementation Order

1. Done: Add internal metrics:
   - `max_complementarity_gap` to `StepResidualSummary` and `PostsolveResiduals`;
   - keep `max_barrier_complementarity_residual` for centrality diagnostics;
   - update iteration log only if needed, not by default.

2. Done: Add focused tests before changing behavior:
   - residual snapshot records both true gap and barrier centrality residual;
   - large `mu`, small true complementarity does not fail solely because of
     `mu_final`;
   - large true complementarity does not pass even if `mu <= mu_final`;
   - postsolve uses fresh residuals, not stale in-loop metrics.

3. Done: Change `TerminationKernel`:
   - strict convergence uses `primal_inf`, `dual_inf`, and true
     `complementarity_inf`;
   - centrality residual remains a diagnostic and barrier-update signal;
   - `mu_final` remains the lower bound for barrier scheduling, not a direct
     final-quality gate.

4. Done: Add config-level profile:
   - `TerminationProfile::STRICT_KKT`;
   - `TerminationProfile::ACCEPTABLE_NMPC`;
   - `TerminationProfile::RTI_FIXED_ITERATION`.

5. Done: Add structured info:
   - `SolverInfo` records final `status`, `loop_status`, `termination_reason`,
     residual channels, `mu`, last accepted `alpha`, and coarse event flags;
   - `get_info()` is read-only and does not replace `solve()`;
   - fixed-size fields only, no solve-time allocation.

6. Done: Update docs and status semantics:
   - explain `OPTIMAL` vs `FEASIBLE`;
   - document tolerance units and scaling assumptions;
   - resolve the old `tol_grad` / `tol_dual` split by treating
     `dual_inf/tol_dual` as the stationarity / Lagrangian residual channel.

7. Run validation:
   - targeted termination/residual tests;
   - full ctest;
   - small benchmark smoke comparing status distribution before/after;
   - only then decide whether benchmark reference data needs updating.

## Deferred Items

- infeasibility/unboundedness certificates: do not add statuses that imply
  certificates until the solver can produce evidence.
- acceptable/reduced-accuracy public status: keep using `FEASIBLE` until there
  is a concrete need for a separate `ACCEPTABLE` status.
- additional `ACCEPTABLE_NMPC` criteria beyond fresh primal feasibility:
  implement only after continuous MPC benchmarks show which relaxed criteria
  preserve control quality.
- scale-aware termination: defer until scaling / constraint normalization has a
  stable design and tests.
- benchmark-driven tuning: run after `SolverInfo` fields are available in
  benchmark reports so status distribution, residuals, alpha, and iteration
  counts are comparable.
