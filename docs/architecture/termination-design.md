# Termination Design

MiniSolver termination should be treated as a solver contract, not a small
condition tweak. The current implementation is intentionally left unchanged
until the metrics and tests below are in place.

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
- `SolverStatus` now distinguishes quality (`OPTIMAL`, `FEASIBLE`) from several
  exit causes (`MAX_ITER`, `STEP_TOO_SMALL`, `INSUFFICIENT_PROGRESS`,
  `LINEAR_SOLVE_FAILED`, `RESTORATION_FAILED`, `INVALID_INPUT`,
  `NUMERICAL_ERROR`).

Internal residuals:

- `StepResidualSummary::barrier_mu` snapshots the internal barrier target used
  for that derivative evaluation.
- `max_barrier_complementarity_residual` currently means
  `max |s * lambda - mu|`, including the L1 soft secondary pair.
- `avg_complementarity_gap` is computed for barrier updates.
- postsolve refresh recomputes primal and dual residuals before classifying
  final quality.

Known semantic gap:

- `TerminationKernel::check_convergence()` currently requires `mu <= mu_final`
  and centrality residual `|s * lambda - mu|`.
- Standard KKT quality is about primal residual, stationarity residual, and
  true complementarity gap `s * lambda`, not just closeness to the current
  barrier target.
- Simply removing `mu <= mu_final` is unsafe unless the solver also tracks true
  complementarity. A large `mu` with `s * lambda ~= mu` is centered, but not
  necessarily optimal.

## Recommended Contract

Keep `solve()` returning one `SolverStatus`, but make termination internally
produce two concepts:

- `SolutionQuality`: quality of the final iterate.
- `TerminationReason`: why the loop stopped.

`SolverStatus` remains the compact public summary for now:

- return `OPTIMAL` only when fresh residuals satisfy the strict quality criteria;
- return `FEASIBLE` when primal feasibility is acceptable but strict quality is
  not met;
- otherwise return the loop termination reason, not a guessed infeasibility
  claim.

Add a future `SolverInfo` / `SolverReport` only after the internal fields are
stable. It can contain:

```cpp
struct SolverInfo {
    SolverStatus status;
    SolverStatus loop_exit_status;
    int iterations;

    double primal_inf;
    double dual_inf;
    double complementarity_inf;
    double barrier_centrality_inf;
    double mu;
    double alpha;
    double objective;

    bool restoration_used;
    bool degraded_direction;
    bool postsolve_refreshed;
};
```

This is intentionally an info object, not a plugin or public strategy layer.

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

Use Option B for strict `OPTIMAL`, plus Option C for practical NMPC exits.

Implementation should not start by editing `TerminationKernel`. First add the
missing metrics and tests, then change semantics in a small commit.

## Implementation Order

1. Add internal metrics only:
   - `max_complementarity_gap` to `StepResidualSummary` and `PostsolveResiduals`;
   - keep `max_barrier_complementarity_residual` for centrality diagnostics;
   - update iteration log only if needed, not by default.

2. Add focused tests before changing behavior:
   - residual snapshot records both true gap and barrier centrality residual;
   - large `mu`, small true complementarity does not fail solely because of
     `mu_final`;
   - large true complementarity does not pass even if `mu <= mu_final`;
   - L1 soft pair contributes to true complementarity;
   - postsolve uses fresh residuals, not stale in-loop metrics.

3. Change `TerminationKernel`:
   - strict convergence uses `primal_inf`, `dual_inf`, and true
     `complementarity_inf`;
   - centrality residual remains a diagnostic and barrier-update signal;
   - `mu_final` remains the lower bound for barrier scheduling, not a direct
     final-quality gate.

4. Update docs and status semantics:
   - explain `OPTIMAL` vs `FEASIBLE`;
   - document tolerance units and scaling assumptions;
   - mark `tol_grad` as unresolved until stationarity vs dual infeasibility
     naming is settled.

5. Run validation:
   - targeted termination/residual tests;
   - full ctest;
   - small benchmark smoke comparing status distribution before/after;
   - only then decide whether benchmark reference data needs updating.

## Deferred Items

- `tol_grad`: do not wire or remove it in the termination pass. It needs a
  separate decision on whether MiniSolver exposes both stationarity and dual
  infeasibility, or treats them as the same metric.
- infeasibility/unboundedness certificates: do not add statuses that imply
  certificates until the solver can produce evidence.
- acceptable/reduced-accuracy public status: keep using `FEASIBLE` until there
  is a concrete need for a separate `ACCEPTABLE` status.

