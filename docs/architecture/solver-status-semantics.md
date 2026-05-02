# Solver Status Semantics

MiniSolver status values report the solver's terminal condition, not just the last residual value.
The postsolve residual refresh remains authoritative for solution quality, but it must not collapse
iteration-budget or algorithmic failures into `INFEASIBLE` unless the solver has evidence that the
problem itself is infeasible.

## Reference From Mature Solvers

Mature solvers distinguish at least these classes:

- successful solve: exact or acceptable/reduced-accuracy solution;
- infeasibility certificate or detected infeasibility;
- resource limits: maximum iterations or time limit;
- globalization failure: minimum step / insufficient progress;
- linear-system or QP subsolver failure;
- invalid input or invalid numbers.

Ipopt exposes separate statuses for `Solve_Succeeded`, `Solved_To_Acceptable_Level`,
`Infeasible_Problem_Detected`, `Search_Direction_Becomes_Too_Small`,
`Maximum_Iterations_Exceeded`, `Restoration_Failed`, `Error_In_Step_Computation`, and invalid
problem/number cases. acados separates success, NaN, max-iteration, min-step, QP failure,
unbounded, timeout, and QP-scaling statuses. OSQP and Clarabel similarly separate solved,
reduced-accuracy solved, infeasible, max-iteration/time, and numerical-error statuses.

## MiniSolver Policy

Detailed termination residual semantics are tracked in
[`termination-design.md`](termination-design.md). In short: `SolverStatus`
should stay compact, while residual channels and loop-exit reasons should be
made explicit before changing convergence behavior.

MiniSolver keeps the public API compact and config-driven. It does not expose a public plugin status
framework. Statuses should be added only when the solver can route them from existing evidence.
`solve()` continues to return one primary `SolverStatus`; `get_info()` exposes the last solve's
fixed-size `SolverInfo`, including `status`, `loop_status`, `termination_reason`, residuals,
iteration count, final `mu`, last `alpha`, and coarse event flags.

Status groups:

- `UNSOLVED`: internal/intermediate state only.
- `OPTIMAL`: fresh postsolve residuals satisfy strict primal, dual, and true
  complementarity tolerances. The internal barrier target `mu` is diagnostic
  state, not a direct final-quality gate.
- `FEASIBLE`: fresh postsolve residuals are primal-acceptable but not fully optimal.
- `MAX_ITER`: iteration budget exhausted and the final iterate is not acceptable.
- `STEP_TOO_SMALL`: globalization collapsed to a tiny step and no restoration path was available.
- `INSUFFICIENT_PROGRESS`: progress/cost stagnation stopped the loop before convergence.
- `LINEAR_SOLVE_FAILED`: Riccati/KKT direction solve failed after configured retries.
- `RESTORATION_FAILED`: feasibility restoration was attempted and failed to produce an acceptable
  recovery.
- `INVALID_INPUT`: reserved for config/model input validation failures.
- `NUMERICAL_ERROR`: invalid arithmetic or invalid search directions after a solve.
- `INFEASIBLE`: reserved for cases where MiniSolver has explicit infeasibility evidence. Because
  MiniSolver currently has no formal infeasibility certificate, this status should not be used for
  plain max-iteration exhaustion or restoration failure.

`UNBOUNDED` is intentionally not exposed yet. MiniSolver currently has no unboundedness detector, so
adding that status would create an API promise without evidence.

Postsolve precedence:

1. Fatal numerical/input failures may return directly if the active iterate is not trustworthy.
2. For non-fatal termination reasons, refresh residuals on the active trajectory.
3. If fresh residuals prove `OPTIMAL` or `FEASIBLE`, return that solution-quality status.
4. Otherwise return the loop termination reason (`MAX_ITER`, `STEP_TOO_SMALL`,
   `RESTORATION_FAILED`, etc.) instead of collapsing to `INFEASIBLE`.

This gives MPC users actionable information: increase `max_iters`, inspect line-search/restoration,
or inspect linear-solver conditioning, instead of treating every unacceptable final residual as a
mathematical infeasibility claim.

`TerminationReason` is more specific than `SolverStatus`: for example a solve can return
`FEASIBLE` while `termination_reason == COST_STAGNATION`, meaning the loop stopped because progress
stalled and postsolve then found the final iterate primal-acceptable. Likewise, fixed-iteration
NMPC can return whatever final quality postsolve proves while keeping
`termination_reason == FIXED_ITERATION`.
