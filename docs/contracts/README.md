# MiniSolver Behavior Contracts

This directory contains stable behavior contracts for MiniSolver internals.
Contracts are not implementation plans. They define what a domain promises,
which module owns it, how failures are reported, and which evidence protects the
promise.

Use contracts when a behavior change would otherwise spread through several
files, alter solver status semantics, touch numerical boundaries, or affect
codegen/runtime assumptions.

## Files

| File | Purpose |
| --- | --- |
| [`_template.md`](_template.md) | Starting point for new contract documents. |
| [`contract-id-policy.md`](contract-id-policy.md) | Contract ID format, ownership, and stability rules. |
| [`solve-loop-contract.md`](solve-loop-contract.md) | Canonical solve route, phase order, callbacks, and final handoff. |
| [`status-semantics-contract.md`](status-semantics-contract.md) | Final status, loop status, reason, and `SolverInfo` meaning. |
| [`termination-contract.md`](termination-contract.md) | Strict convergence, feasible shortcuts, RTI, stagnation, and tiny-step semantics. |
| [`postsolve-contract.md`](postsolve-contract.md) | Final residual refresh and public status classification. |
| [`warm-start-contract.md`](warm-start-contract.md) | Initialization modes, primal-dual reuse, and warm-start scalar selection. |
| [`config-api-contract.md`](config-api-contract.md) | Config validation, mutation rules, backend preservation, and snapshot registry. |
| [`finite-check-boundary-contract.md`](finite-check-boundary-contract.md) | Where non-finite values are checked and how they surface. |
| [`residual-contract.md`](residual-contract.md) | Primal, dual, complementarity, centrality, and unscaled residual metrics. |
| [`scaling-contract.md`](scaling-contract.md) | Objective/constraint/problem scaling and scaled vs unscaled reporting. |
| [`barrier-mu-contract.md`](barrier-mu-contract.md) | Barrier parameter ownership and update semantics. |
| [`regularization-contract.md`](regularization-contract.md) | Linear-solve retry, regularization, and degraded-step diagnostics. |
| [`tolerance-contract.md`](tolerance-contract.md) | Meaning and use of convergence and feasible fallback tolerances. |
| [`initialization-contract.md`](initialization-contract.md) | Slack, dual, soft-slack, and warm-start initialization. |
| [`line-search-contract.md`](line-search-contract.md) | Candidate construction, alpha ownership, and accepted-step refresh. |
| [`merit-line-search-contract.md`](merit-line-search-contract.md) | Merit value, derivative, Armijo, and non-finite scalar behavior. |
| [`filter-line-search-contract.md`](filter-line-search-contract.md) | Filter history, switching, acceptance, and reset behavior. |
| [`riccati-contract.md`](riccati-contract.md) | KKT/Riccati direction solve, sigma convention, and soft derivatives. |
| [`restoration-contract.md`](restoration-contract.md) | Restoration entry, improvement, counters, and failure precedence. |
| [`soc-contract.md`](soc-contract.md) | Second-order correction trial, refresh, scaling, and diagnostics. |
| [`direction-refinement-contract.md`](direction-refinement-contract.md) | Optional dynamics-defect direction refinement. |
| [`hard-constraints-contract.md`](hard-constraints-contract.md) | Structural hard row residual and complementarity semantics. |
| [`soft-constraints-contract.md`](soft-constraints-contract.md) | L1, L2, mixed, and zero/tiny-weight soft semantics. |
| [`constraint-packets-contract.md`](constraint-packets-contract.md) | QP, true, terminal, and SOC constraint packet ownership. |
| [`row-scaling-contract.md`](row-scaling-contract.md) | Row-scale storage, reuse, and unscaled residual transformation. |
| [`model-interface-contract.md`](model-interface-contract.md) | Static C++ model interface consumed by MiniSolver. |
| [`minimodel-codegen-contract.md`](minimodel-codegen-contract.md) | Python DSL validation and generated model semantics. |
| [`generated-packets-contract.md`](generated-packets-contract.md) | Generated packet ownership and clear/full-overwrite behavior. |
| [`model-callback-contract.md`](model-callback-contract.md) | Callback timing, allowed updates, and structural mutation guard. |
| [`parameter-contract.md`](parameter-contract.md) | Per-knot parameters, setters, generated use, and trust boundary. |
| [`memory-allocation-contract.md`](memory-allocation-contract.md) | Solve-time zero-allocation claims and exclusions. |
| [`logging-profiling-contract.md`](logging-profiling-contract.md) | Logger/profiler behavior and performance boundary. |
| [`backend-contract.md`](backend-contract.md) | CPU/GPU backend policy and backend preservation. |
| [`matrix-backend-contract.md`](matrix-backend-contract.md) | Eigen/MiniMatrix selection, parity, and kernel policy. |
| [`build-config-contract.md`](build-config-contract.md) | CMake options, dependencies, and build flags. |
| [`solver-info-contract.md`](solver-info-contract.md) | Public `SolverInfo` field projection and diagnostics. |
| [`snapshot-replay-contract.md`](snapshot-replay-contract.md) | Snapshot format, compatibility, and atomic load failure. |
| [`diagnostics-contract.md`](diagnostics-contract.md) | Internal counters, traces, and diagnostic ownership. |
| [`explicit-integrator-contract.md`](explicit-integrator-contract.md) | Explicit/discrete generated dynamics packet behavior. |
| [`implicit-integrator-contract.md`](implicit-integrator-contract.md) | Implicit integration, Newton solve, and invalid packet marking. |
| [`numerical-jacobian-contract.md`](numerical-jacobian-contract.md) | Finite-difference Jacobian fallback. |

Each domain contract should appear in
[`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md).

## Contract Rules

1. Keep one behavior domain per file.
2. Give every normative requirement a stable contract ID.
3. Do not renumber or reuse retired IDs.
4. State inputs, outputs, owned state, invariants, and failure semantics.
5. Link every ID to tests, benchmark/replay evidence, or an explicit deferred
   evidence path.
6. Prefer internal fixed-size result/state contracts over public plugin APIs.
