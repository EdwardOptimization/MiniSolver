# Solver Development Principles

These are the small principles to keep in mind while changing MiniSolver. They
are not a process checklist; process details live in the agent harness, review
triage checklist, and testing matrix.

## Principles

1. Put invariants at the boundary, not in the hot path.
   If config validation, setters, snapshot loading, or codegen can guarantee a
   condition, do not add scattered defensive checks inside solver internals.

2. Do not restrict valid modeling freedom.
   Solver-core validation should protect solver invariants, not forbid model
   formulations that can be meaningful for users.

3. Public API is expensive.
   Prefer internal seams, existing setters, and `SolverConfig` over public
   plugins, strategy objects, or DTOs that only clean up internal code.

4. Keep the canonical solve route simple.
   `solve()` and the main loop should not accumulate side concerns. Put monitor,
   projection, trial-building, snapshot, and diagnostics behavior behind clear
   owners.

5. Every stateful concept needs an owner.
   Do not keep adding loose fields to shared context structs. Stateful behavior
   should live in the smallest object that owns its lifecycle.

6. Make data projection explicit.
   Runtime state, public `SolverInfo`, snapshot data, and diagnostics are
   different shapes. Keep their projection rules visible instead of relying on
   incidental synchronization.

7. New concepts should reduce change amplification.
   If a small feature forces repeated edits across config, validation,
   snapshot, status, info, and tests, first look for a single source of truth or
   a narrower internal contract.

8. Do not refactor numerical cores for appearance.
   Riccati, matrix kernels, barrier update, line search, and restoration should
   move because of a bug, evidence, or a concrete duplication pain point, not
   because a cleaner abstraction is available.

9. Performance and zero-allocation claims need evidence.
   Use benchmarks, allocation tests, and accuracy/residual checks together.
   Do not infer speed or real-time safety from code shape alone.

10. Keep external-system semantics outside solver core.
    Controller timing, benchmark datasets, cross-solver fairness, and
    application geometry belong in the controller, benchmark, or model/codegen
    layer unless they are generic solver concepts.
