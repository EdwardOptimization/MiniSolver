# MiniSolver Roadmap

This file tracks **milestones** (not daily progress). The goal is to keep the project aligned, avoid overdesign, and make decisions reviewable.

## Completed Milestones

### 2025-12 (Bootstrap And Core Features)

- Core NMPC solver brought up with configurable integration, barrier/regularization strategy, and solver status reporting.
- Python/SymPy `MiniModel` workflow established for symbolic modeling and C++ header generation.
- Early benchmark tooling and auto-tuning utilities added, along with snapshot/replay style debugging helpers.
- Soft constraints and basic nonlinear constraints integrated (foundation for obstacle-style problems).
- `MiniMatrix` (custom fixed-size backend) introduced and iterated, including early sparsity/fused-Riccati performance work.
- CI introduced with basic checks and a style gate.

### 2026-02 (Knot Refactor)

- `KnotPoint` refactor to clarify per-stage data layout and reduce coupling.
- Test suite refactor to improve regression coverage and reduce duplication.

### 2026-04 (Stabilization After Refactors)

- KnotPoint / Trajectory refactor stabilized with regression coverage.
- Solver public API surface tightened; configuration is centralized in `SolverConfig` via `set_config()` / `get_config()`.
- Snapshot/serializer moved away from raw `sizeof(SolverConfig)` dumps; format is explicit and includes soft-constraint state where required.
- Matrix backend selection wired end-to-end (`USE_EIGEN` vs `USE_CUSTOM_MATRIX`), with CI/test coverage on both.
- Zero-malloc discipline strengthened: SOC path and line-search filter avoid heap allocation, with tests and ASan-friendly hooks.
- Numerical robustness fixes landed (e.g., `fast_inverse()` guarded by SPD checks) and a microbenchmark added to quantify overhead.

## Near-Term (Next 2-6 Weeks)

- Establish an external `nmpc-bench` repository:
- Compare MiniSolver vs acados vs CasADi using **C/C++** interfaces (no Python runner for fairness).
- Use a small set of canonical NMPC scenarios and record both accuracy (reference solution) and runtime.
- Define and enforce correctness gates: MiniSolver must match a high-precision reference before optimizing iteration/time.

## Medium-Term (Next 2-3 Months)

- Matrix backend hardening:
- Keep `MiniMatrix` as an **NMPC-specialized** backend; do not grow it into a general linear algebra library.
- Every new kernel must have:
- A `vs Eigen` correctness test.
- A microbenchmark if it is on the hot path.
- Embedded readiness:
- Reduce compile-time/binary-size overhead where possible (especially when Eigen is disabled).
- Harden determinism and real-time constraints (no hidden allocations, stable iteration behavior under neighboring problems).

## Design Guardrails

- Occam's razor: no new public APIs without a concrete use-case, tests, and a performance/correctness justification.
- "Benchmark-driven": performance claims must be backed by reproducible benchmark artifacts, not anecdotes.
- "Correctness-first": always establish a reference solution before tuning heuristics/regularization/line-search behavior.
