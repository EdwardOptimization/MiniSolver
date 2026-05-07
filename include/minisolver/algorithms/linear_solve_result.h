#pragma once

namespace minisolver {

struct LinearSolveResult {
    bool ok = false;
    bool degraded_step = false;
    int degraded_riccati_freeze_count = 0;

    // Riccati robustness diagnostics for a single linear-solve invocation
    // (one full backward+forward pass). Always populated, regardless of
    // RiccatiRobustMode; they surface what the existing fallback paths in
    // cpu_serial_solve already do silently, so callers can detect
    // numerically suspicious solves without forcing a square-root or LDLT
    // rewrite. The Solver-level fields `SolverInfo::riccati_indefinite_blocks`
    // and `SolverInfo::riccati_max_diagonal_perturbation` aggregate these
    // per-solve values across an entire SQP run.
    //
    // riccati_indefinite_blocks: number of backward-pass stages in *this*
    // linear solve where Quu required escalation beyond `reg` to factor
    // (general-path SPD retry, the small-Nu freeze fallback, or
    // SATURATION/IGNORE_SINGULAR repair sweeps). Each stage that fires a
    // fallback bumps the counter by one, so the maximum value is N+1.
    //
    // riccati_max_diagonal_perturbation: largest *extra* diagonal value
    // added to Quu(i,i) in *this* linear solve, on top of the standard
    // `reg` shift. Bounds the worst inertia-correction the legacy path
    // applied during this single Riccati pass.
    int riccati_indefinite_blocks = 0;
    double riccati_max_diagonal_perturbation = 0.0;

    constexpr LinearSolveResult() = default;

    constexpr LinearSolveResult(
        bool ok_value, bool degraded_value = false, int degraded_freeze_count = 0)
        : ok(ok_value)
        , degraded_step(degraded_value)
        , degraded_riccati_freeze_count(degraded_freeze_count)
    {
    }

    constexpr operator bool() const { return ok; }
};

} // namespace minisolver
