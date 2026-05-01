#pragma once

namespace minisolver {

// SolverContext will grow into the boundary between the canonical solve loop
// and strategy kernels. Start with scalar metrics only; keep runtime state
// (mu/reg/iteration) in MiniSolver until serializer and tests move over.
struct SolverMetrics {
    double last_prim_inf = 0.0;
    double last_dual_inf = 0.0;
    double last_alpha = 1.0;

    // Mehrotra predictor diagnostics, used by regression tests and debugging.
    double last_mu_aff = 0.0;
    double last_alpha_aff = 0.0;

    void reset_algorithmic()
    {
        last_prim_inf = 0.0;
        last_dual_inf = 0.0;
        last_alpha = 1.0;
        last_mu_aff = 0.0;
        last_alpha_aff = 0.0;
    }

    void reset_solve() { last_alpha = 1.0; }
};

struct StepResidualSummary {
    double max_kkt_error = 0.0;
    double max_prim_inf = 0.0;
    double avg_gap = 0.0;
};

} // namespace minisolver
