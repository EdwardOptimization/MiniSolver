#pragma once

namespace minisolver {

enum class IntegratorType {
    EULER_EXPLICIT,
    EULER_IMPLICIT,
    RK2_EXPLICIT,
    RK2_IMPLICIT,
    RK4_EXPLICIT,
    RK4_IMPLICIT,
    DISCRETE
};

enum class BarrierStrategy { MONOTONE, ADAPTIVE, MEHROTRA };

// Newton solver parameters for implicit integrators.
struct NewtonConfig {
    int max_iters = 20;
    double tol = 1e-10;
    double regularization = 1e-12; // Fallback diagonal damping for singular Jacobians
};

enum class InertiaStrategy {
    REGULARIZATION,
    SATURATION,
    IGNORE_SINGULAR
    // In the future: FACTORIZATION_MODIFY
};

// Line search strategy for globalization. For real-time NMPC (SQP-style) it's common to
// disable backtracking and simply take a fraction-to-boundary step.
enum class LineSearchType { MERIT, FILTER, NONE };

enum class ConstraintScalingMethod {
    NONE,

    // Scales each constraint row by the inverse inf-norm of its residual/Jacobian packet.
    ROW_INF_NORM
};

enum class ObjectiveScalingMethod {
    NONE,

    // Scale cost/gradient/Hessian by the inverse Gershgorin upper bound of the
    // local objective Hessian packet. User-facing costs remain available unscaled.
    HESSIAN_GERSHGORIN
};

enum class ProblemScalingMethod {
    NONE,

    // Bounded first problem-level profile: constraint row equilibration plus
    // objective Hessian Gershgorin scaling. It intentionally does not transform
    // user variables or Riccati dynamics coordinates yet.
    RUIZ_EQUILIBRATION
};

// Coordinate-level scaling hint. The MiniSolver instance owns per-state,
// per-control, and per-parameter scale factors; this enum selects whether the
// solver consumes them as a normalization hint for stationarity termination
// metrics. The hint never rescales primal variables, dynamics Jacobians, the
// Riccati recursion, or warm-start deltas; full coordinate equilibration in
// that algebra-level sense is tracked separately in the scaling design doc
// (Stage 5).
//
// Convention: scale_i represents the typical magnitude of coordinate i in
// model units. With USER_SUPPLIED active, the dual-stationarity infinity norm
// is computed as `max_i |r_i| * control_scale_i` so coordinates with large
// natural magnitude do not dominate the convergence test for well-trimmed
// coordinates.
enum class CoordinateScalingMethod { NONE, USER_SUPPLIED };

enum class HessianApproximation {
    EXACT, // Exact objective Hessian + constraint Hessian; dynamics Hessian is not included.
    // Objective-only curvature: exact Hessian for general minimize() terms plus
    // true Gauss-Newton J^T W J for add_residual() least-squares terms. Constraint
    // and dynamics Hessians are ignored.
    OBJECTIVE_HESSIAN_ONLY,
    GAUSS_NEWTON = OBJECTIVE_HESSIAN_ONLY // Backward-compatible legacy name.
};

// Print Levels
enum class PrintLevel {
    NONE, // Silent
    WARN, // Warnings and Errors only
    INFO, // Start/End summary only
    ITER, // One line per iteration
    DEBUG // Detailed internal state
};

enum class Backend { CPU_SERIAL, GPU_MPX, GPU_PCR };

// Reset Options
enum class ResetOption {
    // Resets only algorithmic state (mu, reg, iter, filter, timers).
    // Keeps the current trajectory data (x, u, p, s, lam).
    // Note: whether the next solve reuses stored slack/dual values still follows
    // SolverConfig::initialization.
    ALG_STATE,

    // Resets algorithmic state AND wipes trajectory data to defaults.
    // Equivalent to destroying and creating a new MiniSolver.
    // Use this for benchmarks or switching to a completely unrelated problem.
    FULL
};

enum class InitializationMode {
    // Reinitialize barrier/slack/dual state for a fresh solve on the current problem data.
    COLD_START,

    // Reuse the current primal guess (x/u) but rebuild slack/dual/barrier state.
    // This is the typical mode for neighboring problems in MPC.
    REUSE_PRIMAL,

    // Reuse the current primal-dual iterate (x/u/s/lam). If the stored slack/dual state
    // is invalid, the solver will fall back to REUSE_PRIMAL.
    REUSE_PRIMAL_DUAL
};

enum class WarmStartBarrierMode {
    // Conservative default: every solve starts from config.mu_init.
    RESET_TO_MU_INIT,

    // Reuse the previous solve's barrier parameter if the primal-dual iterate is valid.
    REUSE_PREVIOUS_MU,

    // Set mu from the stored complementarity gap s*lam (and L1 soft_s*(w-lam)).
    FROM_COMPLEMENTARITY_GAP
};

enum class WarmStartRegularizationMode {
    // Conservative default: every solve starts from config.reg_init.
    RESET_TO_REG_INIT,

    // Reuse the previous solve's regularization value, clamped to [reg_min, reg_max].
    REUSE_PREVIOUS_REG,

    // Carry regularization but decay it once, matching successful-solve reg cooldown.
    DECAY_PREVIOUS_REG
};

enum class DirectionRefinementMode {
    // No post-Riccati direction correction.
    NONE,

    // Correct only the linearized dynamics defect by rolling dx/du forward through the
    // existing Riccati feedback gains. This is not full KKT iterative refinement and does
    // not rebuild slack/dual directions.
    DYNAMICS_DEFECT_ROLLOUT,

    // Multi-pass dynamics-defect rollout that iterates the DYNAMICS_DEFECT_ROLLOUT
    // correction until the maximum dynamic defect drops below
    // direction_refinement_tol or direction_refinement_max_passes is reached. The
    // mode auto-degrades to a single pass when any inequality dual is non-trivial
    // so the OD-005 dual-consistency hazard is not amplified by repeated
    // primal-only refinements; see the overdesign ledger for the rationale.
    // Despite the name, this remains a structured Riccati-feedback refinement,
    // not full unstructured KKT iterative refinement.
    FULL_KKT_ITERATIVE_REFINEMENT
};

enum class TerminationProfile {
    // Strict solve quality: OPTIMAL requires primal, dual, and true complementarity residuals.
    // FEASIBLE remains a postsolve fallback for primal-acceptable NMPC iterates.
    STRICT_KKT,

    // Same strict OPTIMAL check, but documents that FEASIBLE fallback is an expected
    // control-oriented outcome when primal feasibility is acceptable.
    ACCEPTABLE_NMPC,

    // Exit after one SQP/IPM iteration and let postsolve classify the resulting iterate.
    // This is the config-driven RTI entry point.
    RTI_FIXED_ITERATION
};

struct SolverConfig {
    Backend backend = Backend::CPU_SERIAL;
    InitializationMode initialization = InitializationMode::COLD_START;
    WarmStartBarrierMode warm_start_barrier = WarmStartBarrierMode::RESET_TO_MU_INIT;
    WarmStartRegularizationMode warm_start_regularization
        = WarmStartRegularizationMode::RESET_TO_REG_INIT;
    TerminationProfile termination_profile = TerminationProfile::STRICT_KKT;
    // Scaling changes internal residual/cost packets only. User-facing states,
    // controls, parameters, and reported stage costs remain in model units.
    ConstraintScalingMethod constraint_scaling = ConstraintScalingMethod::NONE;
    ObjectiveScalingMethod objective_scaling = ObjectiveScalingMethod::NONE;
    ProblemScalingMethod problem_scaling = ProblemScalingMethod::NONE;
    // Coordinate-level scaling hint. The per-coordinate scale factors live on
    // the MiniSolver instance because they are template-sized. The default
    // NONE keeps current behaviour byte-for-byte; USER_SUPPLIED activates the
    // weighted dual-stationarity termination metric.
    CoordinateScalingMethod coordinate_scaling = CoordinateScalingMethod::NONE;
    // Bounds applied at API entry to user-supplied coordinate scales. Both
    // ends must be strictly positive and finite so the weighted norm cannot
    // hide a non-stationary coordinate behind a tiny (or huge) scale factor.
    double coordinate_scale_min = 1e-6;
    double coordinate_scale_max = 1e6;
    // Bounds for automatic row/objective scales; keep them finite and positive
    // so scaled residuals stay numerically useful without hiding raw magnitudes.
    double constraint_row_scale_min = 1e-4;
    double constraint_row_scale_max = 1e4;
    double objective_scale_min = 1e-4;
    double objective_scale_max = 1.0;

    // --- Integration ---
    // RK4 is a good balance for general nonlinear problems
    IntegratorType integrator = IntegratorType::RK4_EXPLICIT;
    double default_dt = 0.1;
    NewtonConfig newton_config; // Implicit integrator Newton solver parameters

    // --- Barrier Strategy ---
    // ADAPTIVE is generally the most robust and fast for general nonlinear problems
    BarrierStrategy barrier_strategy = BarrierStrategy::ADAPTIVE;

    double mu_init = 1e-1;
    double mu_final = 1e-6; // Tighter tolerance for high precision
    double mu_linear_decrease_factor = 0.2;
    double barrier_tolerance_factor = 10.0;
    double mu_safety_margin = 0.1;

    // --- Regularization ---
    InertiaStrategy inertia_strategy = InertiaStrategy::REGULARIZATION;
    double reg_init = 1e-4; // Slightly higher init to be safe
    double reg_min = 1e-8;
    double reg_max = 1e9;
    double reg_scale_up = 100.0; // Aggressive scaling to recover quickly
    double reg_scale_down = 2.0;
    double regularization_step = 1e-6; // Step size for regularization

    // Linear solve retry tuning. This is a total attempt count: 1 means the
    // current regularization is tried once with no retry; later attempts may
    // increase regularization after failed factorizations.
    double singular_threshold = 1e-4; // For IGNORE_SINGULAR
    double huge_penalty = 1e9; // Penalty for frozen directions
    int linear_solve_max_attempts = 5;

    // --- Convergence Tolerances ---
    // Primal feasibility tolerance in internal solver units. With constraint or
    // problem scaling enabled, compare SolverInfo::unscaled_primal_inf for raw
    // model-unit feasibility diagnostics.
    double tol_con = 1e-4;
    // Stationarity / dual infeasibility tolerance for the internal Lagrangian
    // residual. Objective scaling can change this residual magnitude by design.
    double tol_dual = 1e-4;
    // Complementarity tolerance in the active internal slack/dual units.
    double tol_mu = 1e-5;
    // Objective Stagnation Tolerance
    // Stops the solver if the cost improvement between iterations is smaller than this value,
    // provided the solution is feasible.
    double tol_cost = 1e-6;
    // FEASIBLE fallback bound is tol_con * feasible_tol_scale in the same
    // internal units as tol_con; raw model feasibility is reported separately.
    double feasible_tol_scale = 10.0;

    // --- Line Search & Robustness ---
    // Filter is generally more robust than Merit without parameter tuning
    LineSearchType line_search_type = LineSearchType::FILTER;
    int line_search_max_iters = 10;
    double line_search_tau = 0.995; // Fraction to boundary
    double line_search_backtrack_factor = 0.5;

    // Filter Method Parameters
    double filter_gamma_theta = 1e-5;
    double filter_gamma_phi = 1e-5;
    double filter_theta_max_factor = 1e4;

    // Merit Line Search: Armijo sufficient decrease constant.
    // Step accepted only if phi(alpha) <= phi(0) + c1 * alpha * dphi.
    // Standard value: 1e-4. Set to 0.0 to revert to simple decrease.
    double armijo_c1 = 1e-4;

    // Barrier Numerical Safety
    double min_barrier_slack = 1e-12;
    double barrier_inf_cost = 1e9;

    // Watchdog / Heuristics
    double slack_reset_trigger = 1e-3; // Only reset if step is VERY small
    double warm_start_slack_init = 1e-6;

    // Globalization. eta_suff_descent is used by f-type filter steps as the
    // Armijo coefficient for objective-barrier decrease.
    double soc_trigger_alpha = 0.5;
    double merit_nu_init = 1000.0;
    double eta_suff_descent = 1e-4;

    // Restoration
    int max_restoration_iters = 5;
    double restoration_mu = 1e-1;
    double restoration_reg = 1e-2;
    double restoration_alpha = 0.8;
    // Restoration succeeds after reaching the feasible bound or reducing
    // infeasibility to this fraction of the pre-restoration value.
    double restoration_sufficient_decrease_factor = 0.9;

    // --- General ---
    int max_iters = 100; // Give it enough room
    PrintLevel print_level = PrintLevel::NONE;
    bool enable_profiling
        = false; // Profiling uses dynamic containers; keep solve() zero-malloc by default.

    // --- Advanced Features ---
    HessianApproximation hessian_approximation
        = HessianApproximation::OBJECTIVE_HESSIAN_ONLY; // Default: fast, ignore constraint
                                                        // curvature

    DirectionRefinementMode direction_refinement = DirectionRefinementMode::NONE;
    // Maximum refinement passes used by FULL_KKT_ITERATIVE_REFINEMENT. The
    // single-pass DYNAMICS_DEFECT_ROLLOUT mode ignores this field. A value of
    // 1 reduces the iterative mode to single-pass behaviour. Each extra pass
    // is bounded work (no allocations, no extra factorizations) so the
    // default of 4 keeps the worst-case overhead small.
    int direction_refinement_max_passes = 4;
    // Convergence tolerance on the dynamic-defect inf-norm used by
    // FULL_KKT_ITERATIVE_REFINEMENT to stop early. Must be finite and
    // strictly positive when the iterative mode is selected.
    double direction_refinement_tol = 1e-12;

    // Line Search Logic
    //
    // Canonical MiniSolver globalization is multiple-shooting: evaluate trial
    // points on z + alpha * dz and keep dynamics defects in the merit/filter
    // residual. Keep rollout disabled for a theory-clean SQP/IPM path.
    //
    // When enabled, rollout is only a dynamics-projection heuristic: x0 is
    // fixed, u/s/lam/soft_s move by alpha*d, and states are re-integrated. It
    // is not a standard multiple-shooting line search and not an iLQR/DDP
    // rollout of the form u + alpha*k + K*(x_rollout - x_nominal).
    bool enable_line_search_rollout = false;

    // Riccati Logic
    bool enable_defect_correction = true;

    // Mehrotra Logic
    bool enable_corrector = true;
    bool enable_aggressive_barrier = true; // Allow aggressive mu reduction based on step size

    // Feasibility and globalization recovery heuristics enabled by default for
    // robust NMPC behavior. The reference config disables these explicitly when
    // a minimal primal-dual IPM path is needed for debugging or regression.
    bool enable_slack_reset = true;
    bool enable_feasibility_restoration = true;
    bool enable_soc = true;

    // --- RTI-lite ---
    // RTI-lite is a safe-by-default warm-start mode for repeated MPC solves.
    // When `enable_rti_lite=true` and the previous solve reached an acceptable
    // verdict (OPTIMAL/FEASIBLE), the next `solve()` call reuses the previous
    // primal-dual iterate and runs at most `rti_lite_max_linearization_age`
    // SQP/IPM iterations under `TerminationProfile::ACCEPTABLE_NMPC`. The
    // reuse path is gated by:
    //   * the L2 distance between the new initial state and the previous
    //     solved initial state must be < `rti_lite_max_state_delta`;
    //   * the previous solve must have been acceptable;
    //   * the linearization age must not have exceeded
    //     `rti_lite_max_linearization_age`.
    // If any gate fails, the solver falls back to a full SQP/IPM solve under
    // the user-provided `max_iters` / `termination_profile` and resets the
    // linearization age. Diagnostic counters live on `SolverInfo`
    // (`rti_lite_reused_linearization`, `rti_lite_linearization_age`).
    bool enable_rti_lite = false;
    int rti_lite_max_linearization_age = 3;
    double rti_lite_max_state_delta = 0.5;
};

}
