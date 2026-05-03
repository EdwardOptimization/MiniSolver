#pragma once
#include "minisolver/core/solver_options.h"
#include "minisolver/matrix/matrix_defs.h"

namespace minisolver {

enum class SolverStatus {
    // Initial / intermediate status.
    UNSOLVED = 0,

    // Successful statuses.
    OPTIMAL = 1, // Fully converged: KKT conditions and constraint violation meet tolerances.
    FEASIBLE = 2, // Acceptable but suboptimal: violation is within the configured bound.

    // Failure statuses.
    INFEASIBLE = 3, // Final iterate is primal-infeasible after a success-like loop verdict.
    MAX_ITER = 4, // Iteration budget exhausted before reaching an acceptable solution.
    STEP_TOO_SMALL = 5, // Globalization collapsed to a tiny step with no recovery path.
    INSUFFICIENT_PROGRESS = 6, // Cost/progress stagnated before convergence.
    LINEAR_SOLVE_FAILED = 7, // Riccati/KKT direction solve failed after retries.
    RESTORATION_FAILED = 8, // Feasibility restoration was attempted but did not recover.
    INVALID_INPUT = 9, // Reserved for config/model input validation failures.
    NUMERICAL_ERROR = 10 // Invalid arithmetic, NaN/Inf, or invalid search direction.
};

enum class TerminationReason {
    NONE = 0,
    CONVERGED = 1,
    PRIMAL_FEASIBLE = 2,
    MAX_ITERATIONS = 3,
    FIXED_ITERATION = 4,
    COST_STAGNATION = 5,
    LINE_SEARCH_FAILED = 6,
    LINEAR_SOLVE_FAILED = 7,
    RESTORATION_FAILED = 8,
    INVALID_INPUT = 9,
    NUMERICAL_ERROR = 10,
    POSTSOLVE_INFEASIBLE = 11
};

enum class ApiStatus {
    OK = 0,
    InvalidHorizon = 1,
    InvalidStage = 2,
    InvalidIndex = 3,
    UnknownName = 4,
    SizeMismatch = 5,
    NonFiniteValue = 6,
    TerminalControl = 7,
    InvalidArgument = 8
};

inline const char* status_to_string(SolverStatus status)
{
    switch (status) {
    case SolverStatus::UNSOLVED:
        return "UNSOLVED";
    case SolverStatus::OPTIMAL:
        return "OPTIMAL";
    case SolverStatus::FEASIBLE:
        return "FEASIBLE";
    case SolverStatus::INFEASIBLE:
        return "INFEASIBLE";
    case SolverStatus::MAX_ITER:
        return "MAX_ITER";
    case SolverStatus::STEP_TOO_SMALL:
        return "STEP_TOO_SMALL";
    case SolverStatus::INSUFFICIENT_PROGRESS:
        return "INSUFFICIENT_PROGRESS";
    case SolverStatus::LINEAR_SOLVE_FAILED:
        return "LINEAR_SOLVE_FAILED";
    case SolverStatus::RESTORATION_FAILED:
        return "RESTORATION_FAILED";
    case SolverStatus::INVALID_INPUT:
        return "INVALID_INPUT";
    case SolverStatus::NUMERICAL_ERROR:
        return "NUMERICAL_ERROR";
    default:
        return "UNKNOWN";
    }
}

inline const char* api_status_to_string(ApiStatus status)
{
    switch (status) {
    case ApiStatus::OK:
        return "OK";
    case ApiStatus::InvalidHorizon:
        return "InvalidHorizon";
    case ApiStatus::InvalidStage:
        return "InvalidStage";
    case ApiStatus::InvalidIndex:
        return "InvalidIndex";
    case ApiStatus::UnknownName:
        return "UnknownName";
    case ApiStatus::SizeMismatch:
        return "SizeMismatch";
    case ApiStatus::NonFiniteValue:
        return "NonFiniteValue";
    case ApiStatus::TerminalControl:
        return "TerminalControl";
    case ApiStatus::InvalidArgument:
        return "InvalidArgument";
    default:
        return "UNKNOWN";
    }
}

inline bool api_status_ok(ApiStatus status)
{
    return status == ApiStatus::OK;
}

inline const char* termination_reason_to_string(TerminationReason reason)
{
    switch (reason) {
    case TerminationReason::NONE:
        return "NONE";
    case TerminationReason::CONVERGED:
        return "CONVERGED";
    case TerminationReason::PRIMAL_FEASIBLE:
        return "PRIMAL_FEASIBLE";
    case TerminationReason::MAX_ITERATIONS:
        return "MAX_ITERATIONS";
    case TerminationReason::FIXED_ITERATION:
        return "FIXED_ITERATION";
    case TerminationReason::COST_STAGNATION:
        return "COST_STAGNATION";
    case TerminationReason::LINE_SEARCH_FAILED:
        return "LINE_SEARCH_FAILED";
    case TerminationReason::LINEAR_SOLVE_FAILED:
        return "LINEAR_SOLVE_FAILED";
    case TerminationReason::RESTORATION_FAILED:
        return "RESTORATION_FAILED";
    case TerminationReason::INVALID_INPUT:
        return "INVALID_INPUT";
    case TerminationReason::NUMERICAL_ERROR:
        return "NUMERICAL_ERROR";
    case TerminationReason::POSTSOLVE_INFEASIBLE:
        return "POSTSOLVE_INFEASIBLE";
    default:
        return "UNKNOWN";
    }
}

struct SolverInfo {
    SolverStatus status = SolverStatus::UNSOLVED;
    SolverStatus loop_status = SolverStatus::UNSOLVED;
    TerminationReason termination_reason = TerminationReason::NONE;

    int iterations = 0;
    // Internal primal feasibility metric used by convergence and FEASIBLE classification.
    // With constraint/problem scaling enabled, inspect unscaled_primal_inf for model-unit
    // residuals.
    double primal_inf = 0.0;
    // Raw model-unit feasibility diagnostic; not used by default to classify SolverStatus.
    double unscaled_primal_inf = 0.0;
    double dual_inf = 0.0;
    double complementarity_inf = 0.0;
    double barrier_centrality_inf = 0.0;
    double mu = 0.0;
    double alpha = 1.0;

    bool linear_ok = true;
    bool line_search_failed = false;
    bool restoration_used = false;
    bool degraded_step = false;
    int degraded_riccati_freeze_count = 0;
    int regularization_escalation_count = 0;
    int soc_attempt_count = 0;
    int soc_accept_count = 0;
    int soc_reject_count = 0;
    int restoration_attempt_count = 0;
    int restoration_success_count = 0;
    bool constraint_scaling_active = false;
    bool objective_scaling_active = false;
    bool problem_scaling_active = false;

    void reset()
    {
        status = SolverStatus::UNSOLVED;
        loop_status = SolverStatus::UNSOLVED;
        termination_reason = TerminationReason::NONE;
        iterations = 0;
        primal_inf = 0.0;
        unscaled_primal_inf = 0.0;
        dual_inf = 0.0;
        complementarity_inf = 0.0;
        barrier_centrality_inf = 0.0;
        mu = 0.0;
        alpha = 1.0;
        linear_ok = true;
        line_search_failed = false;
        restoration_used = false;
        degraded_step = false;
        degraded_riccati_freeze_count = 0;
        regularization_escalation_count = 0;
        soc_attempt_count = 0;
        soc_accept_count = 0;
        soc_reject_count = 0;
        restoration_attempt_count = 0;
        restoration_success_count = 0;
        constraint_scaling_active = false;
        objective_scaling_active = false;
        problem_scaling_active = false;
    }
};

struct LineSearchResult {
    double alpha = 0.0;
    bool soc_attempted = false;
    bool soc_accepted = false;
    bool soc_rejected = false;

    constexpr LineSearchResult() = default;
    constexpr explicit LineSearchResult(double alpha_value)
        : alpha(alpha_value)
    {
    }
};

// =============================================================================
// Sub-structure 1: KnotState
//
// Contains all vectors and scalars that participate in double-buffering.
// These are the fields copied during prepare_candidate() and needed after
// trajectory swap (primal/dual state, evaluation results, search directions,
// cost/barrier gradients, feedback feedforward term).
//
// Skipped from this group are all large matrices, which are recomputed each
// iteration by Model::compute_* and the Riccati solver.
// =============================================================================
template <typename T, int _NX, int _NU, int _NC, int _NP> struct KnotState {
#ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#endif

    // --- Primal Variables ---
    MSVec<T, _NX> x;
    MSVec<T, _NU> u;
    MSVec<T, _NP> p; // Parameters

    // --- Dual Variables & Slacks ---
    MSVec<T, _NC> s; // Slack variables
    MSVec<T, _NC> lam; // Dual variables (Lambda)
    MSVec<T, _NC> soft_s; // Soft constraint slack (L1)
    // Note: The L1 soft constraint dual is (w - lam), computed implicitly.
    // No separate soft_dual variable is needed.

    // --- Evaluation Results ---
    T cost; // Scalar cost value
    T cost_unscaled; // Raw objective value before optional objective/problem scaling
    T objective_scale; // Active objective scale used by cost/q/r/Q/R/H
    MSVec<T, _NC> g_val; // QP/IPM constraint residual packet used with C/D
    MSVec<T, _NC> g_true; // True nonlinear constraint residual used by internal metrics
    MSVec<T, _NC> g_unscaled; // Raw true residual before optional row scaling
    MSVec<T, _NC> constraint_row_scale; // Active scale used by g_val/g_true/C/D
    MSVec<T, _NX> f_resid; // Predicted next state f(x, u)

    // --- Cost Gradients ---
    MSVec<T, _NX> q;
    MSVec<T, _NU> r;

    // --- Barrier Gradients (used in iteration logging) ---
    MSVec<T, _NX> q_bar;
    MSVec<T, _NU> r_bar;

    // --- Search Directions ---
    MSVec<T, _NX> dx;
    MSVec<T, _NU> du;
    MSVec<T, _NC> ds;
    MSVec<T, _NC> dlam;
    MSVec<T, _NC> dsoft_s;

    // --- Feedback Feedforward Term ---
    MSVec<T, _NU> d;
};

// =============================================================================
// Sub-structure 2: KnotMatrices
//
// Contains all large matrices that do NOT participate in double-buffering.
// These are recomputed each iteration:
//   - A, B, C, D, Q, R, H: by Model::compute_dynamics/constraints/cost
//   - Q_bar, R_bar, H_bar: by compute_barrier_derivatives (Riccati)
//   - K: by Riccati backward pass
// =============================================================================
template <typename T, int _NX, int _NU, int _NC> struct KnotMatrices {
#ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#endif

    // --- Dynamics Jacobians ---
    MSMat<T, _NX, _NX> A; // State transition: dx_{k+1}/dx_k
    MSMat<T, _NX, _NU> B; // Control influence: dx_{k+1}/du_k

    // --- Constraint Jacobians ---
    MSMat<T, _NC, _NX> C; // dg/dx
    MSMat<T, _NC, _NU> D; // dg/du

    // --- Cost Hessians ---
    MSMat<T, _NX, _NX> Q; // d²L/dx²
    MSMat<T, _NU, _NU> R; // d²L/du²
    MSMat<T, _NU, _NX> H; // d²L/dudx (cross term)

    // --- Barrier-Modified Hessians (Riccati Input) ---
    MSMat<T, _NX, _NX> Q_bar;
    MSMat<T, _NU, _NU> R_bar;
    MSMat<T, _NU, _NX> H_bar;

    // --- Feedback Gain Matrix ---
    MSMat<T, _NU, _NX> K;
};

// =============================================================================
// KnotPoint: Composite type via multiple inheritance
//
// Inherits from KnotState (vectors/scalars) and KnotMatrices (large matrices).
// All existing access patterns (kp.x, kp.A, kp.dx, etc.) work unchanged
// through inheritance — zero external API change.
//
// This enables:
//   - Selective copy via base class assignment (copy_state_from)
//   - Type-level distinction between buffered and non-buffered data
//   - Foundation for future storage separation in Trajectory
// =============================================================================
template <typename T, int _NX, int _NU, int _NC, int _NP>
struct KnotPoint : KnotState<T, _NX, _NU, _NC, _NP>, KnotMatrices<T, _NX, _NU, _NC> {
#ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#endif

    // === EXPOSE TEMPLATE ARGUMENTS AS CONSTANTS ===
    static const int NX = _NX;
    static const int NU = _NU;
    static const int NC = _NC;
    static const int NP = _NP;

    // === TYPE ALIASES for sub-structures ===
    using StateType = KnotState<T, _NX, _NU, _NC, _NP>;
    using MatricesType = KnotMatrices<T, _NX, _NU, _NC>;

    KnotPoint()
    {
        set_zero();
        initialize_defaults();
    }

    void set_zero()
    {
        // --- KnotState fields (vectors & scalars) ---
        MatOps::setZero(this->x);
        MatOps::setZero(this->u);
        MatOps::setZero(this->p);
        this->s.setOnes();
        this->lam.setOnes();
        this->soft_s.setOnes();

        this->cost = 0;
        this->cost_unscaled = 0;
        this->objective_scale = 1;
        MatOps::setZero(this->g_val);
        MatOps::setZero(this->g_true);
        MatOps::setZero(this->g_unscaled);
        this->constraint_row_scale.setOnes();
        MatOps::setZero(this->f_resid);
        MatOps::setZero(this->q);
        MatOps::setZero(this->r);
        MatOps::setZero(this->q_bar);
        MatOps::setZero(this->r_bar);
        MatOps::setZero(this->dx);
        MatOps::setZero(this->du);
        MatOps::setZero(this->ds);
        MatOps::setZero(this->dlam);
        MatOps::setZero(this->dsoft_s);
        MatOps::setZero(this->d);

        // --- KnotMatrices fields (large matrices) ---
        MatOps::setIdentity(this->A);
        MatOps::setZero(this->B);
        MatOps::setZero(this->C);
        MatOps::setZero(this->D);
        MatOps::setIdentity(this->Q);
        MatOps::setIdentity(this->R);
        MatOps::setZero(this->H);
        MatOps::setZero(this->Q_bar);
        MatOps::setZero(this->R_bar);
        MatOps::setZero(this->H_bar);
        MatOps::setZero(this->K);
    }

    void initialize_defaults()
    {
        this->s.fill(1.0);
        this->lam.fill(1.0);
        this->soft_s.fill(1.0);
        this->constraint_row_scale.fill(1.0);
        this->objective_scale = 1.0;
    }

    // =========================================================================
    // Lightweight copy for double-buffering (Line Search candidate preparation).
    // Copies only the KnotState base (vectors/scalars), skipping KnotMatrices
    // (all large matrices) which are recomputed each iteration.
    // =========================================================================
    void copy_state_from(const KnotPoint& other)
    {
        static_cast<StateType&>(*this) = static_cast<const StateType&>(other);
    }
};
}
