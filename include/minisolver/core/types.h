#pragma once
#include "minisolver/core/solver_options.h"
#include "minisolver/core/matrix_defs.h" 

namespace minisolver {

enum class SolverStatus {
    // [初始/中间状态]
    UNSOLVED,           
    
    // [成功状态]
    OPTIMAL,             // 完美收敛 (Optimal): 满足所有 KKT 条件且违反度 <= tol
    FEASIBLE,           // 工程可用 (Suboptimal): 迭代结束或停滞，但违反度在允许范围内
    
    // [失败状态]
    INFEASIBLE,         // 不可用 (Failed): 迭代结束，违反度过大
    NUMERICAL_ERROR     // 数值错误: 矩阵奇异或 NaN，无法继续
};

inline const char* status_to_string(SolverStatus status) {
    switch(status) {
        case SolverStatus::UNSOLVED: return "UNSOLVED";
        case SolverStatus::OPTIMAL: return "SOLVED";
        case SolverStatus::FEASIBLE: return "FEASIBLE";
        case SolverStatus::INFEASIBLE: return "INFEASIBLE";
        case SolverStatus::NUMERICAL_ERROR: return "NUMERICAL_ERROR";
        default: return "UNKNOWN";
    }
}

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
template<typename T, int _NX, int _NU, int _NC, int _NP>
struct KnotState {
    #ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    #endif

    // --- Primal Variables ---
    MSVec<T, _NX> x;
    MSVec<T, _NU> u;
    MSVec<T, _NP> p; // Parameters

    // --- Dual Variables & Slacks ---
    MSVec<T, _NC> s;         // Slack variables
    MSVec<T, _NC> lam;       // Dual variables (Lambda)
    MSVec<T, _NC> soft_s;    // Soft constraint slack (L1)
    // Note: The L1 soft constraint dual is (w - lam), computed implicitly.
    // No separate soft_dual variable is needed.

    // --- Evaluation Results ---
    T cost;                   // Scalar cost value
    MSVec<T, _NC> g_val;     // Constraint residuals g(x, u)
    MSVec<T, _NX> f_resid;   // Predicted next state f(x, u)

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
template<typename T, int _NX, int _NU, int _NC>
struct KnotMatrices {
    #ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    #endif

    // --- Dynamics Jacobians ---
    MSMat<T, _NX, _NX> A;    // State transition: dx_{k+1}/dx_k
    MSMat<T, _NX, _NU> B;    // Control influence: dx_{k+1}/du_k

    // --- Constraint Jacobians ---
    MSMat<T, _NC, _NX> C;    // dg/dx
    MSMat<T, _NC, _NU> D;    // dg/du

    // --- Cost Hessians ---
    MSMat<T, _NX, _NX> Q;    // d²L/dx²
    MSMat<T, _NU, _NU> R;    // d²L/du²
    MSMat<T, _NU, _NX> H;    // d²L/dudx (cross term)

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
template<typename T, int _NX, int _NU, int _NC, int _NP>
struct KnotPoint : KnotState<T, _NX, _NU, _NC, _NP>,
                   KnotMatrices<T, _NX, _NU, _NC> {
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

    KnotPoint() {
        set_zero();
        initialize_defaults();
    }

    void set_zero() {
        // --- KnotState fields (vectors & scalars) ---
        MatOps::setZero(this->x); MatOps::setZero(this->u); MatOps::setZero(this->p);
        this->s.setOnes(); this->lam.setOnes(); 
        this->soft_s.setOnes();

        this->cost = 0;
        MatOps::setZero(this->g_val);
        MatOps::setZero(this->f_resid);
        MatOps::setZero(this->q); MatOps::setZero(this->r);
        MatOps::setZero(this->q_bar); MatOps::setZero(this->r_bar);
        MatOps::setZero(this->dx); MatOps::setZero(this->du);
        MatOps::setZero(this->ds); MatOps::setZero(this->dlam); 
        MatOps::setZero(this->dsoft_s);
        MatOps::setZero(this->d);

        // --- KnotMatrices fields (large matrices) ---
        MatOps::setIdentity(this->A); MatOps::setZero(this->B);
        MatOps::setZero(this->C); MatOps::setZero(this->D);
        MatOps::setIdentity(this->Q); MatOps::setIdentity(this->R); MatOps::setZero(this->H);
        MatOps::setZero(this->Q_bar); MatOps::setZero(this->R_bar); MatOps::setZero(this->H_bar);
        MatOps::setZero(this->K);
    }

    void initialize_defaults() {
        this->s.fill(1.0);
        this->lam.fill(1.0);
        this->soft_s.fill(1.0);
    }

    // =========================================================================
    // Lightweight copy for double-buffering (Line Search candidate preparation).
    // Copies only the KnotState base (vectors/scalars), skipping KnotMatrices
    // (all large matrices) which are recomputed each iteration.
    // =========================================================================
    void copy_state_from(const KnotPoint& other) {
        static_cast<StateType&>(*this) = static_cast<const StateType&>(other);
    }
};
}
