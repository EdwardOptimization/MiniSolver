#pragma once
#include "minisolver/core/solver_options.h"
#include "minisolver/core/matrix_defs.h" 

namespace minisolver {

enum class SolverStatus {
    SOLVED,                // Converged to tolerances
    MAX_ITER,              // Maximum iterations reached without convergence
    FEASIBLE,              // Feasible but not optimal (e.g. max iters reached but constraints satisfied)
    PRIMAL_INFEASIBLE,     // Problem is likely primal infeasible (restoration failed)
    DUAL_INFEASIBLE,       // Problem is likely dual infeasible (unbounded) - rarely detected in current impl
    NUMERICAL_ERROR,       // Linear solver failed or other numerical issues
    UNSOLVED               // Not yet solved
};

// Helper to get string from status
inline const char* status_to_string(SolverStatus status) {
    switch(status) {
        case SolverStatus::SOLVED: return "SOLVED";
        case SolverStatus::MAX_ITER: return "MAX_ITER";
        case SolverStatus::FEASIBLE: return "FEASIBLE";
        case SolverStatus::PRIMAL_INFEASIBLE: return "PRIMAL_INFEASIBLE";
        case SolverStatus::DUAL_INFEASIBLE: return "DUAL_INFEASIBLE";
        case SolverStatus::NUMERICAL_ERROR: return "NUMERICAL_ERROR";
        case SolverStatus::UNSOLVED: return "UNSOLVED";
        default: return "UNKNOWN";
    }
}

template<typename T, int _NX, int _NU, int _NC, int _NP>
struct KnotPoint {
    // --- Eigen Memory Alignment (Only needed if backend is Eigen) ---
    #ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    #endif

    // === EXPOSE TEMPLATE ARGUMENTS AS CONSTANTS ===
    static const int NX = _NX;
    static const int NU = _NU;
    static const int NC = _NC; // Number of Constraints
    static const int NP = _NP;

    // --- PRIMAL VARIABLES ---
    MSVec<T, NX> x;
    MSVec<T, NU> u;
    MSVec<T, NP> p; // Parameters

    // --- DUAL VARIABLES & SLACKS ---
    // Inequality Constraints: g(x,u) + s = 0, s >= 0
    // Standard IPM slacks/duals
    MSVec<T, NC> s;   // Slack variables (s_hard in L1 formulation)
    MSVec<T, NC> lam; // Dual variables (Lambda)
    
    // [NEW] Soft Constraint Variables
    // soft_s: The slack variable 's_soft' in g(x) - s_soft + s_hard = 0
    // soft_dual: The dual variable 'nu' for the constraint s_soft >= 0
    MSVec<T, NC> soft_s; 
    MSVec<T, NC> soft_dual; 

    // --- MODEL DATA (Derivatives) ---
    // Dynamics: x_{k+1} = A x_k + B u_k + f_resid
    // Here f_resid stores the predicted next state f(x,u)
    MSMat<T, NX, NX> A;
    MSMat<T, NX, NU> B;
    MSVec<T, NX> f_resid;

    // Constraints: C x + D u + g_val + s = 0
    MSMat<T, NC, NX> C;
    MSMat<T, NC, NU> D;
    MSVec<T, NC>  g_val; // Value of g(x,u)

    // Cost: 0.5 x'Qx + x'Qu + ...
    T cost; // Scalar Cost Value
    MSVec<T, NX> q;
    MSVec<T, NU> r;
    MSMat<T, NX, NX> Q;
    MSMat<T, NU, NU> R;
    MSMat<T, NU, NX> H; // Cross term u'H x

    // --- SOLVER DATA (Barrier Modified) ---
    // These are the "effective" Q, R, q, r used in the Riccati pass
    // They include the barrier terms from the Interior Point Method
    MSMat<T, NX, NX> Q_bar;
    MSMat<T, NU, NU> R_bar;
    MSMat<T, NU, NX> H_bar;
    MSVec<T, NX>  q_bar;
    MSVec<T, NU>  r_bar;

    // --- SEARCH DIRECTIONS ---
    MSVec<T, NX> dx;
    MSVec<T, NU> du;
    MSVec<T, NC> ds;
    MSVec<T, NC> dlam;
    
    // [NEW] Soft Slack Steps
    MSVec<T, NC> dsoft_s;
    MSVec<T, NC> dsoft_dual;

    // Feedback Gains
    MSMat<T, NU, NX> K;
    MSVec<T, NU>  d;

    KnotPoint() {
        set_zero();
        initialize_defaults();
    }

    void set_zero() {
        MatOps::setZero(x); MatOps::setZero(u); MatOps::setZero(p);
        s.setOnes(); lam.setOnes(); 
        soft_s.setOnes(); soft_dual.setOnes(); // Initialize to valid interior point

        MatOps::setIdentity(A); MatOps::setZero(B); MatOps::setZero(f_resid);
        MatOps::setZero(C); MatOps::setZero(D); MatOps::setZero(g_val);

        cost = 0; 
        MatOps::setIdentity(Q); MatOps::setIdentity(R); MatOps::setZero(H); MatOps::setZero(q); MatOps::setZero(r);
        
        MatOps::setZero(dx); MatOps::setZero(du); MatOps::setZero(ds); MatOps::setZero(dlam); 
        MatOps::setZero(dsoft_s); MatOps::setZero(dsoft_dual);
        MatOps::setZero(K); MatOps::setZero(d);
    }

    void initialize_defaults() {
        // Default slacks/duals to small positive numbers to avoid NaN in first iteration
        s.fill(1.0);
        lam.fill(1.0);
        soft_s.fill(1.0);
        soft_dual.fill(1.0);
    }
};
}
