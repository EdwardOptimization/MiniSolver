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
// NEW ARCHITECTURE: Split Design
// =============================================================================

// 1. StateNode: Lightweight state variables (需双缓冲 for Line Search)
// 包含所有随迭代更新的变量：primal, dual, soft constraint vars, evaluation results
template<typename T, int _NX, int _NU, int _NC, int _NP>
struct StateNode {
    #ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    #endif

    static const int NX = _NX;
    static const int NU = _NU;
    static const int NC = _NC;
    static const int NP = _NP;

    // === PRIMAL VARIABLES ===
    MSVec<T, NX> x;
    MSVec<T, NU> u;
    MSVec<T, NP> p; // Parameters (e.g., reference trajectory)

    // === DUAL VARIABLES & SLACKS (IPM State) ===
    MSVec<T, NC> s;         // Slack variables (s_hard)
    MSVec<T, NC> lam;       // Lagrange Multipliers (Lambda)
    
    // === SOFT CONSTRAINT VARIABLES ===
    // These must be part of StateNode because they are updated in Line Search
    MSVec<T, NC> soft_s;    // Soft Constraint Slack (s_soft)
    MSVec<T, NC> soft_dual; // Soft Constraint Dual (nu) - for L1, this is implicitly w - lam

    // === EVALUATION RESULTS (Scalars/Vectors) ===
    T cost;                 // Scalar cost value at this point
    MSVec<T, NC> g_val;     // Constraint residuals g(x, u)

    StateNode() {
        set_zero();
        initialize_defaults();
    }

    void set_zero() {
        MatOps::setZero(x);
        MatOps::setZero(u);
        MatOps::setZero(p);
        
        s.setOnes();
        lam.setOnes();
        soft_s.setOnes();
        soft_dual.setOnes();
        
        cost = 0;
        MatOps::setZero(g_val);
    }

    void initialize_defaults() {
        // Default slacks/duals to small positive numbers to avoid NaN
        s.fill(1.0);
        lam.fill(1.0);
        soft_s.fill(1.0);
        soft_dual.fill(1.0);
    }

    // Efficient copy for Line Search candidate preparation
    void copy_from(const StateNode& other) {
        x = other.x;
        u = other.u;
        p = other.p;
        s = other.s;
        lam = other.lam;
        soft_s = other.soft_s;
        soft_dual = other.soft_dual;
        cost = other.cost;
        g_val = other.g_val;
    }
};

// 2. ModelData: Linearization derivatives (单份，只读 during Line Search/SOC)
// 存储当前迭代点的导数信息，在一次 step() 中保持不变
template<typename T, int _NX, int _NU, int _NC>
struct ModelData {
    #ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    #endif

    static const int NX = _NX;
    static const int NU = _NU;
    static const int NC = _NC;

    // === DYNAMICS DERIVATIVES ===
    // Linearized dynamics: x_{k+1} ≈ A x_k + B u_k + f_resid
    MSMat<T, NX, NX> A;
    MSMat<T, NX, NU> B;
    MSVec<T, NX> f_resid;   // Predicted next state f(x, u)

    // === CONSTRAINT DERIVATIVES ===
    // Linearized constraints: g(x,u) ≈ C x + D u + g_val = 0
    // Note: g_val is stored in StateNode because it varies with x
    MSMat<T, NC, NX> C;
    MSMat<T, NC, NU> D;

    // === COST DERIVATIVES (Objective Only, no Barrier terms) ===
    // Quadratic approximation: L ≈ 0.5 x'Qx + x'q + 0.5 u'Ru + u'r + u'Hx
    MSVec<T, NX> q;
    MSVec<T, NU> r;
    MSMat<T, NX, NX> Q;
    MSMat<T, NU, NU> R;
    MSMat<T, NU, NX> H;     // Cross term

    ModelData() {
        set_zero();
    }

    void set_zero() {
        MatOps::setIdentity(A);
        MatOps::setZero(B);
        MatOps::setZero(f_resid);
        
        MatOps::setZero(C);
        MatOps::setZero(D);
        
        MatOps::setIdentity(Q);
        MatOps::setIdentity(R);
        MatOps::setZero(H);
        MatOps::setZero(q);
        MatOps::setZero(r);
    }
};

// 3. SolverWorkspace: Solver working memory (单份，可覆写)
// 包含 Barrier 修正后的 KKT 矩阵和 Riccati 求解的中间结果
template<typename T, int _NX, int _NU, int _NC>
struct SolverWorkspace {
    #ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    #endif

    static const int NX = _NX;
    static const int NU = _NU;
    static const int NC = _NC;

    // === BARRIER MODIFIED MATRICES (Riccati Input) ===
    // These include the barrier terms: Q_bar = Q + C^T Σ C, etc.
    MSMat<T, NX, NX> Q_bar;
    MSMat<T, NU, NU> R_bar;
    MSMat<T, NU, NX> H_bar;
    MSVec<T, NX> q_bar;
    MSVec<T, NU> r_bar;

    // === RICCATI FEEDBACK (Output) ===
    MSMat<T, NU, NX> K;     // Feedback gain
    MSVec<T, NU> d;         // Feedforward term

    // === SEARCH DIRECTIONS (Output) ===
    MSVec<T, NX> dx;
    MSVec<T, NU> du;
    MSVec<T, NC> ds;
    MSVec<T, NC> dlam;
    
    // Soft constraint search directions
    MSVec<T, NC> dsoft_s;
    MSVec<T, NC> dsoft_dual;

    // === HELPER WORKSPACE (for Riccati recursion) ===
    // These are reused across iterations to avoid allocation
    MSMat<T, NX, NX> VxxA;
    MSMat<T, NX, NU> VxxB;
    MSVec<T, NX> Vxx_d;
    MSMat<T, NU, NU> Quu_inv;

    SolverWorkspace() {
        set_zero();
    }

    void set_zero() {
        MatOps::setZero(Q_bar);
        MatOps::setZero(R_bar);
        MatOps::setZero(H_bar);
        MatOps::setZero(q_bar);
        MatOps::setZero(r_bar);
        
        MatOps::setZero(K);
        MatOps::setZero(d);
        
        MatOps::setZero(dx);
        MatOps::setZero(du);
        MatOps::setZero(ds);
        MatOps::setZero(dlam);
        MatOps::setZero(dsoft_s);
        MatOps::setZero(dsoft_dual);
        
        MatOps::setZero(VxxA);
        MatOps::setZero(VxxB);
        MatOps::setZero(Vxx_d);
        MatOps::setZero(Quu_inv);
    }
};

// =============================================================================
// COMPOSITE TYPE: Combines all three for convenience
// This is optional and used for easier migration
// =============================================================================
template<typename T, int _NX, int _NU, int _NC, int _NP>
struct KnotPointV2 {
    #ifdef USE_EIGEN
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    #endif

    static const int NX = _NX;
    static const int NU = _NU;
    static const int NC = _NC;
    static const int NP = _NP;

    // === PRIMAL VARIABLES ===
    MSVec<T, NX> x;
    MSVec<T, NU> u;
    MSVec<T, NP> p;

    // === DUAL VARIABLES & SLACKS ===
    MSVec<T, NC> s;
    MSVec<T, NC> lam;
    MSVec<T, NC> soft_s;
    MSVec<T, NC> soft_dual;

    // === EVALUATION RESULTS ===
    T cost;
    MSVec<T, NC> g_val;

    // === MODEL DATA (Derivatives) ===
    MSMat<T, NX, NX> A;
    MSMat<T, NX, NU> B;
    MSVec<T, NX> f_resid;
    MSMat<T, NC, NX> C;
    MSMat<T, NC, NU> D;
    MSVec<T, NX> q;
    MSVec<T, NU> r;
    MSMat<T, NX, NX> Q;
    MSMat<T, NU, NU> R;
    MSMat<T, NU, NX> H;

    // === SOLVER DATA (Barrier Modified) ===
    MSMat<T, NX, NX> Q_bar;
    MSMat<T, NU, NU> R_bar;
    MSMat<T, NU, NX> H_bar;
    MSVec<T, NX> q_bar;
    MSVec<T, NU> r_bar;

    // === SEARCH DIRECTIONS ===
    MSVec<T, NX> dx;
    MSVec<T, NU> du;
    MSVec<T, NC> ds;
    MSVec<T, NC> dlam;
    MSVec<T, NC> dsoft_s;
    MSVec<T, NC> dsoft_dual;

    // === FEEDBACK GAINS ===
    MSMat<T, NU, NX> K;
    MSVec<T, NU> d;

    KnotPointV2() {
        set_zero();
        initialize_defaults();
    }

    void set_zero() {
        MatOps::setZero(x); MatOps::setZero(u); MatOps::setZero(p);
        s.setOnes(); lam.setOnes(); 
        soft_s.setOnes(); soft_dual.setOnes();

        MatOps::setIdentity(A); MatOps::setZero(B); MatOps::setZero(f_resid);
        MatOps::setZero(C); MatOps::setZero(D); MatOps::setZero(g_val);

        cost = 0; 
        MatOps::setIdentity(Q); MatOps::setIdentity(R); MatOps::setZero(H); 
        MatOps::setZero(q); MatOps::setZero(r);
        
        MatOps::setZero(Q_bar); MatOps::setZero(R_bar); MatOps::setZero(H_bar);
        MatOps::setZero(q_bar); MatOps::setZero(r_bar);
        
        MatOps::setZero(dx); MatOps::setZero(du); MatOps::setZero(ds); MatOps::setZero(dlam); 
        MatOps::setZero(dsoft_s); MatOps::setZero(dsoft_dual);
        MatOps::setZero(K); MatOps::setZero(d);
    }

    void initialize_defaults() {
        s.fill(1.0);
        lam.fill(1.0);
        soft_s.fill(1.0);
        soft_dual.fill(1.0);
    }
};

}
