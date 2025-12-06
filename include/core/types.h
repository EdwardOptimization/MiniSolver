#pragma once
#include "core/solver_options.h"
#include "core/matrix_defs.h" 

namespace minisolver {

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
    MSVec<T, NC> s;   // Slack variables
    MSVec<T, NC> lam; // Dual variables (Lambda)

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

    // Condensed System for Backward Pass (Removed for CPU memory optimization)
    // MSMat<T, NX, NX> op_A;
    // MSVec<T, NX>  op_b;

    // --- SEARCH DIRECTIONS ---
    MSVec<T, NX> dx;
    MSVec<T, NU> du;
    MSVec<T, NC> ds;
    MSVec<T, NC> dlam;

    // Feedback Gains
    MSMat<T, NU, NX> K;
    MSVec<T, NU>  d;

    KnotPoint() {
        set_zero();
        initialize_defaults();
    }

    void set_zero() {
        MatOps::setZero(x); MatOps::setZero(u); MatOps::setZero(p);
        s.setOnes(); lam.setOnes(); // Initialize to valid interior point usually

        MatOps::setIdentity(A); MatOps::setZero(B); MatOps::setZero(f_resid);
        MatOps::setZero(C); MatOps::setZero(D); MatOps::setZero(g_val);

        cost = 0; // Reset cost
        MatOps::setIdentity(Q); MatOps::setIdentity(R); MatOps::setZero(H); MatOps::setZero(q); MatOps::setZero(r);
        
        MatOps::setZero(dx); MatOps::setZero(du); MatOps::setZero(ds); MatOps::setZero(dlam);
        MatOps::setZero(K); MatOps::setZero(d);
    }

    void initialize_defaults() {
        // Default slacks/duals to small positive numbers to avoid NaN in first iteration
        s.fill(1.0);
        lam.fill(1.0);
    }
};
}
