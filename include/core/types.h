#pragma once
#include <Eigen/Dense>

namespace roboopt {

template<typename T, int _NX, int _NU, int _NC, int _NP>
struct KnotPoint {
    // === EXPOSE TEMPLATE ARGUMENTS AS CONSTANTS ===
    static const int NX = _NX;
    static const int NU = _NU;
    static const int NC = _NC; // Number of Constraints
    static const int NP = _NP;

    // --- PRIMAL VARIABLES ---
    Eigen::Matrix<T, NX, 1> x;
    Eigen::Matrix<T, NU, 1> u;
    Eigen::Matrix<T, NP, 1> p; // Parameters

    // --- DUAL VARIABLES & SLACKS ---
    // Inequality Constraints: g(x,u) + s = 0, s >= 0
    Eigen::Matrix<T, NC, 1> s;   // Slack variables
    Eigen::Matrix<T, NC, 1> lam; // Dual variables (Lambda)

    // --- MODEL DATA (Derivatives) ---
    // Dynamics: x_{k+1} = A x_k + B u_k + f_resid
    Eigen::Matrix<T, NX, NX> A;
    Eigen::Matrix<T, NX, NU> B;
    Eigen::Matrix<T, NX, 1> f_resid;

    // Constraints: C x + D u + g_val + s = 0
    Eigen::Matrix<T, NC, NX> C;
    Eigen::Matrix<T, NC, NU> D;
    Eigen::Matrix<T, NC, 1>  g_val; // Value of g(x,u)

    // Cost: 0.5 x'Qx + x'Qu + ...
    Eigen::Matrix<T, NX, 1> q;
    Eigen::Matrix<T, NU, 1> r;
    Eigen::Matrix<T, NX, NX> Q;
    Eigen::Matrix<T, NU, NU> R;
    Eigen::Matrix<T, NU, NX> H; // Cross term u'H x

    // --- SOLVER DATA (Barrier Modified) ---
    // These are the "effective" Q, R, q, r used in the Riccati pass
    // They include the barrier terms from the Interior Point Method
    Eigen::Matrix<T, NX, NX> Q_bar;
    Eigen::Matrix<T, NU, NU> R_bar;
    Eigen::Matrix<T, NU, NX> H_bar;
    Eigen::Matrix<T, NX, 1>  q_bar;
    Eigen::Matrix<T, NU, 1>  r_bar;

    // Condensed System for Backward Pass
    Eigen::Matrix<T, NX, NX> op_A;
    Eigen::Matrix<T, NX, 1>  op_b;

    // --- SEARCH DIRECTIONS ---
    Eigen::Matrix<T, NX, 1> dx;
    Eigen::Matrix<T, NU, 1> du;
    Eigen::Matrix<T, NC, 1> ds;
    Eigen::Matrix<T, NC, 1> dlam;

    // Feedback Gains
    Eigen::Matrix<T, NU, NX> K;
    Eigen::Matrix<T, NU, 1>  d;

    KnotPoint() {
        set_zero();
        initialize_defaults();
    }

    void set_zero() {
        x.setZero(); u.setZero(); p.setZero();
        s.setOnes(); lam.setOnes(); // Initialize to valid interior point usually

        A.setIdentity(); B.setZero(); f_resid.setZero();
        C.setZero(); D.setZero(); g_val.setZero();

        Q.setIdentity(); R.setIdentity(); H.setZero(); q.setZero(); r.setZero();
        
        dx.setZero(); du.setZero(); ds.setZero(); dlam.setZero();
        K.setZero(); d.setZero();
    }

    void initialize_defaults() {
        // Default slacks/duals to small positive numbers to avoid NaN in first iteration
        s.fill(1.0);
        lam.fill(1.0);
    }
};
}
