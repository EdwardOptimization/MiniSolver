#pragma once
#include <vector>
#include <Eigen/Dense>
#include "core/types.h"

namespace roboopt {

    // Helper: Compute Barrier-Modified Q, R, q, r
    //
    // The Log-Barrier Method approximates inequality constraints g(x) <= 0
    // by adding a cost term: -mu * sum(log(-g(x))).
    // 
    // This function computes the derivatives of the barrier-augmented Lagrangian.
    // The Hessian of the barrier term involves Sigma = S^{-1} * Lambda.
    //
    // Q_bar = Q + C^T * Sigma * C
    // R_bar = R + D^T * Sigma * D
    // q_bar = q + C^T * (S^{-1} * (Lambda * r_prim - r_cent))
    template<typename Knot>
    void compute_barrier_derivatives(Knot& kp, double mu) {
        // Unpack shortcuts
        // J = [C, D]
        // Sigma = S^{-1} * V (V is Lambda)
        // r_prim = g_val + s (Primal residual)
        // r_cent = s .* lam - mu (Centrality residual)
        // modified_grad_term = S^{-1} * (V * r_prim - r_cent)

        // 1. Calculate Sigma (Diagonal) and Gradient modifier
        Eigen::Matrix<double, Knot::NC, 1> sigma;
        Eigen::Matrix<double, Knot::NC, 1> grad_mod;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            double lam_i = kp.lam(i);
            
            // Safety against small s to prevent division by zero
            if(s_i < 1e-8) s_i = 1e-8;

            sigma(i) = lam_i / s_i;
            
            double r_prim_i = kp.g_val(i) + s_i;
            double r_cent_i = (s_i * lam_i) - mu;
            
            grad_mod(i) = (1.0 / s_i) * (lam_i * r_prim_i - r_cent_i);
        }

        Eigen::DiagonalMatrix<double, Knot::NC> SigmaMat(sigma);

        // 2. Modify Hessian (Q_bar, R_bar, H_bar)
        // These terms represent the curvature of the barrier function
        // Q_bar = Q + C^T * Sigma * C
        kp.Q_bar = kp.Q + kp.C.transpose() * SigmaMat * kp.C;
        
        // R_bar = R + D^T * Sigma * D
        kp.R_bar = kp.R + kp.D.transpose() * SigmaMat * kp.D;

        // H_bar = H + D^T * Sigma * C
        // Cross-term coupling state and control constraints
        kp.H_bar = kp.H + kp.D.transpose() * SigmaMat * kp.C;

        // 3. Modify Gradient (q_bar, r_bar)
        // These terms drive the solution away from the boundaries
        // q_bar = q + C^T * grad_mod
        kp.q_bar = kp.q + kp.C.transpose() * grad_mod;

        // r_bar = r + D^T * grad_mod
        kp.r_bar = kp.r + kp.D.transpose() * grad_mod;
    }

    template<typename Knot>
    void recover_dual_search_directions(Knot& kp, double mu) {
        // Recover step directions for slack (s) and dual (lambda) variables
        // based on the Primal-Dual system equations:
        // 
        // 1. Linearized Primal Feasibility: C*dx + D*du + ds = -r_prim
        //    => ds = -r_prim - (C*dx + D*du)
        //
        // 2. Linearized Complementarity: S*dlam + Lam*ds = -r_cent
        //    => dlam = -S^{-1} * (r_cent + Lam*ds)

        Eigen::Matrix<double, Knot::NC, 1> r_prim = kp.g_val + kp.s;
        Eigen::Matrix<double, Knot::NC, 1> constraint_step = kp.C * kp.dx + kp.D * kp.du;

        kp.ds = -r_prim - constraint_step;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            if(s_i < 1e-8) s_i = 1e-8;
            double r_cent_i = (s_i * kp.lam(i)) - mu;
            kp.dlam(i) = -(1.0 / s_i) * (r_cent_i + kp.lam(i) * kp.ds(i));
        }
    }

    // UPDATED: Now accepts `reg` parameter
    // Solves the KKT system using the Riccati Recursion (LQR-like Backward Pass)
    template<typename Knot>
    void cpu_serial_solve(std::vector<Knot>& traj, double mu, double reg) {
        int N = traj.size() - 1; // Horizon

        // --- 1. Preparation Phase (Compute Derivatives & Barrier Terms) ---
        // Note: Model::compute() should have been called before this.
        for(auto& kp : traj) {
            compute_barrier_derivatives(kp, mu);
        }

        // --- 2. Backward Pass (Riccati Recursion) ---
        // Propagates the Value Function V(x) backwards from N to 0.
        // V(x) is approximated as a quadratic: 0.5*x'Vxx*x + x'Vx
        
        // Initialize Value Function at Terminal Step (No control, only state cost + constraint)
        Eigen::Matrix<double, Knot::NX, 1> Vx = traj[N].q_bar;
        Eigen::Matrix<double, Knot::NX, Knot::NX> Vxx = traj[N].Q_bar;
        
        // Add State Regularization to terminal cost to ensure positive definiteness
        for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;

        for(int k = N - 1; k >= 0; --k) {
            auto& kp = traj[k];

            // --- A. Expansion of Q-Function (Action-Value Function) ---
            // Q(x,u) approx = 0.5*[x;u]'[Qxx Qxu; Qux Quu][x;u] + [x;u]'[Qx; Qu]
            // Derived from Bellman Equation: V_k(x) = min_u { L_k(x,u) + V_{k+1}(f(x,u)) }
            
            // Q_x = q_bar + A' * V_x
            Eigen::Matrix<double, Knot::NX, 1> Qx = kp.q_bar + kp.A.transpose() * Vx;

            // Q_u = r_bar + B' * V_x
            Eigen::Matrix<double, Knot::NU, 1> Qu = kp.r_bar + kp.B.transpose() * Vx;

            // Q_xx = Q_bar + A' * V_xx * A
            Eigen::Matrix<double, Knot::NX, Knot::NX> Qxx = kp.Q_bar + kp.A.transpose() * Vxx * kp.A;

            // Q_uu = R_bar + B' * V_xx * B
            Eigen::Matrix<double, Knot::NU, Knot::NU> Quu = kp.R_bar + kp.B.transpose() * Vxx * kp.B;

            // Q_ux = H_bar + B' * V_xx * A
            Eigen::Matrix<double, Knot::NU, Knot::NX> Qux = kp.H_bar + kp.B.transpose() * Vxx * kp.A;

            // --- B. Regularization (Levenberg-Marquardt style) ---
            // Ensures Quu is invertible and positive definite.
            // Effectively damps the step size if the Hessian is not convex.
            for(int i=0; i<Knot::NU; ++i) Quu(i,i) += reg;

            // --- C. Compute Gains (Newton Step for u) ---
            // min_u Q(x,u) -> du = -inv(Quu) * (Qu + Qux*dx)
            // du = d + K*dx
            Eigen::LLT<Eigen::Matrix<double, Knot::NU, Knot::NU>> llt(Quu);
            
            if(llt.info() == Eigen::NumericalIssue) {
                // In a robust solver, we would return a status flag to increase 'reg' and retry.
            }

            // Feedforward term: d = -inv(Quu) * Qu
            Eigen::Matrix<double, Knot::NU, 1> d = -llt.solve(Qu);
            // Feedback gain: K = -inv(Quu) * Qux
            Eigen::Matrix<double, Knot::NU, Knot::NX> K = -llt.solve(Qux);

            kp.d = d;
            kp.K = K;

            // --- D. Update Value Function for next step (k-1) ---
            // Substitute optimal u into Q(x,u) to get V(x)
            // Vx = Qx + K' Quu d + K' Qu + Qux' d
            Vx = Qx + K.transpose() * Quu * d + K.transpose() * Qu + Qux.transpose() * d;
            
            // Vxx = Qxx + K' Quu K + K' Qux + Qux' K
            Vxx = Qxx + K.transpose() * Quu * K + K.transpose() * Qux + Qux.transpose() * K;
            
            // Symmetrize Vxx to maintain numerical stability
            Vxx = 0.5 * (Vxx + Vxx.transpose());
            
            // Add State Regularization for next step propagation
            for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;
        }

        // --- 3. Forward Pass (Linear Rollout of Delta) ---
        // Simulates the linearized system: dx_{k+1} = A*dx_k + B*du_k
        traj[0].dx.setZero(); // Initial state fixed (usually)

        for(int k=0; k < N; ++k) {
            auto& kp = traj[k];
            
            // Apply control law: du = K*dx + d
            kp.du = kp.K * kp.dx + kp.d;
            
            // Propagate state delta
            // dx_{k+1} = A*dx + B*du + defect
            // defect accounts for the gap between the linearization point and the actual dynamics
            Eigen::Matrix<double, Knot::NX, 1> defect = kp.f_resid - traj[k+1].x;
            traj[k+1].dx = kp.A * kp.dx + kp.B * kp.du + defect;

            // Recover Dual/Slack Directions now that we know dx and du
            recover_dual_search_directions(kp, mu);
        }
        
        recover_dual_search_directions(traj[N], mu);
    }
}
