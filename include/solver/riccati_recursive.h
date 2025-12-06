#pragma once
#include <vector>
#include <Eigen/Dense>
#include "core/types.h"

namespace roboopt {

    // Helper: Compute Barrier-Modified Q, R, q, r
    template<typename Knot>
    void compute_barrier_derivatives(Knot& kp, double mu) {
        // Unpack shortcuts
        // J = [C, D]
        // Sigma = S^{-1} * V
        // r_prim = g_val + s
        // r_cent = s .* lam - mu
        // modified_grad_term = S^{-1} * (V * r_prim - r_cent)

        // 1. Calculate Sigma (Diagonal) and Gradient modifier
        Eigen::Matrix<double, Knot::NC, 1> sigma;
        Eigen::Matrix<double, Knot::NC, 1> grad_mod;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            double lam_i = kp.lam(i);
            
            // Safety against small s
            if(s_i < 1e-8) s_i = 1e-8;

            sigma(i) = lam_i / s_i;
            
            double r_prim_i = kp.g_val(i) + s_i;
            double r_cent_i = (s_i * lam_i) - mu;
            
            grad_mod(i) = (1.0 / s_i) * (lam_i * r_prim_i - r_cent_i);
        }

        Eigen::DiagonalMatrix<double, Knot::NC> SigmaMat(sigma);

        // 2. Modify Hessian (Q_bar, R_bar, H_bar)
        // Q_bar = Q + C^T * Sigma * C
        kp.Q_bar = kp.Q + kp.C.transpose() * SigmaMat * kp.C;
        
        // R_bar = R + D^T * Sigma * D
        kp.R_bar = kp.R + kp.D.transpose() * SigmaMat * kp.D;

        // H_bar = H + D^T * Sigma * C
        // Note: Our H definition in KnotPoint might be Q_ux or Q_xu depending on convention.
        // Assuming H is term for u^T H x.
        kp.H_bar = kp.H + kp.D.transpose() * SigmaMat * kp.C;

        // 3. Modify Gradient (q_bar, r_bar)
        // q_bar = q + C^T * grad_mod
        kp.q_bar = kp.q + kp.C.transpose() * grad_mod;

        // r_bar = r + D^T * grad_mod
        kp.r_bar = kp.r + kp.D.transpose() * grad_mod;
    }

    template<typename Knot>
    void recover_dual_search_directions(Knot& kp, double mu) {
        // Delta s = -r_prim - (C dx + D du)
        // Delta lam = -S^{-1} (r_cent + V * Delta s)

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

    template<typename Knot>
    void cpu_serial_solve(std::vector<Knot>& traj, double mu) {
        int N = traj.size() - 1; // Horizon

        // --- 1. Preparation Phase (Compute Derivatives & Barrier Terms) ---
        // Note: Model::compute() should have been called before this.
        for(auto& kp : traj) {
            compute_barrier_derivatives(kp, mu);
        }

        // --- 2. Backward Pass (Riccati Recursion) ---
        
        // Initialize Value Function at Terminal Step (No control, only state cost + constraint)
        // Assuming terminal constraints exist? For now, using just Q_bar, q_bar from last step.
        Eigen::Matrix<double, Knot::NX, 1> Vx = traj[N].q_bar;
        Eigen::Matrix<double, Knot::NX, Knot::NX> Vxx = traj[N].Q_bar;

        for(int k = N - 1; k >= 0; --k) {
            auto& kp = traj[k];

            // Q_x = q_bar + A' * Vx
            Eigen::Matrix<double, Knot::NX, 1> Qx = kp.q_bar + kp.A.transpose() * Vx;

            // Q_u = r_bar + B' * Vx
            Eigen::Matrix<double, Knot::NU, 1> Qu = kp.r_bar + kp.B.transpose() * Vx;

            // Q_xx = Q_bar + A' * Vxx * A
            Eigen::Matrix<double, Knot::NX, Knot::NX> Qxx = kp.Q_bar + kp.A.transpose() * Vxx * kp.A;

            // Q_uu = R_bar + B' * Vxx * B
            Eigen::Matrix<double, Knot::NU, Knot::NU> Quu = kp.R_bar + kp.B.transpose() * Vxx * kp.B;

            // Q_ux = H_bar + B' * Vxx * A
            Eigen::Matrix<double, Knot::NU, Knot::NX> Qux = kp.H_bar + kp.B.transpose() * Vxx * kp.A;

            // Regularization
            for(int i=0; i<Knot::NU; ++i) Quu(i,i) += 1e-6;

            // Compute Gains
            Eigen::LLT<Eigen::Matrix<double, Knot::NU, Knot::NU>> llt(Quu);
            // d = -inv(Quu) * Qu
            Eigen::Matrix<double, Knot::NU, 1> d = -llt.solve(Qu);
            // K = -inv(Quu) * Qux
            Eigen::Matrix<double, Knot::NU, Knot::NX> K = -llt.solve(Qux);

            kp.d = d;
            kp.K = K;

            // Update Value Function
            // Vx = Qx + K' Quu d + K' Qu + Qux' d
            Vx = Qx + K.transpose() * Quu * d + K.transpose() * Qu + Qux.transpose() * d;
            
            // Vxx = Qxx + K' Quu K + K' Qux + Qux' K
            Vxx = Qxx + K.transpose() * Quu * K + K.transpose() * Qux + Qux.transpose() * K;
            Vxx = 0.5 * (Vxx + Vxx.transpose());
        }

        // --- 3. Forward Pass (Linear Rollout of Delta) ---
        // This computes the search direction (dx, du) for the whole trajectory
        traj[0].dx.setZero(); // Initial state fixed (usually)

        for(int k=0; k < N; ++k) {
            auto& kp = traj[k];
            
            // du = K dx + d
            kp.du = kp.K * kp.dx + kp.d;

            // dx_{k+1} = A dx + B du + (optional f_resid?)
            // Note: In standard Newton/SQP, we are solving for delta from the current linearization.
            // The defect d = f(x,u) - x_{k+1} is integrated into the "q" and "r" terms usually?
            // Or explicitly: dx_{k+1} = A dx + B du + f_resid.
            // In our CarModel, f_resid is the *next state*.
            // The defect is: d = f(x_bar, u_bar) - x_bar_{k+1}.
            // Standard formulation: dx_{k+1} = A dx_k + B du_k + defect.
            // Our f_resid in CarModel::compute stores explicit f(x,u).
            // So defect = kp.f_resid - traj[k+1].x.
            
            Eigen::Matrix<double, Knot::NX, 1> defect = kp.f_resid - traj[k+1].x;

            // Propagate dx
            traj[k+1].dx = kp.A * kp.dx + kp.B * kp.du + defect;

            // Recover Dual/Slack Directions
            recover_dual_search_directions(kp, mu);
        }
        
        // Recover last step duals (though no control there)
        recover_dual_search_directions(traj[N], mu);
    }
}
