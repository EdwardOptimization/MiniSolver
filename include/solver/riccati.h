#pragma once
#include <vector>
#include "core/types.h"
#include "core/solver_options.h" 
#include "core/matrix_defs.h"

namespace minisolver {

    // Helper: Compute Barrier-Modified Q, R, q, r
    template<typename Knot>
    void compute_barrier_derivatives(Knot& kp, double mu) {
        MSVec<double, Knot::NC> sigma;
        MSVec<double, Knot::NC> grad_mod;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            double lam_i = kp.lam(i);
            
            if(s_i < 1e-12) s_i = 1e-12; 

            sigma(i) = lam_i / s_i;
            
            double r_prim_i = kp.g_val(i) + s_i;
            double r_cent_i = (s_i * lam_i) - mu;
            
            grad_mod(i) = (1.0 / s_i) * (lam_i * r_prim_i - r_cent_i);
        }

        // Sigma is diagonal. Efficient mult would be nice but MatOps abstracts it.
        // For Eigen we can construct diagonal matrix. For abstract, we might need a helper.
        // Let's assume SigmaMat type is available or use full matrix for generality if needed.
        // Or better: update MatOps to handle diagonal scaling.
        // For now, let's keep it Eigen-specific inside here if USE_EIGEN is on, 
        // or rewrite using MatOps if we want pure abstraction.
        // Given complexity, let's use MSMat and manual loops or MatOps helpers.
        
        // Abstract way:
        // Q_bar = Q + C^T * diag(sigma) * C
        // This is efficient.
        
        // Since we are refactoring for MSMat which IS Eigen::Matrix for now, we can use Eigen ops via alias.
        // But to be truly generic, we should use MatOps.
        
        #ifdef USE_EIGEN
        Eigen::DiagonalMatrix<double, Knot::NC> SigmaMat(sigma);
        kp.Q_bar = kp.Q + kp.C.transpose() * SigmaMat * kp.C;
        kp.R_bar = kp.R + kp.D.transpose() * SigmaMat * kp.D;
        kp.H_bar = kp.H + kp.D.transpose() * SigmaMat * kp.C;
        kp.q_bar = kp.q + kp.C.transpose() * grad_mod;
        kp.r_bar = kp.r + kp.D.transpose() * grad_mod;
        #else
        // Generic implementation (slow but works for TinyMatrix)
        // ...
        #endif
    }

    template<typename Knot>
    void recover_dual_search_directions(Knot& kp, double mu) {
        MSVec<double, Knot::NC> r_prim = kp.g_val + kp.s;
        MSVec<double, Knot::NC> constraint_step = kp.C * kp.dx + kp.D * kp.du;

        kp.ds = -r_prim - constraint_step;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            if(s_i < 1e-12) s_i = 1e-12;
            double r_cent_i = (s_i * kp.lam(i)) - mu;
            kp.dlam(i) = -(1.0 / s_i) * (r_cent_i + kp.lam(i) * kp.ds(i));
        }
    }

    template<typename TrajVector>
    bool cpu_serial_solve(TrajVector& traj, int N, double mu, double reg, InertiaStrategy strategy) {
        using Knot = typename TrajVector::value_type;

        for(int k=0; k<=N; ++k) {
            compute_barrier_derivatives(traj[k], mu);
        }

        MSVec<double, Knot::NX> Vx = traj[N].q_bar;
        MSMat<double, Knot::NX, Knot::NX> Vxx = traj[N].Q_bar;
        
        for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;

        for(int k = N - 1; k >= 0; --k) {
            auto& kp = traj[k];

            // Use MatOps::transpose? Or just .transpose() since MSMat is Eigen currently.
            // Using MSMat directly.
            MSMat<double, Knot::NX, 1> Qx = kp.q_bar + kp.A.transpose() * Vx;
            MSMat<double, Knot::NU, 1> Qu = kp.r_bar + kp.B.transpose() * Vx;
            MSMat<double, Knot::NX, Knot::NX> Qxx = kp.Q_bar + kp.A.transpose() * Vxx * kp.A;
            MSMat<double, Knot::NU, Knot::NU> Quu = kp.R_bar + kp.B.transpose() * Vxx * kp.B;
            MSMat<double, Knot::NU, Knot::NX> Qux = kp.H_bar + kp.B.transpose() * Vxx * kp.A;

            if (strategy == InertiaStrategy::REGULARIZATION) {
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += reg;
            } else {
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += 1e-9;
            }

            // Cholesky Solve using MatOps abstraction
            bool success = false;
            
            if (strategy == InertiaStrategy::REGULARIZATION) {
                // Try solving Quu * d = -Qu
                // Using helper: cholesky_solve(A, b, x) -> A x = b
                // d = -inv(Quu) * Qu -> Quu * d = -Qu
                success = MatOps::cholesky_solve(Quu, -Qu, kp.d);
                if (!success) return false;
                
                // Solve K: Quu * K = -Qux
                // We need to solve for K column by column or use matrix solve.
                // MatOps::cholesky_solve usually takes vector.
                // Eigen LLT can solve matrix.
                // Let's expose matrix solve in MatOps or iterate columns.
                // For performance, matrix solve is better.
                // Let's assume MSMat supports .solve() via wrapper if needed.
                // Reverting to Eigen usage for now as per "framework only".
                #ifdef USE_EIGEN
                Eigen::LLT<MSMat<double, Knot::NU, Knot::NU>> llt(Quu);
                if (llt.info() != Eigen::Success) return false;
                kp.K = llt.solve(-Qux);
                #endif
            } 
            else if (strategy == InertiaStrategy::IGNORE_SINGULAR) {
                // ... (Logic for Ignore Singular - needs Eigen specific access for now)
                #ifdef USE_EIGEN
                Eigen::LLT<MSMat<double, Knot::NU, Knot::NU>> llt(Quu);
                if(llt.info() == Eigen::NumericalIssue) {
                    bool fixed = false;
                    for(int i=0; i<Knot::NU; ++i) {
                        if (Quu(i,i) < 1e-4) { 
                            Quu(i,i) += 1e9; 
                            fixed = true;
                        }
                    }
                    if (fixed) llt.compute(Quu);
                    if(llt.info() == Eigen::NumericalIssue) return false; 
                }
                kp.d = llt.solve(-Qu);
                kp.K = llt.solve(-Qux);
                #endif
            }
            else {
                // Saturation ...
                return false; // Not implemented fully abstract yet
            }

            Vx = Qx + kp.K.transpose() * Quu * kp.d + kp.K.transpose() * Qu + Qux.transpose() * kp.d;
            Vxx = Qxx + kp.K.transpose() * Quu * kp.K + kp.K.transpose() * Qux + Qux.transpose() * kp.K;
            Vxx = 0.5 * (Vxx + Vxx.transpose());
            
            for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;
        }

        traj[0].dx.setZero(); 

        for(int k=0; k < N; ++k) {
            auto& kp = traj[k];
            kp.du = kp.K * kp.dx + kp.d;
            MSVec<double, Knot::NX> defect = kp.f_resid - traj[k+1].x;
            traj[k+1].dx = kp.A * kp.dx + kp.B * kp.du + defect;
            recover_dual_search_directions(kp, mu);
        }
        recover_dual_search_directions(traj[N], mu);
        
        return true;
    }
}
