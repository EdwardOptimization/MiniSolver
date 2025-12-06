#pragma once
#include <vector>
#include <Eigen/Dense>
#include "core/types.h"
#include "core/solver_options.h" 
#include "core/matrix_defs.h"

namespace minisolver {

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

        #ifdef USE_EIGEN
        Eigen::DiagonalMatrix<double, Knot::NC> SigmaMat(sigma);
        kp.Q_bar.noalias() = kp.Q + kp.C.transpose() * SigmaMat * kp.C;
        kp.R_bar.noalias() = kp.R + kp.D.transpose() * SigmaMat * kp.D;
        kp.H_bar.noalias() = kp.H + kp.D.transpose() * SigmaMat * kp.C;
        kp.q_bar.noalias() = kp.q + kp.C.transpose() * grad_mod;
        kp.r_bar.noalias() = kp.r + kp.D.transpose() * grad_mod;
        #else
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

    template<typename MatrixType>
    bool fast_inverse(const MatrixType& A, MatrixType& A_inv) {
        if constexpr (MatrixType::RowsAtCompileTime == 1) {
            if (std::abs(A(0,0)) < 1e-9) return false;
            A_inv(0,0) = 1.0 / A(0,0);
            return true;
        } 
        else if constexpr (MatrixType::RowsAtCompileTime == 2) {
            double det = A(0,0)*A(1,1) - A(0,1)*A(1,0);
            if (std::abs(det) < 1e-9) return false;
            double inv_det = 1.0 / det;
            A_inv(0,0) =  A(1,1) * inv_det;
            A_inv(0,1) = -A(0,1) * inv_det;
            A_inv(1,0) = -A(1,0) * inv_det;
            A_inv(1,1) =  A(0,0) * inv_det;
            return true;
        }
        else {
            Eigen::LLT<MatrixType> llt(A);
            if (llt.info() != Eigen::Success) return false;
            A_inv = llt.solve(MatrixType::Identity());
            return true;
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

            // Optimized Expansion
            MSMat<double, Knot::NX, Knot::NX> VxxA;
            VxxA.noalias() = Vxx * kp.A;
            
            MSMat<double, Knot::NX, Knot::NU> VxxB;
            VxxB.noalias() = Vxx * kp.B; 

            // Calculate gradients Qx, Qu
            MSMat<double, Knot::NX, 1> Qx = kp.q_bar;
            Qx.noalias() += kp.A.transpose() * Vx;
            
            MSMat<double, Knot::NU, 1> Qu = kp.r_bar;
            Qu.noalias() += kp.B.transpose() * Vx;
            
            // Calculate Hessians Qxx, Quu, Qux
            // Initialize with Barrier terms, then add expansion terms
            MSMat<double, Knot::NX, Knot::NX> Qxx = kp.Q_bar;
            Qxx.noalias() += kp.A.transpose() * VxxA;
            
            MSMat<double, Knot::NU, Knot::NU> Quu = kp.R_bar;
            Quu.noalias() += kp.B.transpose() * VxxB;
            
            MSMat<double, Knot::NU, Knot::NX> Qux = kp.H_bar;
            Qux.noalias() += kp.B.transpose() * VxxA;

            if (strategy == InertiaStrategy::REGULARIZATION) {
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += reg;
            } else {
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += 1e-9;
            }

            // Cholesky Solve
            if (strategy == InertiaStrategy::REGULARIZATION && Knot::NU <= 2) {
                MSMat<double, Knot::NU, Knot::NU> Quu_inv;
                if (!fast_inverse(Quu, Quu_inv)) return false;
                kp.d.noalias() = -Quu_inv * Qu;
                kp.K.noalias() = -Quu_inv * Qux;
            } 
            else {
                Eigen::LLT<MSMat<double, Knot::NU, Knot::NU>> llt(Quu);
                if(llt.info() == Eigen::NumericalIssue) {
                    if (strategy == InertiaStrategy::REGULARIZATION) return false;
                    if (strategy == InertiaStrategy::IGNORE_SINGULAR) {
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
                }
                kp.d = llt.solve(-Qu);
                kp.K = llt.solve(-Qux);
            }

            // Update Value Function
            // Vx = Qx + K' (Quu d + Qu) + K' Qux' d + Qux' d
            // Simplified: Vx = Qx + Qux' d + Qux' K x ? No.
            // Simplified: Vx = Qx + Qux^T * d + Vxx * ...
            // Correct Simplified: Vx = Qx + Qux^T * d
            // Correct Simplified: Vxx = Qxx + Qux^T * K
            
            // Re-use Qx memory for Vx? No, Vx needs to be propagated.
            // But we can compute Vx directly.
            Vx = Qx;
            Vx.noalias() += Qux.transpose() * kp.d;

            Vxx = Qxx;
            Vxx.noalias() += Qux.transpose() * kp.K;
            
            // Symmetrize
            // Vxx = 0.5 * (Vxx + Vxx.transpose()); 
            // In high perf, maybe just copy upper to lower? Or trust symmetry?
            // Trusting symmetry saves ops. But accumulated error might be an issue.
            // Let's do it efficiently:
            // Vxx.template triangularView<Eigen::Lower>() = Vxx.transpose();
            // This copies upper to lower.
            // For 4x4, full add + scale might be faster due to SIMD.
            Vxx = 0.5 * (Vxx + Vxx.transpose());
            
            for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;
        }

        traj[0].dx.setZero(); 

        for(int k=0; k < N; ++k) {
            auto& kp = traj[k];
            kp.du.noalias() = kp.K * kp.dx + kp.d;
            MSVec<double, Knot::NX> defect = kp.f_resid - traj[k+1].x;
            // dx_next = A dx + B du + defect
            // Use noalias
            traj[k+1].dx.noalias() = kp.A * kp.dx;
            traj[k+1].dx.noalias() += kp.B * kp.du;
            traj[k+1].dx += defect;
            
            recover_dual_search_directions(kp, mu);
        }
        recover_dual_search_directions(traj[N], mu);
        
        return true;
    }
}
