#pragma once
#include <vector>
#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif
#include "core/types.h"
#include "core/solver_options.h" 
#include "core/matrix_defs.h"

namespace minisolver {

    template<typename Knot>
    void compute_barrier_derivatives(Knot& kp, double mu, const minisolver::SolverConfig& config, const Knot* aff_kp = nullptr) {
        MSVec<double, Knot::NC> sigma;
        MSVec<double, Knot::NC> grad_mod;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            double lam_i = kp.lam(i);
            
            if(s_i < config.min_barrier_slack) s_i = config.min_barrier_slack;

            sigma(i) = lam_i / s_i;
            
            double r_prim_i = kp.g_val(i) + s_i;
            double r_cent_i = (s_i * lam_i) - mu;
            
            if (aff_kp) {
                // Mehrotra Corrector: Add ds_aff * dlam_aff
                r_cent_i += aff_kp->ds(i) * aff_kp->dlam(i);
            }
            
            grad_mod(i) = (1.0 / s_i) * (lam_i * r_prim_i - r_cent_i);
        }

        #ifdef USE_EIGEN
        Eigen::DiagonalMatrix<double, Knot::NC> SigmaMat(sigma);
        
        MSMat<double, Knot::NC, Knot::NX> tempC = SigmaMat * kp.C;
        MSMat<double, Knot::NC, Knot::NU> tempD = SigmaMat * kp.D;

        kp.Q_bar.noalias() = kp.Q + kp.C.transpose() * tempC;
        kp.R_bar.noalias() = kp.R + kp.D.transpose() * tempD;
        kp.H_bar.noalias() = kp.H + kp.D.transpose() * tempC;
        
        kp.q_bar.noalias() = kp.q + kp.C.transpose() * grad_mod;
        kp.r_bar.noalias() = kp.r + kp.D.transpose() * grad_mod;
        
        #else
        // Custom Matrix Library with Diagonal Support
        MSDiag<double, Knot::NC> SigmaMat(sigma);
        
        MSMat<double, Knot::NC, Knot::NX> tempC = SigmaMat * kp.C;
        MSMat<double, Knot::NC, Knot::NU> tempD = SigmaMat * kp.D;

        kp.Q_bar.noalias() = kp.Q + kp.C.transpose() * tempC;
        kp.R_bar.noalias() = kp.R + kp.D.transpose() * tempD;
        kp.H_bar.noalias() = kp.H + kp.D.transpose() * tempC;
        
        kp.q_bar.noalias() = kp.q + kp.C.transpose() * grad_mod;
        kp.r_bar.noalias() = kp.r + kp.D.transpose() * grad_mod;
        #endif
    }

    template<typename Knot>
    void recover_dual_search_directions(Knot& kp, double mu, const minisolver::SolverConfig& config) {
        MSVec<double, Knot::NC> r_prim = kp.g_val + kp.s;
        MSVec<double, Knot::NC> constraint_step = kp.C * kp.dx + kp.D * kp.du;

        kp.ds = -r_prim - constraint_step;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            if(s_i < config.min_barrier_slack) s_i = config.min_barrier_slack; // Use Config
            double r_cent_i = (s_i * kp.lam(i)) - mu;
            kp.dlam(i) = -(1.0 / s_i) * (r_cent_i + kp.lam(i) * kp.ds(i));
        }
    }

    template<typename MatrixType>
    bool fast_inverse(const MatrixType& A, MatrixType& A_inv) {
        // Safe dimension check compatible with C++17
        constexpr int ROWS = MatrixType::RowsAtCompileTime; 

        if constexpr (ROWS == 1) { 
            if (std::abs(A(0,0)) < 1e-9) return false;
            A_inv(0,0) = 1.0 / A(0,0);
            return true;
        } 
        else if constexpr (ROWS == 2) {
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
            return MatOps::cholesky_solve(A, MatrixType::Identity(), A_inv);
        }
    }

    template<typename TrajVector>
    bool cpu_serial_solve(TrajVector& traj, int N, double mu, double reg, minisolver::InertiaStrategy strategy, 
                          const minisolver::SolverConfig& config = minisolver::SolverConfig(),
                          const TrajVector* affine_traj = nullptr) {
        using Knot = typename TrajVector::value_type;

        for(int k=0; k<=N; ++k) {
            const Knot* aff_kp = (affine_traj) ? &((*affine_traj)[k]) : nullptr;
            compute_barrier_derivatives(traj[k], mu, config, aff_kp); 
        }

        MSVec<double, Knot::NX> Vx = traj[N].q_bar;
        MSMat<double, Knot::NX, Knot::NX> Vxx = traj[N].Q_bar;
        
        for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;

        for(int k = N - 1; k >= 0; --k) {
            auto& kp = traj[k];

            MSMat<double, Knot::NX, Knot::NX> VxxA;
            VxxA.noalias() = Vxx * kp.A;
            
            MSMat<double, Knot::NX, Knot::NU> VxxB;
            VxxB.noalias() = Vxx * kp.B; 

            if (config.enable_defect_correction) {
                MSVec<double, Knot::NX> defect = kp.f_resid - traj[k+1].x;
                MSVec<double, Knot::NX> Vxx_d = Vxx * defect;

                kp.q_bar.noalias() += kp.A.transpose() * Vx;
                kp.q_bar.noalias() += kp.A.transpose() * Vxx_d; 
                
                kp.r_bar.noalias() += kp.B.transpose() * Vx;
                kp.r_bar.noalias() += kp.B.transpose() * Vxx_d; 
            } else {
                kp.q_bar.noalias() += kp.A.transpose() * Vx;
                kp.r_bar.noalias() += kp.B.transpose() * Vx;
            }
            
            MSVec<double, Knot::NX> Qx = kp.q_bar; // Local copy for clarity
            MSVec<double, Knot::NU> Qu = kp.r_bar; // Local copy for clarity

            MSMat<double, Knot::NX, Knot::NX> Qxx = kp.Q_bar;
            Qxx.noalias() += kp.A.transpose() * VxxA;
            
            MSMat<double, Knot::NU, Knot::NU> Quu = kp.R_bar;
            Quu.noalias() += kp.B.transpose() * VxxB;
            
            MSMat<double, Knot::NU, Knot::NX> Qux = kp.H_bar;
            Qux.noalias() += kp.B.transpose() * VxxA;

            if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += reg;
            } else {
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += config.reg_min; 
            }

            if (strategy == minisolver::InertiaStrategy::REGULARIZATION && Knot::NU <= 2) {
                MSMat<double, Knot::NU, Knot::NU> Quu_inv;
                if (!fast_inverse(Quu, Quu_inv)) return false;
                kp.d.noalias() = -Quu_inv * Qu;
                kp.K.noalias() = -Quu_inv * Qux;
            } 
            else {
                if(!MatOps::is_pos_def(Quu)) {
                     if (strategy == minisolver::InertiaStrategy::REGULARIZATION) return false;
                     if (strategy == minisolver::InertiaStrategy::IGNORE_SINGULAR) {
                         bool fixed = false;
                         for(int i=0; i<Knot::NU; ++i) {
                             if (Quu(i,i) < config.singular_threshold) { 
                                 Quu(i,i) += config.huge_penalty;
                                 fixed = true;
                             }
                         }
                         if (fixed && !MatOps::is_pos_def(Quu)) return false; 
                     }
                }
                
                MSVec<double, Knot::NU> neg_Qu = -Qu;
                if(!MatOps::cholesky_solve(Quu, neg_Qu, kp.d)) return false;
                
                MSMat<double, Knot::NU, Knot::NX> neg_Qux = -Qux;
                if(!MatOps::cholesky_solve(Quu, neg_Qux, kp.K)) return false;
            }

            Vx = Qx;
            Vx.noalias() += Qux.transpose() * kp.d;

            Vxx = Qxx;
            Vxx.noalias() += Qux.transpose() * kp.K;
            
            Vxx = 0.5 * (Vxx + Vxx.transpose());
            for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;
        }

        traj[0].dx.setZero(); 

        for(int k=0; k < N; ++k) {
            auto& kp = traj[k];
            kp.du.noalias() = kp.K * kp.dx + kp.d;
            MSVec<double, Knot::NX> defect = kp.f_resid - traj[k+1].x;
            traj[k+1].dx.noalias() = kp.A * kp.dx;
            traj[k+1].dx.noalias() += kp.B * kp.du;
            traj[k+1].dx += defect;
            
            recover_dual_search_directions(kp, mu, config); 
        }
        recover_dual_search_directions(traj[N], mu, config); 
        
        return true;
    }
}
