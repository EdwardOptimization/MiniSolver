#pragma once
#include <vector>
#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h" 
#include "minisolver/core/matrix_defs.h"

namespace minisolver {

namespace internal {
    // SFINAE Helper: Detect if Model has compute_fused_riccati_step<double> static method
    template <typename T, typename = void>
    struct has_fused_riccati_step : std::false_type {};

    template <typename T>
    struct has_fused_riccati_step<T, std::void_t<decltype(&T::template compute_fused_riccati_step<double>)>> : std::true_type {};
}

    template<typename Knot, typename ModelType>
    void compute_kkt_residual(Knot& kp, double mu, const minisolver::SolverConfig& config,
                              MSVec<double, Knot::NX>& r_Lx,
                              MSVec<double, Knot::NU>& r_Lu,
                              const MSVec<double, Knot::NX>& lam_x_next) {
        
        // KKT Stationarity:
        // Lx = q + A' lam_x_next + C' lam_c (where lam_c is modified barrier dual)
        // Lu = r + B' lam_x_next + D' lam_c
        
        // Use current lambda from knot point
        MSVec<double, Knot::NC> lam_total = kp.lam;
        
        // Standard KKT residual calculation
        if constexpr (Knot::NC > 0) {
             r_Lx = kp.q + kp.A.transpose() * lam_x_next + kp.C.transpose() * lam_total;
             r_Lu = kp.r + kp.B.transpose() * lam_x_next + kp.D.transpose() * lam_total;
        } else {
             r_Lx = kp.q + kp.A.transpose() * lam_x_next;
             r_Lu = kp.r + kp.B.transpose() * lam_x_next;
        }
    }

    template<typename Knot, typename ModelType>
    void compute_barrier_derivatives(Knot& kp, double mu, const minisolver::SolverConfig& config, const Knot* aff_kp = nullptr, const Knot* soc_kp = nullptr) {
        MSVec<double, Knot::NC> sigma;
        MSVec<double, Knot::NC> grad_mod;

        // SOC Residual Override
        // If soc_kp is provided, it contains the candidate residuals g(x_cand).
        // Standard IPM linearizes around x_k: g(x_k) + J dx = -residual.
        // Usually residual = 0 (feasibility).
        // In perturbed KKT (barrier), we target g + s = 0.
        // r_prim = g_val + s.
        // For SOC, we want to correct the second order error.
        // g(x_k + dx) approx g(x_k) + J dx + 0.5 dx' H dx
        // We want g(x_k + dx + dx_soc) = 0
        // g(x_k + dx) + J dx_soc = 0
        // J dx_soc = - g(x_k + dx).
        // So we want the linear system J dx_soc = -r_soc.
        // This means we should replace `g_val` in r_prim with `soc_kp->g_val`.
        // Note: s is from base point x_k? Yes, we linearize at x_k.

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            double lam_i = kp.lam(i);
            
            if(s_i < config.min_barrier_slack) s_i = config.min_barrier_slack;

            double w = 0.0;
            int type = 0;
            if constexpr (Knot::NC > 0) {
                 if (static_cast<size_t>(i) < ModelType::constraint_types.size()) {
                    type = ModelType::constraint_types[i];
                    w = ModelType::constraint_weights[i];
                 }
            }

            double sigma_val = lam_i / s_i;
            
            if (type == 2 && w > 1e-6) { // L2 Soft (Dual Reg)
                sigma_val = 1.0 / (s_i/lam_i + 1.0/w);
            }
            else if (type == 1 && w > 1e-6) { // L1 Soft (Dual Box)
                double soft_s_i = kp.soft_s(i); 
                if (soft_s_i < config.min_barrier_slack) soft_s_i = config.min_barrier_slack;
                
                double term_hard = s_i / lam_i;
                double term_soft = soft_s_i / (w - lam_i); 
                
                sigma_val = 1.0 / (term_hard + term_soft);
            }
            
            sigma(i) = sigma_val;
            
            double r_y = s_i * lam_i - mu;
            if (aff_kp) r_y += aff_kp->ds(i) * aff_kp->dlam(i);
            
            // Primal residual base
            double g_val_i = (soc_kp) ? soc_kp->g_val(i) : kp.g_val(i);
            
            if (type == 1 && w > 1e-6) {
                double soft_s_i = kp.soft_s(i); 
                if (soft_s_i < config.min_barrier_slack) soft_s_i = config.min_barrier_slack;

                double r_eq = g_val_i + s_i - soft_s_i;
                double r_z = soft_s_i * (w - lam_i) - mu;
                if (aff_kp) {
                    double dsoft_s_i = aff_kp->dsoft_s(i);
                    double dlam_aff_i = aff_kp->dlam(i);
                    r_z += dsoft_s_i * (-dlam_aff_i);
                }
                
                // Corrected Signs:
                // grad_mod = lam + sigma * (r_eq - r_y/lam + r_z/(w-lam))
                double term_correction = r_eq - r_y/lam_i + r_z/(w - lam_i);
                grad_mod(i) = lam_i + sigma_val * term_correction;
            }
            else {
                // Standard / L2
                double term2;
                if (type == 2 && w > 1e-6) {
                    double r_prim_L2 = g_val_i + s_i - lam_i/w;
                    term2 = sigma_val * (r_y / lam_i);
                    grad_mod(i) = sigma_val * r_prim_L2 - term2 + lam_i;
                } else {
                    double r_eq = g_val_i + s_i;
                    term2 = r_y / s_i;
                    grad_mod(i) = sigma_val * r_eq - term2 + lam_i;
                }
            }
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

    template<typename Knot, typename ModelType>
    void recover_dual_search_directions(Knot& kp, double mu, const minisolver::SolverConfig& config, const Knot* soc_kp = nullptr, const Knot* aff_kp = nullptr) {
        // Use soc_kp->g_val if available
        
        MSVec<double, Knot::NC> constraint_step = kp.C * kp.dx + kp.D * kp.du;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            if(s_i < config.min_barrier_slack) s_i = config.min_barrier_slack; 
            
            double w = 0.0;
            int type = 0;
            if constexpr (Knot::NC > 0) {
                 if (static_cast<size_t>(i) < ModelType::constraint_types.size()) {
                    type = ModelType::constraint_types[i];
                    w = ModelType::constraint_weights[i];
                 }
            }
            
            double lam_i = kp.lam(i);
            double r_y = s_i * lam_i - mu; 
            if (aff_kp) r_y += aff_kp->ds(i) * aff_kp->dlam(i);
            
            double g_val_i = (soc_kp) ? soc_kp->g_val(i) : kp.g_val(i);

            if (type == 1 && w > 1e-6) { // L1 Soft
                double soft_s_i = kp.soft_s(i);
                if (soft_s_i < config.min_barrier_slack) soft_s_i = config.min_barrier_slack;
                
                double term_hard = s_i / lam_i;
                double term_soft = soft_s_i / (w - lam_i);
                double sigma_val = 1.0 / (term_hard + term_soft);
                
                double r_eq = g_val_i + s_i - soft_s_i;
                double r_z = soft_s_i * (w - lam_i) - mu;
                if (aff_kp) {
                    r_z += aff_kp->dsoft_s(i) * (-aff_kp->dlam(i));
                }
                
                // Corrected Signs for dlam recovery
                // dlam = sigma * (C dx + r_eq - r_y/lam + r_z/(w-lam))
                double eff_r = r_eq - r_y/lam_i + r_z/(w - lam_i);
                
                double dlam = sigma_val * (constraint_step(i) + eff_r);
                kp.dlam(i) = dlam;
                
                kp.ds(i) = (-r_y - s_i * dlam) / lam_i; 
                kp.dsoft_s(i) = -(r_z - soft_s_i * dlam) / (w - lam_i); 
            }
            else if (type == 2 && w > 1e-6) { // L2 Soft
                double r_prim_L2 = g_val_i + s_i - lam_i/w;
                double term_rhs = -r_y + lam_i * (r_prim_L2 + constraint_step(i));
                double factor = 1.0 / (s_i + lam_i/w);
                
                kp.dlam(i) = factor * term_rhs;
                kp.ds(i) = -r_prim_L2 - constraint_step(i) + kp.dlam(i)/w;
            }
            else { // Hard
                double r_prim = g_val_i + s_i;
                double term_rhs = -r_y + lam_i * (r_prim + constraint_step(i));
                
                kp.dlam(i) = (1.0 / s_i) * term_rhs;
                kp.ds(i) = -r_prim - constraint_step(i);
            }
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

    template<typename TrajVector, typename ModelType>
    bool cpu_serial_solve(TrajVector& traj, int N, double mu, double reg, minisolver::InertiaStrategy strategy, 
                          const minisolver::SolverConfig& config = minisolver::SolverConfig(),
                          const TrajVector* affine_traj = nullptr,
                          const TrajVector* soc_traj = nullptr) { // [NEW] Added arg
        using Knot = typename TrajVector::value_type;

        for(int k=0; k<=N; ++k) {
            const Knot* aff_kp = (affine_traj) ? &((*affine_traj)[k]) : nullptr;
            const Knot* soc_kp = (soc_traj) ? &((*soc_traj)[k]) : nullptr;
            compute_barrier_derivatives<Knot, ModelType>(traj[k], mu, config, aff_kp, soc_kp); 
        }

        MSVec<double, Knot::NX> Vx = traj[N].q_bar;
        MSMat<double, Knot::NX, Knot::NX> Vxx = traj[N].Q_bar;
        
        for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;

        for(int k = N - 1; k >= 0; --k) {
            auto& kp = traj[k];

            // [FUSED KERNEL OPTIMIZATION]
            if constexpr (internal::has_fused_riccati_step<ModelType>::value) {
                // One-shot update of Qxx, Quu, Qux, qx, ru using fused kernel
                ModelType::compute_fused_riccati_step(Vxx, Vx, kp);
                
                // If defect correction is enabled, apply the additional defect term
                if (config.enable_defect_correction) {
                    MSVec<double, Knot::NX> defect;
                    if (soc_traj) {
                        defect = (*soc_traj)[k].f_resid - (*soc_traj)[k+1].x;
                    } else {
                        defect = kp.f_resid - traj[k+1].x;
                    }
                    
                    // Vxx * defect
                    MSVec<double, Knot::NX> Vxx_d = Vxx * defect;
                    
                    // Add A^T * Vxx_d and B^T * Vxx_d to gradients
#ifdef USE_EIGEN
                    kp.q_bar.noalias() += kp.A.transpose() * Vxx_d; 
                    kp.r_bar.noalias() += kp.B.transpose() * Vxx_d; 
#else
                    kp.q_bar.add_At_mul_v(kp.A, Vxx_d);
                    kp.r_bar.add_At_mul_v(kp.B, Vxx_d);
#endif
                }
            }
            else {
                // [LEGACY / SEPARATE KERNEL PATH]
                // In-place updates for Riccati backward pass
                // Use kp.Q_bar as accumulator for Qxx
                // Use kp.R_bar as accumulator for Quu
                // Use kp.H_bar as accumulator for Qux
                
                MSMat<double, Knot::NX, Knot::NX> VxxA;
                VxxA.noalias() = Vxx * kp.A;
                
                MSMat<double, Knot::NX, Knot::NU> VxxB;
                VxxB.noalias() = Vxx * kp.B; 

                if (config.enable_defect_correction) {
                    // Defect correction
                    MSVec<double, Knot::NX> defect;
                    if (soc_traj) {
                        defect = (*soc_traj)[k].f_resid - (*soc_traj)[k+1].x;
                    } else {
                        defect = kp.f_resid - traj[k+1].x;
                    }
                    
                    MSVec<double, Knot::NX> Vxx_d = Vxx * defect;

                    // Update Gradient Qx (kp.q_bar) and Qu (kp.r_bar) in-place
                    // kp.q_bar += A^T * Vx
    #ifdef USE_EIGEN
                    kp.q_bar.noalias() += kp.A.transpose() * Vx;
                    kp.q_bar.noalias() += kp.A.transpose() * Vxx_d; 
                    
                    kp.r_bar.noalias() += kp.B.transpose() * Vx;
                    kp.r_bar.noalias() += kp.B.transpose() * Vxx_d; 
    #else
                    kp.q_bar.add_At_mul_v(kp.A, Vx);
                    kp.q_bar.add_At_mul_v(kp.A, Vxx_d);
                    
                    kp.r_bar.add_At_mul_v(kp.B, Vx);
                    kp.r_bar.add_At_mul_v(kp.B, Vxx_d);
    #endif
                } else {
                    // Update Gradient Qx (kp.q_bar) and Qu (kp.r_bar) in-place
    #ifdef USE_EIGEN
                    kp.q_bar.noalias() += kp.A.transpose() * Vx;
                    kp.r_bar.noalias() += kp.B.transpose() * Vx;
    #else
                    kp.q_bar.add_At_mul_v(kp.A, Vx);
                    kp.r_bar.add_At_mul_v(kp.B, Vx);
    #endif
                }
                
                // Update Hessian Qxx (kp.Q_bar), Quu (kp.R_bar), Qux (kp.H_bar) in-place
    #ifdef USE_EIGEN
                kp.Q_bar.noalias() += kp.A.transpose() * VxxA;
                kp.R_bar.noalias() += kp.B.transpose() * VxxB;
                kp.H_bar.noalias() += kp.B.transpose() * VxxA;
    #else
                kp.Q_bar.add_At_mul_B(kp.A, VxxA);
                kp.R_bar.add_At_mul_B(kp.B, VxxB);
                kp.H_bar.add_At_mul_B(kp.B, VxxA);
    #endif
            } // End of Legacy Path

            if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
                for(int i=0; i<Knot::NU; ++i) kp.R_bar(i,i) += reg;
            } else {
                for(int i=0; i<Knot::NU; ++i) kp.R_bar(i,i) += config.reg_min; 
            }

            if (strategy == minisolver::InertiaStrategy::REGULARIZATION && Knot::NU <= 2) {
                MSMat<double, Knot::NU, Knot::NU> Quu_inv;
                if (!fast_inverse(kp.R_bar, Quu_inv)) return false;
                kp.d.noalias() = -Quu_inv * kp.r_bar; // Qu is in kp.r_bar
                kp.K.noalias() = -Quu_inv * kp.H_bar; // Qux is in kp.H_bar
            } 
            else {
                if(!MatOps::is_pos_def(kp.R_bar)) {
                     if (strategy == minisolver::InertiaStrategy::REGULARIZATION) return false;
                     if (strategy == minisolver::InertiaStrategy::IGNORE_SINGULAR) {
                         bool fixed = false;
                         for(int i=0; i<Knot::NU; ++i) {
                             if (kp.R_bar(i,i) < config.singular_threshold) { 
                                 kp.R_bar(i,i) += config.huge_penalty;
                                 fixed = true;
                             }
                         }
                         if (fixed && !MatOps::is_pos_def(kp.R_bar)) return false; 
                     }
                }
                
                // In-place solve: neg_Qu and neg_Qux logic replacement
                // Solve Quu * d = -Qu  => Quu * d = -kp.r_bar
                // Solve Quu * K = -Qux => Quu * K = -kp.H_bar
                
                kp.d = -kp.r_bar;
                if(!MatOps::cholesky_solve(kp.R_bar, kp.d, kp.d)) return false; // In-place solve supported? Assumed yes for now based on MatOps
                
                kp.K = -kp.H_bar;
                if(!MatOps::cholesky_solve(kp.R_bar, kp.K, kp.K)) return false;
            }

            Vx = kp.q_bar;
#ifdef USE_EIGEN
            Vx.noalias() += kp.H_bar.transpose() * kp.d;
#else
            Vx.add_At_mul_v(kp.H_bar, kp.d);
#endif

            Vxx = kp.Q_bar;
#ifdef USE_EIGEN
            Vxx.noalias() += kp.H_bar.transpose() * kp.K;
            Vxx = 0.5 * (Vxx + Vxx.transpose());
#else
            Vxx.add_At_mul_B(kp.H_bar, kp.K);
            Vxx.symmetrize();
#endif
            for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;
        }

        traj[0].dx.setZero(); 

        for(int k=0; k < N; ++k) {
            auto& kp = traj[k];
            kp.du.noalias() = kp.K * kp.dx + kp.d;
            
            MSVec<double, Knot::NX> defect;
            if (soc_traj) {
                defect = (*soc_traj)[k].f_resid - (*soc_traj)[k+1].x;
            } else {
                defect = kp.f_resid - traj[k+1].x;
            }

            traj[k+1].dx.noalias() = kp.A * kp.dx;
            traj[k+1].dx.noalias() += kp.B * kp.du;
            traj[k+1].dx += defect;
            
            const Knot* soc_kp = (soc_traj) ? &((*soc_traj)[k]) : nullptr;
            const Knot* aff_kp = (affine_traj) ? &((*affine_traj)[k]) : nullptr;
            recover_dual_search_directions<Knot, ModelType>(kp, mu, config, soc_kp, aff_kp); 
        }
        
        const Knot* soc_kp_N = (soc_traj) ? &((*soc_traj)[N]) : nullptr;
        const Knot* aff_kp_N = (affine_traj) ? &((*affine_traj)[N]) : nullptr;
        recover_dual_search_directions<Knot, ModelType>(traj[N], mu, config, soc_kp_N, aff_kp_N); 
        
        return true;
    }
}
