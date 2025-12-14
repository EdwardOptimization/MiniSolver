#pragma once
#include <vector>
#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif
#include "minisolver/core/types.h"
#include "minisolver/core/solver_options.h" 
#include "minisolver/core/matrix_defs.h"

namespace minisolver {

template<typename Knot>
struct RiccatiWorkspace {
    // --- For compute_barrier_derivatives ---
    MSDiag<double, Knot::NC> sigma_mat;
    MSMat<double, Knot::NC, Knot::NX> temp_C;
    MSMat<double, Knot::NC, Knot::NU> temp_D;
    
    // --- For backward_pass (Riccati Step) ---
    MSMat<double, Knot::NX, Knot::NX> VxxA;
    MSMat<double, Knot::NX, Knot::NU> VxxB;
    MSVec<double, Knot::NX> Vxx_d;
    MSMat<double, Knot::NU, Knot::NU> Quu_inv; // For fast_inverse case

    // --- Linear Solver ---
    MSLLT<MSMat<double, Knot::NU, Knot::NU>> llt_solver;

    RiccatiWorkspace() {
        // Pre-allocate or initialize if needed
    }
};

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
    void compute_barrier_derivatives(Knot& kp, double mu, const minisolver::SolverConfig& config, RiccatiWorkspace<Knot>& ws, const Knot* aff_kp = nullptr, const Knot* soc_kp = nullptr) {
        (void)ws; // Suppress unused parameter warning
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

        MSDiag<double, Knot::NC> SigmaMat(sigma);
        
        MSMat<double, Knot::NC, Knot::NX> tempC = SigmaMat * kp.C;
        MSMat<double, Knot::NC, Knot::NU> tempD = SigmaMat * kp.D;

        kp.Q_bar.noalias() = kp.Q + kp.C.transpose() * tempC;
        kp.R_bar.noalias() = kp.R + kp.D.transpose() * tempD;
        kp.H_bar.noalias() = kp.H + kp.D.transpose() * tempC;
        
        kp.q_bar.noalias() = kp.q + kp.C.transpose() * grad_mod;
        kp.r_bar.noalias() = kp.r + kp.D.transpose() * grad_mod;
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
    bool fast_inverse(const MatrixType& A, MatrixType& A_inv, double epsilon = 1e-9) {
        // Safe dimension check compatible with C++17
        constexpr int ROWS = MatrixType::RowsAtCompileTime; 

        // --- Case 1: 1x1 Matrix ---
        if constexpr (ROWS == 1) { 
            double val = A(0,0);
            if (std::abs(val) < epsilon) return false;
            A_inv(0,0) = 1.0 / val;
            return true;
        } 
        // --- Case 2: 2x2 Matrix ---
        else if constexpr (ROWS == 2) {
            double det = A(0,0)*A(1,1) - A(0,1)*A(1,0);
            if (std::abs(det) < epsilon) return false;
            
            double inv_det = 1.0 / det;
            A_inv(0,0) =  A(1,1) * inv_det;
            A_inv(0,1) = -A(0,1) * inv_det;
            A_inv(1,0) = -A(1,0) * inv_det;
            A_inv(1,1) =  A(0,0) * inv_det;
            return true;
        }
        // --- Case 3: 3x3 Matrix (Optional, high performance) ---
        else if constexpr (ROWS == 3) {
            // Sarrus Rule or Expansion by minors
            double A00 = A(0,0), A01 = A(0,1), A02 = A(0,2);
            double A10 = A(1,0), A11 = A(1,1), A12 = A(1,2);
            double A20 = A(2,0), A21 = A(2,1), A22 = A(2,2);

            double det = A00*(A11*A22 - A12*A21) -
                         A01*(A10*A22 - A12*A20) +
                         A02*(A10*A21 - A11*A20);
                         
            if (std::abs(det) < epsilon) return false;
            double inv_det = 1.0 / det;

            A_inv(0,0) = (A11*A22 - A12*A21) * inv_det;
            A_inv(0,1) = (A02*A21 - A01*A22) * inv_det;
            A_inv(0,2) = (A01*A12 - A02*A11) * inv_det;

            A_inv(1,0) = (A12*A20 - A10*A22) * inv_det;
            A_inv(1,1) = (A00*A22 - A02*A20) * inv_det;
            A_inv(1,2) = (A02*A10 - A00*A12) * inv_det;

            A_inv(2,0) = (A10*A21 - A11*A20) * inv_det;
            A_inv(2,1) = (A01*A20 - A00*A21) * inv_det;
            A_inv(2,2) = (A00*A11 - A01*A10) * inv_det;
            return true;
        }
        // --- Case 4: General N > 3 (Fallback) ---
        else {
            return MatOps::cholesky_solve(A, MatrixType::Identity(), A_inv);
        }
    }

    template<typename TrajectoryType, typename ModelType>
    bool cpu_serial_solve(TrajectoryType& traj, int N, double mu, double reg, minisolver::InertiaStrategy strategy, 
                          const minisolver::SolverConfig& config,
                          RiccatiWorkspace<typename TrajectoryType::KnotType>& ws,
                          const TrajectoryType* affine_traj = nullptr,
                          const TrajectoryType* soc_traj = nullptr) { // [NEW] Added arg
        using Knot = typename TrajectoryType::KnotType;

        // Get pointers to the three layers
        auto* state = traj.get_active_state();
        auto* model = traj.get_model_data();
        auto* workspace = traj.get_workspace();

        for(int k=0; k<=N; ++k) {
            // TODO: Update compute_barrier_derivatives to use split architecture
            // For now, create temporary KnotPoint-like access
            // const Knot* aff_kp = (affine_traj) ? &((*affine_traj)[k]) : nullptr;
            // const Knot* soc_kp = (soc_traj) ? &((*soc_traj)[k]) : nullptr;
            // compute_barrier_derivatives<Knot, ModelType>(traj[k], mu, config, ws, aff_kp, soc_kp);
            
            // SIMPLIFIED: Just assemble KKT without barrier derivatives for now
            // Will implement proper barrier derivatives later
        }

        MSVec<double, Knot::NX> Vx = workspace[N].q_bar;
        MSMat<double, Knot::NX, Knot::NX> Vxx = workspace[N].Q_bar;
        
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
                        defect = model[k].f_resid - traj[k+1].x;
                    }
                    
                    // Vxx * defect
                    MSVec<double, Knot::NX> Vxx_d = Vxx * defect;
                    
                    // Add A^T * Vxx_d and B^T * Vxx_d to gradients
                    MatOps::mult_add_transA_v(workspace[k].q_bar, model[k].A, Vxx_d);
                    MatOps::mult_add_transA_v(workspace[k].r_bar, model[k].B, Vxx_d);
                }
            }
            else {
                // [LEGACY / SEPARATE KERNEL PATH]
                // In-place updates for Riccati backward pass
                // Use workspace[k].Q_bar as accumulator for Qxx
                // Use workspace[k].R_bar as accumulator for Quu
                // Use workspace[k].H_bar as accumulator for Qux
                
                // [OPTIMIZED] Use Workspace for temporary matrices
                ws.VxxA.noalias() = Vxx * model[k].A;
                ws.VxxB.noalias() = Vxx * model[k].B; 

                if (config.enable_defect_correction) {
                    // Defect correction
                    MSVec<double, Knot::NX> defect;
                    if (soc_traj) {
                        defect = (*soc_traj)[k].f_resid - (*soc_traj)[k+1].x;
                    } else {
                        defect = model[k].f_resid - traj[k+1].x;
                    }
                    
                    ws.Vxx_d = Vxx * defect;

                    // Update Gradient Qx (workspace[k].q_bar) and Qu (workspace[k].r_bar) in-place
                    // workspace[k].q_bar += A^T * Vx
                    MatOps::mult_add_transA_v(workspace[k].q_bar, model[k].A, Vx);
                    MatOps::mult_add_transA_v(workspace[k].q_bar, model[k].A, ws.Vxx_d);

                    MatOps::mult_add_transA_v(workspace[k].r_bar, model[k].B, Vx);
                    MatOps::mult_add_transA_v(workspace[k].r_bar, model[k].B, ws.Vxx_d);
                } else {
                    // Update Gradient Qx (workspace[k].q_bar) and Qu (workspace[k].r_bar) in-place
                    MatOps::mult_add_transA_v(workspace[k].q_bar, model[k].A, Vx);
                    MatOps::mult_add_transA_v(workspace[k].r_bar, model[k].B, Vx);
                }
                
                // Update Hessian Qxx (workspace[k].Q_bar), Quu (workspace[k].R_bar), Qux (workspace[k].H_bar) in-place
                MatOps::mult_add_transA(workspace[k].Q_bar, model[k].A, ws.VxxA);
                MatOps::mult_add_transA(workspace[k].R_bar, model[k].B, ws.VxxB);
                MatOps::mult_add_transA(workspace[k].H_bar, model[k].B, ws.VxxA);
            } // End of Legacy Path

            if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
                for(int i=0; i<Knot::NU; ++i) workspace[k].R_bar(i,i) += reg;
            } else {
                for(int i=0; i<Knot::NU; ++i) workspace[k].R_bar(i,i) += config.reg_min; 
            }

            if (strategy == minisolver::InertiaStrategy::REGULARIZATION && Knot::NU <= 3) {
                if (!fast_inverse(workspace[k].R_bar, ws.Quu_inv, config.singular_threshold)) return false;
                
                workspace[k].d.noalias() = -ws.Quu_inv * workspace[k].r_bar; // Qu is in workspace[k].r_bar
                workspace[k].K.noalias() = -ws.Quu_inv * workspace[k].H_bar; // Qux is in workspace[k].H_bar
            }
            else {
                // [General Path] "Factorize Once, Solve Twice"
                
                // 1. Try to factorize the matrix
                // Use the solver in workspace, do not call is_pos_def
                ws.llt_solver.compute(kp.R_bar); 
                
                // 2. Check the factorization result
                if (!MatOps::is_llt_success(ws.llt_solver)) {
                     
                     // Failure handling A: Directly return false
                     if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
                        // Try the last chance: unified regularization
                        for(int i=0; i<Knot::NU; ++i) kp.R_bar(i,i) += config.regularization_step; // e.g. 1e-6
                        
                        ws.llt_solver.compute(kp.R_bar);
                        if (!MatOps::is_llt_success(ws.llt_solver)) return false; // Give up
                    }
                     
                     // Failure handling B: Try to fix (Ignore Singular)
                     if (strategy == minisolver::InertiaStrategy::IGNORE_SINGULAR) {
                         bool fixed = false;
                         for(int i=0; i<Knot::NU; ++i) {
                             if (kp.R_bar(i,i) < config.singular_threshold) { 
                                 kp.R_bar(i,i) += config.huge_penalty;
                                 fixed = true;
                             }
                         }
                         
                         // If no element is less than the threshold but the factorization still fails (non-diagonal dominant等情况),
                         // or the corrected matrix needs to be re-calculated:
                         if (fixed) {
                             // Re-factorize the corrected matrix (Retry Factorization)
                             ws.llt_solver.compute(kp.R_bar);
                             if (!MatOps::is_llt_success(ws.llt_solver)) return false;
                         } else {
                             // The matrix is not positive definite, but the diagonal elements are all greater than the threshold, 
                             // indicating a structural problem that cannot be simply fixed
                             return false; 
                         }
                     }
                }
                
                // Here llt_solver has stored the successful factorization result L
                // Now directly solve, no need to compute again

                // 3. Solve Quu * d = -Qu
                kp.d = -kp.r_bar;
                MatOps::solve_llt_inplace(ws.llt_solver, kp.d);
                
                // 4. Solve Quu * K = -Qux
                kp.K = -kp.H_bar;
                MatOps::solve_llt_inplace(ws.llt_solver, kp.K);
            }

            Vx = kp.q_bar;
            MatOps::mult_add_transA_v(Vx, kp.H_bar, kp.d);

            Vxx = kp.Q_bar;
            MatOps::mult_add_transA(Vxx, kp.H_bar, kp.K);
            MatOps::symmetrize(Vxx);
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
            MatOps::mult_add(traj[k+1].dx, kp.B, kp.du);
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
