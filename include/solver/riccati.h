#pragma once
#include <vector>
#include <Eigen/Dense>
#include "core/types.h"
#include "core/solver_options.h" // Need InertiaStrategy enum

namespace roboopt {

    // Helper: Compute Barrier-Modified Q, R, q, r
    template<typename Knot>
    void compute_barrier_derivatives(Knot& kp, double mu) {
        Eigen::Matrix<double, Knot::NC, 1> sigma;
        Eigen::Matrix<double, Knot::NC, 1> grad_mod;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            double lam_i = kp.lam(i);
            
            // Safety against small s
            if(s_i < 1e-12) s_i = 1e-12; 

            sigma(i) = lam_i / s_i;
            
            double r_prim_i = kp.g_val(i) + s_i;
            double r_cent_i = (s_i * lam_i) - mu;
            
            grad_mod(i) = (1.0 / s_i) * (lam_i * r_prim_i - r_cent_i);
        }

        Eigen::DiagonalMatrix<double, Knot::NC> SigmaMat(sigma);

        kp.Q_bar = kp.Q + kp.C.transpose() * SigmaMat * kp.C;
        kp.R_bar = kp.R + kp.D.transpose() * SigmaMat * kp.D;
        kp.H_bar = kp.H + kp.D.transpose() * SigmaMat * kp.C;
        kp.q_bar = kp.q + kp.C.transpose() * grad_mod;
        kp.r_bar = kp.r + kp.D.transpose() * grad_mod;
    }

    template<typename Knot>
    void recover_dual_search_directions(Knot& kp, double mu) {
        Eigen::Matrix<double, Knot::NC, 1> r_prim = kp.g_val + kp.s;
        Eigen::Matrix<double, Knot::NC, 1> constraint_step = kp.C * kp.dx + kp.D * kp.du;

        kp.ds = -r_prim - constraint_step;

        for(int i=0; i<Knot::NC; ++i) {
            double s_i = kp.s(i);
            if(s_i < 1e-12) s_i = 1e-12;
            double r_cent_i = (s_i * kp.lam(i)) - mu;
            kp.dlam(i) = -(1.0 / s_i) * (r_cent_i + kp.lam(i) * kp.ds(i));
        }
    }

    // UPDATED: Multi-Strategy Inertia Correction
    template<typename Knot>
    bool cpu_serial_solve(std::vector<Knot>& traj, double mu, double reg, InertiaStrategy strategy) {
        int N = traj.size() - 1; 

        for(auto& kp : traj) {
            compute_barrier_derivatives(kp, mu);
        }

        Eigen::Matrix<double, Knot::NX, 1> Vx = traj[N].q_bar;
        Eigen::Matrix<double, Knot::NX, Knot::NX> Vxx = traj[N].Q_bar;
        
        for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;

        for(int k = N - 1; k >= 0; --k) {
            auto& kp = traj[k];

            Eigen::Matrix<double, Knot::NX, 1> Qx = kp.q_bar + kp.A.transpose() * Vx;
            Eigen::Matrix<double, Knot::NU, 1> Qu = kp.r_bar + kp.B.transpose() * Vx;
            Eigen::Matrix<double, Knot::NX, Knot::NX> Qxx = kp.Q_bar + kp.A.transpose() * Vxx * kp.A;
            Eigen::Matrix<double, Knot::NU, Knot::NU> Quu = kp.R_bar + kp.B.transpose() * Vxx * kp.B;
            Eigen::Matrix<double, Knot::NU, Knot::NX> Qux = kp.H_bar + kp.B.transpose() * Vxx * kp.A;

            // Strategy 1: Regularization (Always add reg)
            if (strategy == InertiaStrategy::REGULARIZATION) {
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += reg;
            } else {
                // For other strategies, we might only add tiny numerical stability term
                for(int i=0; i<Knot::NU; ++i) Quu(i,i) += 1e-9;
            }

            // Attempt Cholesky
            Eigen::LLT<Eigen::Matrix<double, Knot::NU, Knot::NU>> llt(Quu);
            
            if(llt.info() == Eigen::NumericalIssue) {
                // Cholesky Failed (Not PD)
                
                if (strategy == InertiaStrategy::REGULARIZATION) {
                    // Just fail, let outer loop increase reg
                    return false; 
                }
                else if (strategy == InertiaStrategy::IGNORE_SINGULAR) {
                    // "Freeze" strategy: heavily penalize near-zero directions
                    bool fixed = false;
                    // Check diagonal relative to trace or max val?
                    // Simple check: absolute threshold
                    for(int i=0; i<Knot::NU; ++i) {
                        if (Quu(i,i) < 1e-4) { // Threshold
                            Quu(i,i) += 1e9; // Freeze
                            fixed = true;
                        }
                    }
                    if (fixed) llt.compute(Quu);
                    if(llt.info() == Eigen::NumericalIssue) return false; // Still failed (e.g. off-diagonal)
                }
                else if (strategy == InertiaStrategy::SATURATION) {
                    // Force eigenvalues up.
                    // Instead of full eigendecomposition, just add enough to diagonal locally.
                    // This is similar to Reg, but we do it *per matrix* inside the loop,
                    // effectively finding minimal reg for this specific step.
                    // Simple implementation: Iterative local regularization
                    double local_reg = 1e-4;
                    for(int iter=0; iter<10; ++iter) {
                        // Create temp copy
                        auto Quu_mod = Quu;
                        for(int i=0; i<Knot::NU; ++i) Quu_mod(i,i) += local_reg;
                        llt.compute(Quu_mod);
                        if(llt.info() != Eigen::NumericalIssue) break;
                        local_reg *= 10.0;
                    }
                    if(llt.info() == Eigen::NumericalIssue) return false;
                }
            }

            Eigen::Matrix<double, Knot::NU, 1> d = -llt.solve(Qu);
            Eigen::Matrix<double, Knot::NU, Knot::NX> K = -llt.solve(Qux);

            kp.d = d;
            kp.K = K;

            Vx = Qx + K.transpose() * Quu * d + K.transpose() * Qu + Qux.transpose() * d;
            Vxx = Qxx + K.transpose() * Quu * K + K.transpose() * Qux + Qux.transpose() * K;
            Vxx = 0.5 * (Vxx + Vxx.transpose());
            
            for(int i=0; i<Knot::NX; ++i) Vxx(i,i) += reg;
        }

        traj[0].dx.setZero(); 

        for(int k=0; k < N; ++k) {
            auto& kp = traj[k];
            kp.du = kp.K * kp.dx + kp.d;
            Eigen::Matrix<double, Knot::NX, 1> defect = kp.f_resid - traj[k+1].x;
            traj[k+1].dx = kp.A * kp.dx + kp.B * kp.du + defect;
            recover_dual_search_directions(kp, mu);
        }
        recover_dual_search_directions(traj[N], mu);
        
        return true;
    }
}
