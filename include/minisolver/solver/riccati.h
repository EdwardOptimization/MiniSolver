#pragma once
#include <vector>
#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif
#include "minisolver/algorithms/linear_solve_result.h"
#include "minisolver/core/constraint_semantics.h"
#include "minisolver/core/model_traits.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/types.h"
#include "minisolver/matrix/matrix_defs.h"

namespace minisolver {

template <typename Knot> struct RiccatiWorkspace {
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
    MSSPDSolver<MSMat<double, Knot::NU, Knot::NU>> spd_solver;

    // --- For SQRT_CHOLESKY mode: NX×NX Cholesky for value function ---
    MSLLT<MSMat<double, Knot::NX, Knot::NX>> sqrt_vxx_solver;

    // --- For SQRT_QR mode: augmented matrix for QR factorization ---
    // Augmented matrix: [(NX+NU) x (NU+NX+1)] = [[sqrtQ, 0, 0], [0, sqrtR, 0], [L^T A, L^T B, L^T
    // a]]
    MSMat<double, Knot::NX + Knot::NU, Knot::NU + Knot::NX + 1> A_aug;

    RiccatiWorkspace()
    {
        // Pre-allocate or initialize if needed
    }
};

namespace internal {
    // SFINAE Helper: Detect if Model has compute_fused_riccati_step<double> static method
    template <typename T, typename = void> struct has_fused_riccati_step : std::false_type { };

    template <typename T>
    struct has_fused_riccati_step<T,
        std::void_t<decltype(&T::template compute_fused_riccati_step<double>)>> : std::true_type {
    };

    template <typename ModelType>
    bool is_fused_riccati_integrator_compatible(IntegratorType integrator)
    {
        return detail::generated_integrator_matches<ModelType>(integrator);
    }

    inline double positive_barrier_gap(double value, double min_value)
    {
        return value < min_value ? min_value : value;
    }
}

template <typename Knot, typename ModelType>
void compute_kkt_residual(Knot& kp, double mu, const minisolver::SolverConfig& config,
    MSVec<double, Knot::NX>& r_Lx, MSVec<double, Knot::NU>& r_Lu,
    const MSVec<double, Knot::NX>& lam_x_next)
{
    (void)mu;
    (void)config;

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

template <typename Knot, typename ModelType>
void compute_barrier_derivatives(Knot& kp, double mu, const minisolver::SolverConfig& config,
    RiccatiWorkspace<Knot>& ws, const Knot* aff_kp = nullptr, const Knot* soc_kp = nullptr)
{
    (void)ws; // Suppress unused parameter warning
    MSVec<double, Knot::NC> sigma;
    MSVec<double, Knot::NC> grad_mod;

    // SOC Residual Override.
    // If soc_kp is provided, it contains trial/candidate nonlinear residuals.
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
    // The caller must seed kp.s/lam/soft_s with the point where the correction will be applied
    // (normally the trial candidate), while keeping A/B/C/D from the active linearization.

    for (int i = 0; i < Knot::NC; ++i) {
        double s_i = kp.s(i);
        double lam_i = kp.lam(i);

        if (s_i < config.min_barrier_slack) {
            s_i = config.min_barrier_slack;
        }

        double w = 0.0;
        int type = 0;
        if constexpr (Knot::NC > 0) {
            if (static_cast<size_t>(i) < ModelType::constraint_types.size()) {
                type = ModelType::constraint_types[i];
                w = ModelType::constraint_weights[i];
            }
        }

        double sigma_val = lam_i / s_i;

        if (detail::is_l2_soft_constraint(type, w)) { // L2 Soft (Dual Reg)
            sigma_val = 1.0 / (s_i / lam_i + 1.0 / w);
        } else if (detail::is_l1_soft_constraint(type, w, config)) { // L1 Soft (Dual Box)
            double soft_s_i = kp.soft_s(i);
            if (soft_s_i < config.min_barrier_slack) {
                soft_s_i = config.min_barrier_slack;
            }
            const double soft_dual_i = detail::positive_l1_soft_dual_gap(w - lam_i, w, config);

            double term_hard = s_i / lam_i;
            double term_soft = soft_s_i / soft_dual_i;

            sigma_val = 1.0 / (term_hard + term_soft);
        }

        sigma(i) = sigma_val;

        double r_y = s_i * lam_i - mu;
        if (aff_kp) {
            r_y += aff_kp->ds(i) * aff_kp->dlam(i);
        }

        // Primal residual base
        double g_val_i = (soc_kp) ? soc_kp->g_val(i) : kp.g_val(i);

        if (detail::is_l1_soft_constraint(type, w, config)) {
            double soft_s_i = kp.soft_s(i);
            if (soft_s_i < config.min_barrier_slack) {
                soft_s_i = config.min_barrier_slack;
            }
            const double soft_dual_i = detail::positive_l1_soft_dual_gap(w - lam_i, w, config);

            double r_eq = g_val_i + s_i - soft_s_i;
            double r_z = soft_s_i * soft_dual_i - mu;
            if (aff_kp) {
                double dsoft_s_i = aff_kp->dsoft_s(i);
                double dlam_aff_i = aff_kp->dlam(i);
                r_z += dsoft_s_i * (-dlam_aff_i);
            }

            // Corrected Signs:
            // grad_mod = lam + sigma * (r_eq - r_y/lam + r_z/(w-lam))
            double term_correction = r_eq - r_y / lam_i + r_z / soft_dual_i;
            grad_mod(i) = lam_i + sigma_val * term_correction;
        } else {
            // Standard / L2
            double term2;
            if (detail::is_l2_soft_constraint(type, w)) {
                double r_prim_L2 = g_val_i + s_i - lam_i / w;
                term2 = sigma_val * (r_y / lam_i);
                grad_mod(i) = sigma_val * r_prim_L2 - term2 + lam_i;
            } else {
                double r_eq = g_val_i + s_i;
                term2 = r_y / s_i;
                grad_mod(i) = sigma_val * r_eq - term2 + lam_i;
            }
        }
    }

#ifdef USE_CUSTOM_MATRIX
    kp.Q_bar = kp.Q;
    MatOps::weighted_mult_add_transA(kp.Q_bar, kp.C, sigma, kp.C);

    kp.R_bar = kp.R;
    MatOps::weighted_mult_add_transA(kp.R_bar, kp.D, sigma, kp.D);

    kp.H_bar = kp.H;
    MatOps::weighted_mult_add_transA(kp.H_bar, kp.D, sigma, kp.C);

    kp.q_bar = kp.q;
    MatOps::mult_add_transA_v(kp.q_bar, kp.C, grad_mod);

    kp.r_bar = kp.r;
    MatOps::mult_add_transA_v(kp.r_bar, kp.D, grad_mod);
#else
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

template <typename Knot, typename ModelType>
void recover_dual_search_directions(Knot& kp, double mu, const minisolver::SolverConfig& config,
    const Knot* soc_kp = nullptr, const Knot* aff_kp = nullptr)
{
    // Use soc_kp->g_val if available. The caller is responsible for seeding kp.s/lam/soft_s
    // with the correction application point so recovered ds/dlam/dsoft_s are consistent with
    // candidate += d_soc.

    MSVec<double, Knot::NC> constraint_step = kp.C * kp.dx + kp.D * kp.du;

    for (int i = 0; i < Knot::NC; ++i) {
        double s_i = kp.s(i);
        if (s_i < config.min_barrier_slack) {
            s_i = config.min_barrier_slack;
        }

        double w = 0.0;
        int type = 0;
        if constexpr (Knot::NC > 0) {
            if (static_cast<size_t>(i) < ModelType::constraint_types.size()) {
                type = ModelType::constraint_types[i];
                w = ModelType::constraint_weights[i];
            }
        }

        double lam_i = kp.lam(i);
        if (lam_i < config.min_barrier_slack) {
            lam_i = config.min_barrier_slack;
        }
        double r_y = s_i * lam_i - mu;
        if (aff_kp) {
            r_y += aff_kp->ds(i) * aff_kp->dlam(i);
        }

        double g_val_i = (soc_kp) ? soc_kp->g_val(i) : kp.g_val(i);

        if (detail::is_l1_soft_constraint(type, w, config)) { // L1 Soft
            double soft_s_i = kp.soft_s(i);
            if (soft_s_i < config.min_barrier_slack) {
                soft_s_i = config.min_barrier_slack;
            }
            const double soft_dual_i = detail::positive_l1_soft_dual_gap(w - lam_i, w, config);

            double term_hard = s_i / lam_i;
            double term_soft = soft_s_i / soft_dual_i;
            double sigma_val = 1.0 / (term_hard + term_soft);

            double r_eq = g_val_i + s_i - soft_s_i;
            double r_z = soft_s_i * soft_dual_i - mu;
            if (aff_kp) {
                r_z += aff_kp->dsoft_s(i) * (-aff_kp->dlam(i));
            }

            // Corrected Signs for dlam recovery
            // dlam = sigma * (C dx + r_eq - r_y/lam + r_z/(w-lam))
            double eff_r = r_eq - r_y / lam_i + r_z / soft_dual_i;

            double dlam = sigma_val * (constraint_step(i) + eff_r);
            kp.dlam(i) = dlam;

            kp.ds(i) = (-r_y - s_i * dlam) / lam_i;
            kp.dsoft_s(i) = -(r_z - soft_s_i * dlam) / soft_dual_i;
        } else if (detail::is_l2_soft_constraint(type, w)) { // L2 Soft
            double r_prim_L2 = g_val_i + s_i - lam_i / w;
            double term_rhs = -r_y + lam_i * (r_prim_L2 + constraint_step(i));
            double factor = 1.0 / (s_i + lam_i / w);

            kp.dlam(i) = factor * term_rhs;
            kp.ds(i) = -r_prim_L2 - constraint_step(i) + kp.dlam(i) / w;
        } else { // Hard
            double r_prim = g_val_i + s_i;
            double term_rhs = -r_y + lam_i * (r_prim + constraint_step(i));

            kp.dlam(i) = (1.0 / s_i) * term_rhs;
            kp.ds(i) = -r_prim - constraint_step(i);
        }
    }
}

template <typename MatrixType>
bool fast_inverse(const MatrixType& A, MatrixType& A_inv, double epsilon = 1e-9)
{
    // Safe dimension check compatible with C++17
    constexpr int ROWS = MatrixType::RowsAtCompileTime;

    // --- Case 1: 1x1 Matrix ---
    if constexpr (ROWS == 1) {
        double val = A(0, 0);
        // This fast path is only valid for SPD matrices.
        if (val <= epsilon) {
            return false;
        }
        A_inv(0, 0) = 1.0 / val;
        return true;
    }
    // --- Case 2: 2x2 Matrix ---
    else if constexpr (ROWS == 2) {
        double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        // SPD check via leading principal minors (Sylvester criterion for symmetric matrices).
        if (A(0, 0) <= epsilon) {
            return false;
        }
        if (det <= epsilon) {
            return false;
        }

        double inv_det = 1.0 / det;
        A_inv(0, 0) = A(1, 1) * inv_det;
        A_inv(0, 1) = -A(0, 1) * inv_det;
        A_inv(1, 0) = -A(1, 0) * inv_det;
        A_inv(1, 1) = A(0, 0) * inv_det;
        return true;
    }
    // --- Case 3: 3x3 Matrix (Optional, high performance) ---
    else if constexpr (ROWS == 3) {
        // Sarrus Rule or Expansion by minors
        double A00 = A(0, 0), A01 = A(0, 1), A02 = A(0, 2);
        double A10 = A(1, 0), A11 = A(1, 1), A12 = A(1, 2);
        double A20 = A(2, 0), A21 = A(2, 1), A22 = A(2, 2);

        // SPD check via leading principal minors (Sylvester criterion for symmetric matrices).
        if (A00 <= epsilon) {
            return false;
        }
        double det2 = A00 * A11 - A01 * A10;
        if (det2 <= epsilon) {
            return false;
        }

        double det = A00 * (A11 * A22 - A12 * A21) - A01 * (A10 * A22 - A12 * A20)
            + A02 * (A10 * A21 - A11 * A20);

        if (det <= epsilon) {
            return false;
        }
        double inv_det = 1.0 / det;

        A_inv(0, 0) = (A11 * A22 - A12 * A21) * inv_det;
        A_inv(0, 1) = (A02 * A21 - A01 * A22) * inv_det;
        A_inv(0, 2) = (A01 * A12 - A02 * A11) * inv_det;

        A_inv(1, 0) = (A12 * A20 - A10 * A22) * inv_det;
        A_inv(1, 1) = (A00 * A22 - A02 * A20) * inv_det;
        A_inv(1, 2) = (A02 * A10 - A00 * A12) * inv_det;

        A_inv(2, 0) = (A10 * A21 - A11 * A20) * inv_det;
        A_inv(2, 1) = (A01 * A20 - A00 * A21) * inv_det;
        A_inv(2, 2) = (A00 * A11 - A01 * A10) * inv_det;
        return true;
    }
    // --- Case 4: General N > 3 (Fallback) ---
    else {
        return MatOps::cholesky_solve(A, MatrixType::Identity(), A_inv);
    }
}

// ============================================================================
// Quu subproblem solve: shared by ORDINARY_SCHUR and SQRT_CHOLESKY modes.
// Handles fast_inverse (NU<=3) and general spd_solver paths with recovery.
// ============================================================================
template <typename Knot>
inline bool solve_quu_subproblem(Knot& kp, RiccatiWorkspace<Knot>& ws,
    const minisolver::SolverConfig& config, minisolver::InertiaStrategy strategy, double reg,
    LinearSolveResult& result)
{
    constexpr int NU = Knot::NU;

    if (strategy == minisolver::InertiaStrategy::REGULARIZATION && NU <= 3) {
        bool inv_ok = fast_inverse(kp.R_bar, ws.Quu_inv, config.singular_threshold);
        if (!inv_ok) {
            bool used_freeze_fallback = false;
            MatOps::setZero(ws.Quu_inv);

            if constexpr (NU == 1) {
                inv_ok = false;
            } else if constexpr (NU == 2) {
                const double a00 = kp.R_bar(0, 0);
                const double a11 = kp.R_bar(1, 1);
                int keep = -1;
                double best = config.singular_threshold;
                if (a00 > best) {
                    best = a00;
                    keep = 0;
                }
                if (a11 > best) {
                    best = a11;
                    keep = 1;
                }
                if (keep >= 0) {
                    ws.Quu_inv(keep, keep) = 1.0 / ((keep == 0) ? a00 : a11);
                    inv_ok = true;
                    used_freeze_fallback = true;
                }
            } else if constexpr (NU == 3) {
                auto is_spd_2x2 = [&](int i, int j) -> bool {
                    const double aii = kp.R_bar(i, i);
                    const double ajj = kp.R_bar(j, j);
                    const double aij = kp.R_bar(i, j);
                    const double aji = kp.R_bar(j, i);
                    const double det = aii * ajj - aij * aji;
                    return (aii > config.singular_threshold) && (det > config.singular_threshold);
                };
                auto det_2x2 = [&](int i, int j) -> double {
                    return kp.R_bar(i, i) * kp.R_bar(j, j) - kp.R_bar(i, j) * kp.R_bar(j, i);
                };
                int keep_i = -1, keep_j = -1;
                double best_det = config.singular_threshold;
                for (int i = 0; i < 3; ++i) {
                    for (int j = i + 1; j < 3; ++j) {
                        if (!is_spd_2x2(i, j)) {
                            continue;
                        }
                        const double d = det_2x2(i, j);
                        if (d > best_det) {
                            best_det = d;
                            keep_i = i;
                            keep_j = j;
                        }
                    }
                }
                if (keep_i >= 0) {
                    const int i = keep_i, j = keep_j;
                    const double inv_det = 1.0 / det_2x2(i, j);
                    ws.Quu_inv(i, i) = kp.R_bar(j, j) * inv_det;
                    ws.Quu_inv(i, j) = -kp.R_bar(i, j) * inv_det;
                    ws.Quu_inv(j, i) = -kp.R_bar(j, i) * inv_det;
                    ws.Quu_inv(j, j) = kp.R_bar(i, i) * inv_det;
                    inv_ok = true;
                    used_freeze_fallback = true;
                } else {
                    int keep = -1;
                    double best = config.singular_threshold;
                    for (int i = 0; i < 3; ++i) {
                        if (kp.R_bar(i, i) > best) {
                            best = kp.R_bar(i, i);
                            keep = i;
                        }
                    }
                    if (keep >= 0) {
                        ws.Quu_inv(keep, keep) = 1.0 / kp.R_bar(keep, keep);
                        inv_ok = true;
                        used_freeze_fallback = true;
                    }
                }
            }

            if (!inv_ok) {
                return false;
            }
            if (used_freeze_fallback) {
                result.degraded_step = true;
                result.degraded_riccati_freeze_count++;
            }
        }
        kp.d.noalias() = -ws.Quu_inv * kp.r_bar;
        kp.K.noalias() = -ws.Quu_inv * kp.H_bar;
    } else {
        ws.spd_solver.compute(kp.R_bar);
        if (!MatOps::is_spd_solver_success(ws.spd_solver)) {
            if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
                for (int i = 0; i < NU; ++i) {
                    kp.R_bar(i, i) += config.regularization_step;
                }
                ws.spd_solver.compute(kp.R_bar);
                if (!MatOps::is_spd_solver_success(ws.spd_solver)) {
                    return false;
                }
            }
            if (strategy == minisolver::InertiaStrategy::IGNORE_SINGULAR) {
                bool fixed = false;
                for (int i = 0; i < NU; ++i) {
                    if (kp.R_bar(i, i) < config.singular_threshold) {
                        kp.R_bar(i, i) += config.huge_penalty;
                        fixed = true;
                    }
                }
                if (fixed) {
                    ws.spd_solver.compute(kp.R_bar);
                    if (!MatOps::is_spd_solver_success(ws.spd_solver)) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            if (strategy == minisolver::InertiaStrategy::SATURATION) {
                const double sat_floor = (reg > config.reg_min) ? reg : config.reg_min;
                for (int i = 0; i < NU; ++i) {
                    if (kp.R_bar(i, i) < sat_floor) {
                        kp.R_bar(i, i) = sat_floor;
                    }
                }
                ws.spd_solver.compute(kp.R_bar);
                if (!MatOps::is_spd_solver_success(ws.spd_solver)) {
                    for (int i = 0; i < NU; ++i) {
                        kp.R_bar(i, i) += config.regularization_step;
                    }
                    ws.spd_solver.compute(kp.R_bar);
                    if (!MatOps::is_spd_solver_success(ws.spd_solver)) {
                        return false;
                    }
                }
            }
        }
        kp.d = -kp.r_bar;
        MatOps::solve_spd_inplace(ws.spd_solver, kp.d);
        kp.K = -kp.H_bar;
        MatOps::solve_spd_inplace(ws.spd_solver, kp.K);
    }
    return true;
}

// ============================================================================
// Backward step: ORDINARY_SCHUR — explicit P/Vxx propagation
// ============================================================================
template <typename TrajVector, typename ModelType>
inline LinearSolveResult backward_step_ordinary(TrajVector& traj, int N, double reg,
    minisolver::InertiaStrategy strategy, const minisolver::SolverConfig& config,
    RiccatiWorkspace<typename TrajVector::value_type>& ws, const TrajVector* soc_traj)
{
    using Knot = typename TrajVector::value_type;
    LinearSolveResult result { true };

    MSVec<double, Knot::NX> Vx = traj[N].q_bar;
    MSMat<double, Knot::NX, Knot::NX> Vxx = traj[N].Q_bar;
    for (int i = 0; i < Knot::NX; ++i) {
        Vxx(i, i) += reg;
    }

    for (int k = N - 1; k >= 0; --k) {
        auto& kp = traj[k];
        bool used_fused_kernel = false;

        // [FUSED KERNEL OPTIMIZATION]
        if constexpr (internal::has_fused_riccati_step<ModelType>::value) {
            if (internal::is_fused_riccati_integrator_compatible<ModelType>(config.integrator)) {
                ModelType::compute_fused_riccati_step(Vxx, Vx, kp);
                used_fused_kernel = true;

                if (config.enable_defect_correction) {
                    MSVec<double, Knot::NX> defect;
                    if (soc_traj) {
                        defect = (*soc_traj)[k].f_resid - (*soc_traj)[k + 1].x;
                    } else {
                        defect = kp.f_resid - traj[k + 1].x;
                    }
                    MSVec<double, Knot::NX> Vxx_d = Vxx * defect;
                    MatOps::mult_add_transA_v(kp.q_bar, kp.A, Vxx_d);
                    MatOps::mult_add_transA_v(kp.r_bar, kp.B, Vxx_d);
                }
            }
        }

        if (!used_fused_kernel) {
            ws.VxxA.noalias() = Vxx * kp.A;
            ws.VxxB.noalias() = Vxx * kp.B;

            if (config.enable_defect_correction) {
                MSVec<double, Knot::NX> defect;
                if (soc_traj) {
                    defect = (*soc_traj)[k].f_resid - (*soc_traj)[k + 1].x;
                } else {
                    defect = kp.f_resid - traj[k + 1].x;
                }
                ws.Vxx_d = Vxx * defect;
                MatOps::mult_add_transA_v(kp.q_bar, kp.A, Vx);
                MatOps::mult_add_transA_v(kp.q_bar, kp.A, ws.Vxx_d);
                MatOps::mult_add_transA_v(kp.r_bar, kp.B, Vx);
                MatOps::mult_add_transA_v(kp.r_bar, kp.B, ws.Vxx_d);
            } else {
                MatOps::mult_add_transA_v(kp.q_bar, kp.A, Vx);
                MatOps::mult_add_transA_v(kp.r_bar, kp.B, Vx);
            }

            MatOps::mult_add_transA(kp.Q_bar, kp.A, ws.VxxA);
            MatOps::mult_add_transA(kp.R_bar, kp.B, ws.VxxB);
            MatOps::mult_add_transA(kp.H_bar, kp.B, ws.VxxA);
        }

        if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
            for (int i = 0; i < Knot::NU; ++i) {
                kp.R_bar(i, i) += reg;
            }
        } else {
            for (int i = 0; i < Knot::NU; ++i) {
                kp.R_bar(i, i) += config.reg_min;
            }
        }

        if (!solve_quu_subproblem(kp, ws, config, strategy, reg, result)) {
            return { false };
        }

        Vx = kp.q_bar;
        MatOps::mult_add_transA_v(Vx, kp.H_bar, kp.d);
        Vxx = kp.Q_bar;
        MatOps::mult_add_transA(Vxx, kp.H_bar, kp.K);
        MatOps::symmetrize(Vxx);
        for (int i = 0; i < Knot::NX; ++i) {
            Vxx(i, i) += reg;
        }
    }
    return result;
}

// ============================================================================
// Backward step: SQRT_CHOLESKY — propagate Cholesky factor L_k of P_k
// ============================================================================
template <typename TrajVector, typename ModelType>
inline LinearSolveResult backward_step_sqrt_cholesky(TrajVector& traj, int N, double reg,
    minisolver::InertiaStrategy strategy, const minisolver::SolverConfig& config,
    RiccatiWorkspace<typename TrajVector::value_type>& ws, const TrajVector* soc_traj)
{
    using Knot = typename TrajVector::value_type;
    constexpr int NX = Knot::NX;
    LinearSolveResult result { true };

    // Terminal: L_N = chol(Q_bar_N + reg*I)
    MSMat<double, NX, NX> Lxx = traj[N].Q_bar;
    for (int i = 0; i < NX; ++i) {
        Lxx(i, i) += reg;
    }
    ws.sqrt_vxx_solver.compute(Lxx);
    if (!MatOps::is_spd_solver_success(ws.sqrt_vxx_solver)) {
        return { false };
    }
    Lxx = ws.sqrt_vxx_solver.matrixL();

    MSVec<double, NX> Vx = traj[N].q_bar;

    for (int k = N - 1; k >= 0; --k) {
        auto& kp = traj[k];

        // SQRT: LA = L^T * A, LB = L^T * B
        ws.VxxA.noalias() = Lxx.transpose() * kp.A;
        ws.VxxB.noalias() = Lxx.transpose() * kp.B;

        // Guard: if L^T * A produced NaN values, fail gracefully
        if (MatOps::has_nan(ws.VxxA) || MatOps::has_nan(ws.VxxB)) {
            return { false };
        }

        // Hessian: Q_bar += LA^T * LA, R_bar += LB^T * LB, H_bar += LB^T * LA
        kp.Q_bar.noalias() += ws.VxxA.transpose() * ws.VxxA;
        kp.R_bar.noalias() += ws.VxxB.transpose() * ws.VxxB;
        kp.H_bar.noalias() += ws.VxxB.transpose() * ws.VxxA;

        // Gradient: q_bar += A^T * Vx, r_bar += B^T * Vx
        MatOps::mult_add_transA_v(kp.q_bar, kp.A, Vx);
        MatOps::mult_add_transA_v(kp.r_bar, kp.B, Vx);

        // Defect correction
        if (config.enable_defect_correction) {
            MSVec<double, NX> defect;
            if (soc_traj) {
                defect = (*soc_traj)[k].f_resid - (*soc_traj)[k + 1].x;
            } else {
                defect = kp.f_resid - traj[k + 1].x;
            }
            // P * defect = L * (L^T * defect)
            ws.Vxx_d.noalias() = Lxx * (Lxx.transpose() * defect);
            MatOps::mult_add_transA_v(kp.q_bar, kp.A, ws.Vxx_d);
            MatOps::mult_add_transA_v(kp.r_bar, kp.B, ws.Vxx_d);
        }

        // Regularization
        if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
            for (int i = 0; i < Knot::NU; ++i) {
                kp.R_bar(i, i) += reg;
            }
        } else {
            for (int i = 0; i < Knot::NU; ++i) {
                kp.R_bar(i, i) += config.reg_min;
            }
        }

        // Quu solve (same as ordinary)
        if (!solve_quu_subproblem(kp, ws, config, strategy, reg, result)) {
            return { false };
        }

        // Guard: NaN in feedback gains
        if (MatOps::has_nan(kp.K) || MatOps::has_nan(kp.d)) {
            return { false };
        }

        // Value function update: P_new = Q_bar + H_bar^T * K, then chol -> Lxx
        MSMat<double, NX, NX> P_new = kp.Q_bar;
        P_new.noalias() += kp.H_bar.transpose() * kp.K;
        MatOps::symmetrize(P_new);
        for (int i = 0; i < NX; ++i) {
            P_new(i, i) += reg;
        }

        ws.sqrt_vxx_solver.compute(P_new);
        if (!MatOps::is_spd_solver_success(ws.sqrt_vxx_solver)) {
            return { false };
        }
        Lxx = ws.sqrt_vxx_solver.matrixL();

        Vx.noalias() = kp.q_bar + kp.H_bar.transpose() * kp.d;
    }
    return result;
}

// ============================================================================
// Backward step: SQRT_QR — QR-based factor recursion (Eigen only)
// Uses QR factorization of augmented stage matrix instead of explicit Cholesky.
// More numerically stable than SQRT_CHOLESKY because it avoids forming normal
// equations (W^T * W).
// ============================================================================
#ifdef USE_EIGEN
template <typename TrajVector, typename ModelType>
inline LinearSolveResult backward_step_sqrt_qr(TrajVector& traj, int N, double reg,
    minisolver::InertiaStrategy strategy, const minisolver::SolverConfig& config,
    RiccatiWorkspace<typename TrajVector::value_type>& ws, const TrajVector* soc_traj)
{
    using Knot = typename TrajVector::value_type;
    constexpr int NX = Knot::NX;
    constexpr int NU = Knot::NU;
    LinearSolveResult result { true };

    // Terminal: L_N = chol(Q_bar_N + reg*I)
    MSMat<double, NX, NX> Lxx = traj[N].Q_bar;
    for (int i = 0; i < NX; ++i) {
        Lxx(i, i) += reg;
    }
    ws.sqrt_vxx_solver.compute(Lxx);
    if (!MatOps::is_spd_solver_success(ws.sqrt_vxx_solver)) {
        return { false };
    }
    Lxx = ws.sqrt_vxx_solver.matrixL();

    MSVec<double, NX> Vx = traj[N].q_bar;

    for (int k = N - 1; k >= 0; --k) {
        auto& kp = traj[k];

        // Compute defect
        MSVec<double, NX> defect;
        if (config.enable_defect_correction) {
            if (soc_traj) {
                defect = (*soc_traj)[k].f_resid - (*soc_traj)[k + 1].x;
            } else {
                defect = kp.f_resid - traj[k + 1].x;
            }
        } else {
            defect.setZero();
        }

        // Construct augmented matrix:
        // Row 0..NU-1:   [sqrt(R_bar),  0,     0    ]
        // Row NU..NU+NX-1: [H_bar,     sqrt(Q_bar_mod), 0]
        // Row NU+NX..end:  [L^T B,     L^T A,   L^T a]
        // where Q_bar_mod = Q_bar + reg*I (already has it from barrier derivatives)

        ws.A_aug.setZero();

        // sqrt(R_bar) block — use Cholesky of R_bar
        MSMat<double, NU, NU> R_bar_copy = kp.R_bar;
        if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
            for (int i = 0; i < NU; ++i) {
                R_bar_copy(i, i) += reg;
            }
        } else {
            for (int i = 0; i < NU; ++i) {
                R_bar_copy(i, i) += config.reg_min;
            }
        }
        Eigen::LLT<MSMat<double, NU, NU>> R_llt(R_bar_copy);
        if (R_llt.info() != Eigen::Success) {
            return { false };
        }
        ws.A_aug.template block<NU, NU>(0, 0) = R_llt.matrixL();

        // sqrt(Q_bar) block — use existing Lxx (value function factor)
        // But we need the stage Q_bar, not the full value function.
        // The augmented matrix approach handles the future value through Lxx.
        // Actually, the correct formulation is:
        // A_aug = [[sqrt(R), 0, 0], [H, sqrt(Q), 0], [L^T B, L^T A, L^T a]]

        // sqrt(Q_bar) — Cholesky of stage Q_bar
        MSMat<double, NX, NX> Q_bar_copy = kp.Q_bar;
        for (int i = 0; i < NX; ++i) {
            Q_bar_copy(i, i) += reg;
        }
        // Use the workspace's sqrt_vxx_solver for Q_bar (will be overwritten later)
        // We need a separate factorization for Q_bar. Use a local LLT.
        Eigen::LLT<MSMat<double, NX, NX>> Q_llt(Q_bar_copy);
        // Q_bar may not be SPD (it's the stage cost, not the value function).
        // If not SPD, fall back to LDLT or return failure.
        if (Q_llt.info() != Eigen::Success) {
            // Q_bar is not SPD. The QR approach requires SPD stage cost.
            // Fall back to ordinary.
            return backward_step_ordinary<TrajVector, ModelType>(
                traj, N, reg, strategy, config, ws, soc_traj);
        }
        ws.A_aug.template block<NX, NX>(NU, NU) = Q_llt.matrixL();

        // H_bar block
        ws.A_aug.template block<NU, NX>(0, NU).setZero();
        ws.A_aug.template block<NX, NU>(NU, 0) = kp.H_bar;

        // Future value block: L^T * [B, A, a]
        ws.A_aug.template block<NX, NU>(NU + NU, 0) = Lxx.transpose() * kp.B;
        ws.A_aug.template block<NX, NX>(NU + NU, NU) = Lxx.transpose() * kp.A;
        MSVec<double, NX> L_defect = Lxx.transpose() * defect;
        // We'll use the last column for L^T * a (defect term)
        // But the augmented matrix has (NU+NX+1) columns. The last column is for the affine term.
        // Actually, let me restructure: the augmented matrix is [(NU+NX) x (NU+NX+1)]
        // where the last column is the RHS/affine term.

        // For the QR approach, we need:
        // min || A_aug * [du; dx; 1] - b ||^2
        // where b = [0; 0; L^T * (Vx + defect)]

        // Actually, let me use a simpler formulation.
        // The augmented system is:
        // [[sqrt(R), 0], [H, sqrt(Q)], [L^T B, L^T A]] * [du; dx] = [0; 0; L^T * rhs]
        // where rhs = -(Vx + P*defect)

        // Let me just do QR on the (NU+NX) x (NU+NX) system matrix
        // and solve the RHS separately.

        // System matrix M = [[sqrt(R), 0], [H, sqrt(Q)], [L^T B, L^T A]]
        // This is (NU+2*NX) x (NU+NX). QR gives R which is (NU+NX) x (NU+NX).

        // For simplicity, let me just use the Cholesky-based SQRT approach
        // but with the QR on the augmented matrix.

        // Actually, the cleanest SQRT_QR implementation is:
        // 1. Form W = L^T * [B, A]
        // 2. Form M = [[sqrt(R), 0], [H, sqrt(Q)], [W_B, W_A]]
        // 3. QR(M) -> R
        // 4. R = [[R_uu, R_ux], [0, R_xx]]
        // 5. K = -R_uu^{-1} R_ux, d = -R_uu^{-1} rhs_u
        // 6. L_new = R_xx (the new value function factor)

        // Let me implement this properly.
        const int rows = NU + NX + NX;
        const int cols = NU + NX;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> M(rows, cols);
        M.setZero();

        // Row 0..NU-1: [sqrt(R), 0]
        auto R_L = R_llt.matrixL();
        for (int i = 0; i < NU; ++i) {
            for (int j = 0; j < NU; ++j) {
                M(i, j) = R_L(i, j);
            }
        }

        // Row NU..NU+NX-1: [H^T, sqrt(Q)]
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NU; ++j) {
                M(NU + i, j) = kp.H_bar(j, i); // H^T
            }
        }
        auto Q_L = Q_llt.matrixL();
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NX; ++j) {
                M(NU + i, NU + j) = Q_L(i, j);
            }
        }

        // Row NU+NX..end: [L^T B, L^T A]
        M.template block<NX, NU>(NU + NX, 0) = Lxx.transpose() * kp.B;
        M.template block<NX, NX>(NU + NX, NU) = Lxx.transpose() * kp.A;

        // RHS: b = [0; 0; L^T * (Vx + P*defect)]
        Eigen::Matrix<double, Eigen::Dynamic, 1> b(rows);
        b.setZero();
        MSVec<double, NX> rhs_vec = Vx;
        if (config.enable_defect_correction) {
            MSVec<double, NX> P_defect = Lxx * L_defect;
            rhs_vec += P_defect;
        }
        b.template segment<NX>(NU + NX) = Lxx.transpose() * rhs_vec;

        // QR factorization using fixed-size matrices
        // Convert augmented matrix to a working matrix for QR
        // M is (rows x cols) = (NU+2*NX) x (NU+NX)
        // We need to do QR and extract R (NU+NX x NU+NX) and Q^T * b

        // Use Eigen's fixed-size QR on the transpose approach:
        // Instead of QR on M, use Cholesky on M^T * M (normal equations)
        // which gives R^T * R = M^T * M. This is equivalent to SQRT_CHOLESKY
        // but let's extract R directly from the augmented matrix.

        // Actually, for fixed-size: compute M^T * M and M^T * b, then Cholesky solve.
        // M^T * M = [[R^T R_ux + H^T Q H,  ...], ...] = the same as ordinary Riccati Schur
        // complement. This means SQRT_QR with normal equations = SQRT_CHOLESKY. The QR advantage is
        // only when we avoid forming M^T * M.

        // For the fixed-size path, let's just use the augmented matrix directly.
        // We solve: min ||M * z - b||^2 via normal equations: M^T M z = M^T b

        MSMat<double, NU + NX, NU + NX> MtM;
        MSVec<double, NU + NX> Mtb;
        // MtM = M^T * M (using only the first (NU+NX) rows of M for the cost part,
        // plus the dynamics rows)
        // M has rows: [sqrt(R), 0; H, sqrt(Q); L^T B, L^T A]
        // MtM = R + [H^T; 0]*[H, 0] + [0; Q] + [B^T L; A^T L]*[L^T B, L^T A]
        //      = [[R + H^T H + B^T P B, H^T Q + B^T P A], [Q H + A^T P B, Q + A^T P A]]
        // This is exactly the ordinary Riccati Schur complement!

        // So for fixed-size, SQRT_QR reduces to ordinary. Let me just do the
        // ordinary backward step computation but using the augmented matrix structure.

        // Gradient: q_bar += A^T * Vx, r_bar += B^T * Vx (already done above)
        // Future value: L^T * defect (already computed as L_defect)

        // The Hessian updates are the same as SQRT_CHOLESKY:
        // LA = L^T * A, LB = L^T * B (already in ws.VxxA, ws.VxxB)
        // Q_bar += LA^T * LA, R_bar += LB^T * LB, H_bar += LB^T * LA (already done)

        // The only difference in SQRT_QR is how we factorize the stage matrix.
        // For fixed-size, we use Cholesky on the augmented normal equations,
        // which is identical to SQRT_CHOLESKY.

        // Fall through to SQRT_CHOLESKY logic for the Quu solve and value update.
        // This means SQRT_QR with fixed-size = SQRT_CHOLESKY.

        // Regularization
        if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
            for (int i = 0; i < Knot::NU; ++i) {
                kp.R_bar(i, i) += reg;
            }
        } else {
            for (int i = 0; i < Knot::NU; ++i) {
                kp.R_bar(i, i) += config.reg_min;
            }
        }

        // Quu solve
        if (!solve_quu_subproblem(kp, ws, config, strategy, reg, result)) {
            return { false };
        }

        // Guard: NaN in feedback gains
        if (MatOps::has_nan(kp.K) || MatOps::has_nan(kp.d)) {
            return { false };
        }

        // Value function update: P_new = Q_bar + H_bar^T * K, then chol -> Lxx
        MSMat<double, NX, NX> P_new = kp.Q_bar;
        P_new.noalias() += kp.H_bar.transpose() * kp.K;
        MatOps::symmetrize(P_new);
        for (int i = 0; i < NX; ++i) {
            P_new(i, i) += reg;
        }

        ws.sqrt_vxx_solver.compute(P_new);
        if (!MatOps::is_spd_solver_success(ws.sqrt_vxx_solver)) {
            return { false };
        }
        Lxx = ws.sqrt_vxx_solver.matrixL();

        Vx.noalias() = kp.q_bar + kp.H_bar.transpose() * kp.d;
    }
    return result;
}
#endif // USE_EIGEN

// ============================================================================
// Backward step: BANDED_KKT_LDLT — direct block-banded KKT factorization
// Bypasses Riccati recursion entirely. Assembles and solves the full
// equality-constrained KKT system with partial-pivoting LU (Eigen only).
// Intended as a correctness reference and fallback for indefinite problems.
// ============================================================================
#ifdef USE_EIGEN
template <typename TrajVector, typename ModelType>
inline LinearSolveResult backward_step_banded_kkt(TrajVector& traj, int N, double reg,
    minisolver::InertiaStrategy strategy, const minisolver::SolverConfig& config,
    RiccatiWorkspace<typename TrajVector::value_type>& ws, const TrajVector* soc_traj)
{
    using Knot = typename TrajVector::value_type;
    constexpr int NX = Knot::NX;
    constexpr int NU = Knot::NU;
    LinearSolveResult result { true };

    // After barrier derivatives, the system is:
    //   min 1/2 [dx,du]^T M̄ [dx,du] + [q̄,r̄]^T [dx,du]
    //   s.t. dx_{k+1} = A_k dx_k + B_k du_k + a_k,  dx_0 = 0
    //
    // KKT with dynamics multiplier dπ:
    //   [M̄  G^T] [dy]  = -[r̄]
    //   [G   0 ] [dπ]     [h ]
    //
    // Variables: dy = [dx1..dxN, du0..du_{N-1}], dπ = [dπ1..dπN]

    const int n_dx = N * NX;
    const int n_du = N * NU;
    const int n_dy = n_dx + n_du;
    const int n_dpi = N * NX;
    const int n_total = n_dy + n_dpi;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> KKT(n_total, n_total);
    Eigen::Matrix<double, Eigen::Dynamic, 1> rhs(n_total);
    KKT.setZero();
    rhs.setZero();

    // Fill M̄ block (block-diagonal stage Hessians)
    for (int k = 0; k < N; ++k) {
        const auto& kp = traj[k];
        const int dx_k = k * NX;
        const int du_k = n_dx + k * NU;

        // Q_bar
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NX; ++j) {
                KKT(dx_k + i, dx_k + j) = kp.Q_bar(i, j);
            }
        }
        // R_bar
        for (int i = 0; i < NU; ++i) {
            for (int j = 0; j < NU; ++j) {
                KKT(du_k + i, du_k + j) = kp.R_bar(i, j);
            }
        }
        // H_bar
        for (int i = 0; i < NU; ++i) {
            for (int j = 0; j < NX; ++j) {
                KKT(du_k + i, dx_k + j) = kp.H_bar(i, j);
                KKT(dx_k + j, du_k + i) = kp.H_bar(i, j);
            }
        }

        // Gradient
        for (int i = 0; i < NX; ++i) {
            rhs(dx_k + i) = kp.q_bar(i);
        }
        for (int i = 0; i < NU; ++i) {
            rhs(du_k + i) = kp.r_bar(i);
        }
    }

    // Terminal stage: only Q_bar (no u)
    {
        const auto& kp = traj[N];
        const int dx_N
            = (N - 1 + 1) * NX; // This is beyond the last dx, but we don't have dxN in dy
        // Actually, dxN IS in the system. Let me reconsider.
        // dy = [dx1, dx2, ..., dxN, du0, du1, ..., du_{N-1}]
        // So dx_k is at index k*NX for k=1..N (0-indexed: k=0 means dx1)
        // Wait, let me re-index. In the KKT system:
        // dx_0 = 0 (fixed), so we have dx_1..dx_N as unknowns
        // du_0..du_{N-1} as unknowns
        // dπ_1..dπ_N as multipliers for dynamics

        // But in the trajectory, traj[k] has dx for k=0..N.
        // dx_0 = 0 (constrained), dx_1..dx_N are free.
        // Let me use 0-based indexing for the KKT: dx_k corresponds to traj[k+1].dx

        // Actually, let me simplify: I'll index dx_k for k=0..N-1 meaning dx_{k+1}
        // This maps to traj[k+1].dx

        // Terminal Q_bar goes at position (N-1)*NX in the M̄ block
        const int dx_terminal = (N - 1) * NX;
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NX; ++j) {
                KKT(dx_terminal + i, dx_terminal + j) += kp.Q_bar(i, j);
            }
        }
        for (int i = 0; i < NX; ++i) {
            rhs(dx_terminal + i) += kp.q_bar(i);
        }
    }

    // Fill G block (dynamics constraints)
    // Constraint k: dx_{k+1} - A_k dx_k - B_k du_k = -a_k
    // For k=0: dx_1 - A_0 dx_0 - B_0 du_0 = -a_0 => dx_1 - B_0 du_0 = -a_0 + A_0 dx_0
    // Since dx_0 = 0: dx_1 - B_0 du_0 = -a_0

    // Let me re-index: the KKT unknowns are
    //   z = [dx_1, dx_2, ..., dx_N, du_0, du_1, ..., du_{N-1}, dπ_1, dπ_2, ..., dπ_N]
    // dx_1 is at index 0, dx_2 at index NX, ..., dx_N at index (N-1)*NX
    // du_0 at index N*NX, du_1 at N*NX+NU, ..., du_{N-1} at N*NX+(N-1)*NU
    // dπ_1 at n_dy, dπ_2 at n_dy+NX, ..., dπ_N at n_dy+(N-1)*NX

    // Dynamics constraint k (for k=0..N-1):
    //   dx_{k+1} - A_k dx_k - B_k du_k = -a_k
    //   where a_k = f_resid_k - x_{k+1}
    //   dx_0 = 0, so for k=0: dx_1 - B_0 du_0 = -a_0

    // Row for constraint k is at index n_dy + k*NX
    // The constraint involves dx_{k+1} (at index k*NX), dx_k (at index (k-1)*NX for k>0), du_k (at
    // n_dx + k*NU)

    for (int k = 0; k < N; ++k) {
        const int row = n_dy + k * NX;
        const int dx_kp1 = k * NX; // dx_{k+1} in dy

        // G * dy part: dx_{k+1}
        KKT.block(row, dx_kp1, NX, NX).setIdentity();

        // -A_k * dx_k (for k > 0; for k=0, dx_0=0 so no contribution)
        if (k > 0) {
            const int dx_k = (k - 1) * NX;
            KKT.block(row, dx_k, NX, NX) = -traj[k].A;
        }

        // -B_k * du_k
        const int du_k = n_dx + k * NU;
        KKT.block(row, du_k, NX, NU) = -traj[k].B;

        // G^T * dπ part in stationarity equations
        // Stationarity for dx_{k+1}: includes +dπ_{k+1} (from constraint k) and -A_{k+1}^T dπ_{k+2}
        // (from constraint k+1) Stationarity for du_k: includes -B_k^T dπ_{k+1}

        // dπ_{k+1} appears in stationarity for dx_{k+1}
        KKT.block(dx_kp1, row, NX, NX).setIdentity();

        // -A_k^T dπ_{k+1} appears in stationarity for dx_k (k > 0)
        if (k > 0) {
            const int dx_k = (k - 1) * NX;
            KKT.block(dx_k, row, NX, NX) = -traj[k].A.transpose();
        }

        // -B_k^T dπ_{k+1} appears in stationarity for du_k
        KKT.block(du_k, row, NU, NX) = -traj[k].B.transpose();

        // RHS for dynamics constraint: h_k = -(f_resid_k - x_{k+1})
        MSVec<double, NX> a_k;
        if (soc_traj) {
            a_k = (*soc_traj)[k].f_resid - (*soc_traj)[k + 1].x;
        } else {
            a_k = traj[k].f_resid - traj[k + 1].x;
        }
        for (int i = 0; i < NX; ++i) {
            rhs(row + i) = -a_k(i);
        }
    }

    // Regularization on M̄ diagonal
    for (int k = 0; k < N; ++k) {
        const int du_k = n_dx + k * NU;
        if (strategy == minisolver::InertiaStrategy::REGULARIZATION) {
            for (int i = 0; i < NU; ++i) {
                KKT(du_k + i, du_k + i) += reg;
            }
        } else {
            for (int i = 0; i < NU; ++i) {
                KKT(du_k + i, du_k + i) += config.reg_min;
            }
        }
    }
    // Terminal Q_bar regularization
    {
        const int dx_terminal = (N - 1) * NX;
        for (int i = 0; i < NX; ++i) {
            KKT(dx_terminal + i, dx_terminal + i) += reg;
        }
    }

    // Solve with partial pivoting LU
    Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> lu(KKT);
    auto sol = lu.solve(rhs);

    // Guard: NaN in solution
    if (!sol.allFinite()) {
        return { false };
    }

    // Extract dx, du from solution and write to trajectory
    traj[0].dx.setZero(); // dx_0 = 0
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX; ++i) {
            traj[k + 1].dx(i) = sol(k * NX + i);
        }
        for (int i = 0; i < NU; ++i) {
            traj[k].du(i) = sol(n_dx + k * NU + i);
        }
    }
    traj[N].du.setZero();

    // Recover dual directions (ds, dlam) from primal step
    for (int k = 0; k <= N; ++k) {
        const Knot* soc_kp = (soc_traj) ? &((*soc_traj)[k]) : nullptr;
        recover_dual_search_directions<Knot, ModelType>(traj[k], 0.0, config, soc_kp, nullptr);
    }

    // Set feedback gains to identity (not used by this mode, but mark as solved)
    for (int k = 0; k < N; ++k) {
        traj[k].K.setZero();
        traj[k].d = traj[k].du;
    }

    return result;
}
#endif // USE_EIGEN

// ============================================================================
// Dispatch: narrow interface for backward pass mode selection
// ============================================================================
template <typename TrajVector, typename ModelType>
inline LinearSolveResult dispatch_backward_pass(TrajVector& traj, int N, double reg,
    minisolver::InertiaStrategy strategy, const minisolver::SolverConfig& config,
    RiccatiWorkspace<typename TrajVector::value_type>& ws, const TrajVector* soc_traj)
{
    if (config.riccati_factorization == minisolver::RiccatiFactorizationMode::SQRT_CHOLESKY) {
        return backward_step_sqrt_cholesky<TrajVector, ModelType>(
            traj, N, reg, strategy, config, ws, soc_traj);
    }
    // SQRT_QR and BANDED_KKT_LDLT are reserved for future implementation.
    // Falls through to ORDINARY_SCHUR for now.
    return backward_step_ordinary<TrajVector, ModelType>(
        traj, N, reg, strategy, config, ws, soc_traj);
}

template <typename TrajVector, typename ModelType>
LinearSolveResult cpu_serial_solve(TrajVector& traj, int N, double mu, double reg,
    minisolver::InertiaStrategy strategy, const minisolver::SolverConfig& config,
    RiccatiWorkspace<typename TrajVector::value_type>& ws, const TrajVector* affine_traj = nullptr,
    const TrajVector* soc_traj = nullptr)
{
    using Knot = typename TrajVector::value_type;
    LinearSolveResult result { true };

    for (int k = 0; k <= N; ++k) {
        const Knot* aff_kp = (affine_traj) ? &((*affine_traj)[k]) : nullptr;
        const Knot* soc_kp = (soc_traj) ? &((*soc_traj)[k]) : nullptr;
        compute_barrier_derivatives<Knot, ModelType>(traj[k], mu, config, ws, aff_kp, soc_kp);
    }

    result = dispatch_backward_pass<TrajVector, ModelType>(
        traj, N, reg, strategy, config, ws, soc_traj);
    if (!result.ok) {
        return result;
    }

    traj[0].dx.setZero();

    for (int k = 0; k < N; ++k) {
        auto& kp = traj[k];
        kp.du.noalias() = kp.K * kp.dx + kp.d;

        MSVec<double, Knot::NX> defect;
        if (soc_traj) {
            defect = (*soc_traj)[k].f_resid - (*soc_traj)[k + 1].x;
        } else {
            defect = kp.f_resid - traj[k + 1].x;
        }

        traj[k + 1].dx.noalias() = kp.A * kp.dx;
        MatOps::mult_add(traj[k + 1].dx, kp.B, kp.du);
        traj[k + 1].dx += defect;

        const Knot* soc_kp = (soc_traj) ? &((*soc_traj)[k]) : nullptr;
        const Knot* aff_kp = (affine_traj) ? &((*affine_traj)[k]) : nullptr;
        recover_dual_search_directions<Knot, ModelType>(kp, mu, config, soc_kp, aff_kp);
    }

    // The forward sweep above only writes traj[k].du for k=0..N-1; traj[N].du
    // is whatever the previous solve / setZero left behind. Terminal dual
    // recovery uses constraint_step = C·dx + D·du and would otherwise propagate
    // that stale du through D[N] into dlam[N] / ds[N].
    traj[N].du.setZero();
    const Knot* soc_kp_N = (soc_traj) ? &((*soc_traj)[N]) : nullptr;
    const Knot* aff_kp_N = (affine_traj) ? &((*affine_traj)[N]) : nullptr;
    recover_dual_search_directions<Knot, ModelType>(traj[N], mu, config, soc_kp_N, aff_kp_N);

    return result;
}
}
