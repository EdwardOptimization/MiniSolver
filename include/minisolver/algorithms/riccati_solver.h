#pragma once
#include "minisolver/algorithms/linear_solver.h"
#include "minisolver/core/logger.h"
#include "minisolver/solver/riccati.h" // Reuse existing implementation functions

namespace minisolver {

// We need to pass ModelType to RiccatiSolver to access constraint metadata
// But LinearSolver base class doesn't have ModelType.
// The easiest way is to template RiccatiSolver on ModelType as well,
// or pass metadata dynamically (but we prefer static for performance).
// However, MiniSolver<Model, N> owns the RiccatiSolver.
// So we can make RiccatiSolver<TrajArray, Model>

template <typename TrajArray, typename Model> class RiccatiSolver : public LinearSolver<TrajArray> {
public:
    using Knot = typename TrajArray::value_type;

    // Persistent workspace to avoid re-allocation
    RiccatiWorkspace<Knot> workspace;

    LinearSolveResult solve(TrajArray& traj, int N, double mu, double reg, InertiaStrategy strategy,
        const SolverConfig& config, const TrajArray* affine_traj = nullptr) override
    {

        // GPU backends are reserved but not implemented. Do not silently run CPU when the
        // requested backend is GPU; that would make benchmark/deployment results misleading.
        if (config.backend == Backend::GPU_MPX || config.backend == Backend::GPU_PCR) {
#ifdef USE_CUDA
            MLOG_ERROR("GPU backend is not implemented yet.");
#else
            MLOG_ERROR("CUDA not enabled; GPU backend is unsupported.");
#endif
            return { false };
        }

        // Call the static/template function with Model type info
        return cpu_serial_solve<TrajArray, Model>(
            traj, N, mu, reg, strategy, config, workspace, affine_traj);
    }

    // SOC Implementation
    LinearSolveResult solve_soc(TrajArray& traj, const TrajArray& soc_rhs_traj, int N, double mu,
        double reg, InertiaStrategy strategy, const SolverConfig& config) override
    {
        return cpu_serial_solve<TrajArray, Model>(
            traj, N, mu, reg, strategy, config, workspace, nullptr, &soc_rhs_traj);
    }

    bool evaluate_dual_residual(TrajArray& scratch_traj, int N, double mu, double reg,
        InertiaStrategy strategy, const SolverConfig& config, double& max_dual_inf) override
    {
        RiccatiWorkspace<Knot> scratch_workspace;
        const LinearSolveResult result = cpu_serial_solve<TrajArray, Model>(
            scratch_traj, N, mu, reg, strategy, config, scratch_workspace);
        if (!result.ok) {
            max_dual_inf = std::numeric_limits<double>::infinity();
            return false;
        }

        max_dual_inf = 0.0;
        for (int k = 0; k <= N; ++k) {
            const double r_norm = MatOps::norm_inf(scratch_traj[k].r_bar);
            if (r_norm > max_dual_inf) {
                max_dual_inf = r_norm;
            }
        }
        return true;
    }

    // Direction refinement implementation.
    //
    // DYNAMICS_DEFECT_ROLLOUT: a single pass that corrects the linearized
    // dynamics defect by rolling dx/du forward through the existing Riccati
    // feedback gains. This is not full KKT iterative refinement.
    //
    // FULL_KKT_ITERATIVE_REFINEMENT: repeats the same dynamics-defect rollout
    // up to direction_refinement_max_passes times or until the maximum
    // dynamic-defect inf-norm drops below direction_refinement_tol. Despite
    // the name it remains a *structured* refinement that reuses the existing
    // Riccati feedback gains; it does not re-factorize the KKT matrix and
    // does not rebuild slack/dual directions. The mode auto-degrades to a
    // single pass when any inequality dual is non-trivial so the OD-005
    // dual-consistency hazard is not amplified by repeated primal-only
    // refinements.
    bool refine_direction(TrajArray& traj, const TrajArray& original_system, int N, double /*mu*/,
        double /*reg*/, const SolverConfig& config) override
    {
        this->last_refine_passes_ = 0;
        this->last_refine_defect_ = 0.0;

        if (config.direction_refinement == DirectionRefinementMode::NONE) {
            return true;
        }

        const bool iterative = (config.direction_refinement
            == DirectionRefinementMode::FULL_KKT_ITERATIVE_REFINEMENT);
        int max_passes = 1;
        if (iterative) {
            max_passes = std::max(1, config.direction_refinement_max_passes);
            if (has_active_inequality_duals_(traj, N)) {
                max_passes = 1;
                if (config.print_level >= PrintLevel::DEBUG) {
                    MLOG_DEBUG("FULL_KKT_ITERATIVE_REFINEMENT auto-degraded to single pass: "
                               "active inequality duals detected");
                }
            }
        }

        const double tol = std::max(0.0, config.direction_refinement_tol);
        double last_defect = 0.0;
        int passes_done = 0;
        for (int pass = 0; pass < max_passes; ++pass) {
            last_defect = run_dynamics_defect_rollout_pass_(traj, original_system, N);
            ++passes_done;
            if (config.print_level >= PrintLevel::DEBUG) {
                MLOG_DEBUG("Direction refinement pass "
                    << passes_done << "/" << max_passes
                    << ": max dynamic defect = " << last_defect);
            }
            if (last_defect <= tol) {
                break;
            }
        }

        this->last_refine_passes_ = passes_done;
        this->last_refine_defect_ = last_defect;
        return true;
    }

private:
    static double run_dynamics_defect_rollout_pass_(
        TrajArray& traj, const TrajArray& original_system, int N)
    {
        // Single sweep of the linearized dynamics-defect refinement. Returns
        // the maximum dynamic defect encountered before applying the
        // correction so callers can decide whether to iterate again.
        const TrajArray& sys = original_system;

        MSVec<double, Model::NX> delta_x;
        delta_x.setZero();
        MSVec<double, Model::NU> delta_u;

        double max_defect = 0.0;
        for (int k = 0; k < N; ++k) {
            delta_u.noalias() = traj[k].K * delta_x;

            MSVec<double, Model::NX> linearization_defect = sys[k].f_resid - sys[k + 1].x;
            MSVec<double, Model::NX> predicted_dx_next;
            predicted_dx_next.noalias() = sys[k].A * traj[k].dx;
            predicted_dx_next.noalias() += sys[k].B * traj[k].du;
            predicted_dx_next += linearization_defect;

            MSVec<double, Model::NX> error = predicted_dx_next - traj[k + 1].dx;
            const double err_norm = MatOps::norm_inf(error);
            if (err_norm > max_defect) {
                max_defect = err_norm;
            }

            MSVec<double, Model::NX> delta_x_next;
            delta_x_next.noalias() = sys[k].A * delta_x;
            delta_x_next.noalias() += sys[k].B * delta_u;
            delta_x_next += error;

            traj[k].dx += delta_x;
            traj[k].du += delta_u;

            delta_x = delta_x_next;
        }
        traj[N].dx += delta_x;
        return max_defect;
    }

    static bool has_active_inequality_duals_(const TrajArray& traj, int N)
    {
        // "Active" here uses the same conservative threshold as OD-005: any
        // inequality multiplier above 1e-6 indicates a binding constraint
        // whose dual direction should not silently drift across multiple
        // refinement passes.
        constexpr double active_threshold = 1e-6;
        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < Model::NC; ++i) {
                if (traj[k].lam(i) > active_threshold) {
                    return true;
                }
            }
        }
        return false;
    }
};

}
