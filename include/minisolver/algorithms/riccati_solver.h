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

    bool last_solve_degraded() const { return last_solve_degraded_; }

    bool solve(TrajArray& traj, int N, double mu, double reg, InertiaStrategy strategy,
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
            return false;
        }

        // Call the static/template function with Model type info
        bool degraded = false;
        const bool ok = cpu_serial_solve<TrajArray, Model>(
            traj, N, mu, reg, strategy, config, workspace, affine_traj, nullptr, &degraded);
        last_solve_degraded_ = degraded;
        return ok;
    }

    // SOC Implementation
    bool solve_soc(TrajArray& traj, const TrajArray& soc_rhs_traj, int N, double mu, double reg,
        InertiaStrategy strategy, const SolverConfig& config) override
    {
        bool degraded = false;
        const bool ok = cpu_serial_solve<TrajArray, Model>(
            traj, N, mu, reg, strategy, config, workspace, nullptr, &soc_rhs_traj, &degraded);
        last_solve_degraded_ = degraded;
        return ok;
    }

    bool evaluate_dual_residual(TrajArray& scratch_traj, int N, double mu, double reg,
        InertiaStrategy strategy, const SolverConfig& config, double& max_dual_inf) override
    {
        RiccatiWorkspace<Knot> scratch_workspace;
        const bool ok = cpu_serial_solve<TrajArray, Model>(
            scratch_traj, N, mu, reg, strategy, config, scratch_workspace);
        if (!ok) {
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
    // Current strategy: correct the linearized dynamics defect by rolling dx/du forward through
    // the existing Riccati feedback gains. This is not full KKT iterative refinement.
    bool refine_direction(TrajArray& traj, const TrajArray& original_system, int N, double /*mu*/,
        double /*reg*/, const SolverConfig& config) override
    {
        if (config.direction_refinement != DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT) {
            return true;
        }

        // This pass enforces strict dynamic feasibility of the linear solution:
        // dx_{k+1} = A dx_k + B du_k + defect
        // It propagates the calculation error accumulated during the backward/forward Riccati pass.

        MSVec<double, Model::NX> delta_x;
        delta_x.setZero(); // Initial state correction is zero (x0 fixed)

        MSVec<double, Model::NU> delta_u;

        // Use workspace (original_system) to access A, B matrices
        const TrajArray& sys = original_system;

        double max_defect = 0.0;

        for (int k = 0; k < N; ++k) {
            // 1. Compute Control Correction via Feedback
            // du_new = K * (dx + delta_x) + d - du_old
            // Since du_old = K * dx + d, then:
            // delta_u = K * delta_x
            delta_u.noalias() = traj[k].K * delta_x;

            // 2. Compute Dynamic Defect of the current solution
            // expected_next = A * dx + B * du + (f_resid - x_next_base)
            // defect = expected_next - dx_next_actual

            // Reconstruct the affine term (linearization defect)
            // In Riccati: defect_term = sys[k].f_resid - sys[k+1].x;
            MSVec<double, Model::NX> linearization_defect = sys[k].f_resid - sys[k + 1].x;

            MSVec<double, Model::NX> predicted_dx_next;
            predicted_dx_next.noalias() = sys[k].A * traj[k].dx;
            predicted_dx_next.noalias() += sys[k].B * traj[k].du;
            predicted_dx_next += linearization_defect;

            MSVec<double, Model::NX> error = predicted_dx_next - traj[k + 1].dx;
            // [FIX] Use MatOps::norm_inf
            double err_norm = MatOps::norm_inf(error);
            if (err_norm > max_defect) {
                max_defect = err_norm;
            }

            // 3. Propagate Correction
            // delta_x_{k+1} = A * delta_x + B * delta_u + error
            // This ensures (dx + delta_x)_{k+1} matches the dynamics of (dx+delta_x)_k
            MSVec<double, Model::NX> delta_x_next;
            delta_x_next.noalias() = sys[k].A * delta_x;
            delta_x_next.noalias() += sys[k].B * delta_u;
            delta_x_next += error;

            // 4. Apply to Trajectory
            traj[k].dx += delta_x;
            traj[k].du += delta_u;

            delta_x = delta_x_next;
        }

        // Apply last state correction
        traj[N].dx += delta_x;

        if (config.print_level >= PrintLevel::DEBUG) {
            MLOG_DEBUG("Direction refinement: max dynamic defect corrected = " << max_defect);
        }

        return true;
    }

private:
    bool last_solve_degraded_ = false;
};

}
