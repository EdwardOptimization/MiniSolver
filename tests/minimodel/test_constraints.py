import tempfile

from common import (
    Dot,
    OptimalControlModel,
    compile_and_run,
    expect_value_error,
    reject,
    require,
)


def test_quad_boundary_projection_codegen_uses_unique_temps():
    model = OptimalControlModel("TwoProjModel")
    x0, x1 = model.state("x0", "x1")
    u0 = model.control("u0")
    model.subject_to(Dot(x0) == u0)
    model.subject_to(Dot(x1) == 0)
    model.subject_to_quad(
        [[1, 0], [0, 1]], [x0, x1], center=[0, 0], rhs=1.0,
        type="outside", linearize_at_boundary=True)
    model.subject_to_quad(
        [[1, 0], [0, 1]], [x0, x1], center=[1, 1], rhs=1.0,
        type="outside", linearize_at_boundary=True)
    model.minimize(x0**2 + x1**2 + u0**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        with open(f"{tmpdir}/twoprojmodel.h", "r", encoding="utf-8") as f:
            text = f.read()

    require(text, "T d2_0 =")
    require(text, "T rhs_0 =")
    require(text, "T scale_0 =")
    require(text, "T d2_1 =")
    require(text, "T rhs_1 =")
    require(text, "T scale_1 =")
    reject(text, "T d2 =")


def test_quad_boundary_projection_generates_soc_override():
    model = OptimalControlModel("SocProjectionModel")
    x0, x1 = model.state("x0", "x1")
    u0 = model.control("u0")
    model.subject_to(Dot(x0) == u0)
    model.subject_to(Dot(x1) == 0)
    model.subject_to_quad(
        [[1, 0], [0, 1]], [x0, x1], center=[0, 0], rhs=1.0,
        type="outside", linearize_at_boundary=True)
    model.minimize(x0**2 + x1**2 + u0**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "soc_projection_check.cpp",
            "soc_projection_check",
            """
            #include "socprojectionmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::SocProjectionModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> active;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> trial;
                active.set_zero();
                trial.set_zero();

                active.x(0) = 2.0;
                active.x(1) = 0.0;
                trial.x(0) = 0.0;
                trial.x(1) = 2.0;

                Model::compute_constraints(trial);
                if (std::abs(trial.g_val(0) - (-2.0)) > 1e-8) return 1;

                Model::compute_soc_constraints(active, trial);
                if (std::abs(trial.g_val(0) - 2.0) > 1e-8) return 2;
                return 0;
            }
            """,
        )


def test_quad_boundary_projection_splits_qp_and_true_residuals():
    model = OptimalControlModel("ConstraintPacketModel")
    x0, x1 = model.state("x0", "x1")
    u0 = model.control("u0")
    model.subject_to(Dot(x0) == u0)
    model.subject_to(Dot(x1) == 0)
    model.subject_to_quad(
        [[1, 0], [0, 1]], [x0, x1], center=[0, 0], rhs=1.0,
        type="outside", linearize_at_boundary=True)
    model.minimize(x0**2 + x1**2 + u0**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "constraint_packet_check.cpp",
            "constraint_packet_check",
            """
            #include "constraintpacketmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::ConstraintPacketModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 0.0;
                kp.x(1) = 2.0;

                Model::compute_qp_constraints(kp);
                Model::compute_true_constraints(kp);

                if (std::abs(kp.g_val(0) - (-2.0)) > 1e-8) return 1;
                if (std::abs(kp.g_true(0) - (-1.000000125)) > 1e-6) return 2;
                kp.g_val(0) = 999.0;
                Model::compute_constraints(kp);
                if (std::abs(kp.g_val(0) - (-2.0)) > 1e-8) return 3;
                return 0;
            }
            """,
        )


def test_quad_norm2_generates_exact_constraint_hessian():
    model = OptimalControlModel("QuadNorm2Model")
    x = model.state("x")
    u0, u1, rho = model.control("u0", "u1", "rho")
    model.subject_to(Dot(x) == 0)
    model.subject_to_quad(
        [[1, 0], [0, 1]], [u0, u1], rhs=rho, rhs_mode="norm2", type="inside")
    model.minimize(rho + 1.0e-8 * (u0**2 + u1**2 + rho**2))

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "quad_norm2_check.cpp",
            "quad_norm2_check",
            """
            #include "quadnorm2model.h"
            #include <cmath>

            int main() {
                using Model = minisolver::QuadNorm2Model;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.u(0) = 3.0;
                kp.u(1) = 4.0;
                kp.u(2) = 5.0;
                kp.lam(0) = 2.0;

                Model::compute_constraints(kp);
                if (std::abs(kp.g_val(0)) > 1e-8) return 1;
                if (std::abs(kp.D(0, 0) - 0.6) > 1e-8) return 2;
                if (std::abs(kp.D(0, 1) - 0.8) > 1e-8) return 3;
                if (std::abs(kp.D(0, 2) + 1.0) > 1e-8) return 4;

                Model::compute_cost_exact(kp);
                const double base = 2.0e-8;
                if (std::abs((kp.R(0, 0) - base) - 2.0 * 0.128) > 1e-6) return 5;
                if (std::abs(kp.R(0, 1) - 2.0 * -0.096) > 1e-6) return 6;
                if (std::abs((kp.R(1, 1) - base) - 2.0 * 0.072) > 1e-6) return 7;
                return 0;
            }
            """,
        )


def test_stage_only_constraint_zeros_terminal_row():
    model = OptimalControlModel("StageOnlyConstraintModel")
    x = model.state("x")
    u = model.control("u")
    model.subject_to(Dot(x) == u)
    model.subject_to(u - 2.0 <= 0, include_terminal=False)
    model.subject_to_quad([[1]], [u], rhs=3.0, rhs_mode="norm2",
                          type="inside", include_terminal=False)
    model.minimize(x**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "stage_only_constraint_check.cpp",
            "stage_only_constraint_check",
            """
            #include "stageonlyconstraintmodel.h"
            #include <cmath>

            int main() {
                using Model = minisolver::StageOnlyConstraintModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.u(0) = 5.0;

                Model::compute_constraints(kp);
                if (std::abs(kp.g_val(0) - 3.0) > 1e-8) return 1;
                if (std::abs(kp.g_val(1) - 2.0) > 1e-6) return 2;

                Model::compute_terminal_qp_constraints(kp);
                Model::compute_terminal_true_constraints(kp);
                if (std::abs(kp.g_val(0)) > 1e-12) return 3;
                if (std::abs(kp.g_val(1)) > 1e-12) return 4;
                if (std::abs(kp.g_true(0)) > 1e-12) return 5;
                if (std::abs(kp.g_true(1)) > 1e-12) return 6;
                return 0;
            }
            """,
        )


def test_quad_constraint_domain_guards():
    def negative_rhs():
        model = OptimalControlModel("BadRhsModel")
        x = model.state("x")
        model.subject_to_quad([[1]], [x], rhs=-1.0, type="outside")

    def non_psd_q():
        model = OptimalControlModel("BadQModel")
        x0, x1 = model.state("x0", "x1")
        model.subject_to_quad([[1, 0], [0, -1]], [x0, x1], rhs=1.0, type="outside")

    def invalid_rhs_mode():
        model = OptimalControlModel("BadModeModel")
        x = model.state("x")
        model.subject_to_quad([[1]], [x], rhs=1.0, rhs_mode="sqrt_norm2")

    def negative_norm2_rhs():
        model = OptimalControlModel("BadNorm2RhsModel")
        x = model.state("x")
        model.subject_to_quad([[1]], [x], rhs=-1.0, rhs_mode="norm2", type="inside")

    expect_value_error(negative_rhs, "rhs")
    expect_value_error(non_psd_q, "PSD")
    expect_value_error(invalid_rhs_mode, "rhs_mode")
    expect_value_error(negative_norm2_rhs, "norm2 rhs")


def test_generated_model_uses_automatic_constraint_row_scaling():
    model = OptimalControlModel("GeneratedScaleModel")
    x = model.state("x")
    u = model.control("u")
    model.subject_to(Dot(x) == 0)
    model.subject_to(x - 1.0 <= 0)
    model.subject_to(1000.0 * (x - 1.0) <= 0)
    model.minimize(x**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "generated_scale_check.cpp",
            "generated_scale_check",
            """
            #include "generatedscalemodel.h"
            #include "minisolver/solver/solver.h"
            #include <cmath>

            int main() {
                using Model = minisolver::GeneratedScaleModel;
                static_assert(Model::NC == 2, "expected two constraint rows");

                minisolver::SolverConfig config;
                config.print_level = minisolver::PrintLevel::NONE;
                config.max_iters = 0;
                config.integrator = minisolver::IntegratorType::EULER_EXPLICIT;
                config.constraint_scaling = minisolver::ConstraintScalingMethod::ROW_INF_NORM;

                minisolver::MiniSolver<Model, 1> solver(0, minisolver::Backend::CPU_SERIAL, config);
                solver.set_initial_state("x", 2.0);
                const minisolver::SolverStatus status = solver.solve();
                const minisolver::SolverInfo& info = solver.get_info();

                if (status != minisolver::SolverStatus::MAX_ITER) return 1;
                if (!info.constraint_scaling_active) return 2;
                if (std::abs(info.primal_inf - 2.0) > 1e-5) return 3;
                if (std::abs(info.unscaled_primal_inf - 2000.0) > 1e-2) return 4;
                return 0;
            }
            """,
        )


if __name__ == "__main__":
    test_quad_boundary_projection_codegen_uses_unique_temps()
    test_quad_boundary_projection_generates_soc_override()
    test_quad_boundary_projection_splits_qp_and_true_residuals()
    test_quad_norm2_generates_exact_constraint_hessian()
    test_stage_only_constraint_zeros_terminal_row()
    test_quad_constraint_domain_guards()
    test_generated_model_uses_automatic_constraint_row_scaling()
