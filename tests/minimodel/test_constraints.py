import tempfile

from common import (
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
    model.set_dynamics(x0, u0)
    model.set_dynamics(x1, 0)
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
    model.set_dynamics(x0, u0)
    model.set_dynamics(x1, 0)
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
    model.set_dynamics(x0, u0)
    model.set_dynamics(x1, 0)
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


def test_quad_constraint_domain_guards():
    def negative_rhs():
        model = OptimalControlModel("BadRhsModel")
        x = model.state("x")
        model.subject_to_quad([[1]], [x], rhs=-1.0, type="outside")

    def non_psd_q():
        model = OptimalControlModel("BadQModel")
        x0, x1 = model.state("x0", "x1")
        model.subject_to_quad([[1, 0], [0, -1]], [x0, x1], rhs=1.0, type="outside")

    expect_value_error(negative_rhs, "rhs")
    expect_value_error(non_psd_q, "PSD")


if __name__ == "__main__":
    test_quad_boundary_projection_codegen_uses_unique_temps()
    test_quad_boundary_projection_generates_soc_override()
    test_quad_boundary_projection_splits_qp_and_true_residuals()
    test_quad_constraint_domain_guards()
