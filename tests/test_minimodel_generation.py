import os
import subprocess
import sys
import tempfile
import textwrap


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from minisolver.MiniModel import OptimalControlModel


def require(text, needle):
    if needle not in text:
        raise AssertionError(f"missing generated snippet: {needle}")


def reject(text, needle):
    if needle in text:
        raise AssertionError(f"unexpected generated snippet: {needle}")


def expect_value_error(fn, needle):
    try:
        fn()
    except ValueError as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected error containing {needle!r}, got {exc!r}") from exc
        return
    raise AssertionError("expected ValueError")


def generate_chain_model(integrator_type):
    model = OptimalControlModel("ImplicitPatternRegressionModel")

    x0, x1, x2 = model.state("x0", "x1", "x2")
    u = model.control("u")

    # Continuous lower chain:
    #   Jx has (1,0), (2,1), so explicit Euler A misses A(2,0).
    #   (I - dt*Jx)^-1 has transitive fill-in, so implicit Riccati pattern
    #   must conservatively keep A(2,0) and B(2,0).
    model.set_dynamics(x0, u)
    model.set_dynamics(x1, x0)
    model.set_dynamics(x2, x1)
    model.minimize(x0**2 + x1**2 + x2**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, use_fused_riccati=True, integrator_type=integrator_type)
        header_path = os.path.join(tmpdir, "implicitpatternregressionmodel.h")
        with open(header_path, "r", encoding="utf-8") as f:
            return f.read()


def check_implicit_chain_pattern(integrator_type):
    text = generate_chain_model(integrator_type)

    require(text, f"generated_integrator = IntegratorType::{integrator_type}")

    # The selected implicit integrator has a solve/inverse in its discrete
    # Jacobian path, so the lower chain must keep transitive fill-in.
    require(text, "T A_2_0 = kp.A(2,0);")
    require(text, "T B_2_0 = kp.B(2,0);")

    # But the directed chain should not be widened into an undirected dense
    # component. That would be correct but slower and would hide whether the
    # integrator-specific pattern path is actually tighter.
    reject(text, "T A_0_2 = kp.A(0,2);")
    reject(text, "T A_0_1 = kp.A(0,1);")


def test_implicit_riccati_pattern_keeps_inverse_fill_in():
    for integrator_type in ("EULER_IMPLICIT", "RK2_IMPLICIT", "RK4_IMPLICIT"):
        check_implicit_chain_pattern(integrator_type)


def test_cpp_identifier_validation_rejects_keywords_and_duplicates():
    def keyword_state():
        model = OptimalControlModel("KeywordModel")
        model.state("class")

    def duplicate_names():
        model = OptimalControlModel("DuplicateModel")
        model.state("x")
        model.control("x")

    def generated_temp_collision():
        model = OptimalControlModel("TempCollisionModel")
        model.state("dt")

    expect_value_error(keyword_state, "C++")
    expect_value_error(duplicate_names, "duplicate")
    expect_value_error(generated_temp_collision, "reserved")


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
        header_path = os.path.join(tmpdir, "twoprojmodel.h")
        with open(header_path, "r", encoding="utf-8") as f:
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
        source = os.path.join(tmpdir, "soc_projection_check.cpp")
        exe = os.path.join(tmpdir, "soc_projection_check")
        with open(source, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(
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
                """))
        subprocess.run(
            [
                "g++", "-std=c++17", "-DUSE_CUSTOM_MATRIX",
                f"-I{ROOT}/include", f"-I{tmpdir}", source, "-o", exe
            ],
            check=True,
        )
        subprocess.run([exe], check=True)


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
        source = os.path.join(tmpdir, "constraint_packet_check.cpp")
        exe = os.path.join(tmpdir, "constraint_packet_check")
        with open(source, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(
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
                """))
        subprocess.run(
            [
                "g++", "-std=c++17", "-DUSE_CUSTOM_MATRIX",
                f"-I{ROOT}/include", f"-I{tmpdir}", source, "-o", exe
            ],
            check=True,
        )
        subprocess.run([exe], check=True)


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


def test_generated_terminal_stage_uses_x_only_projection():
    model = OptimalControlModel("TerminalProjectionModel")
    x = model.state("x")
    u = model.control("u")
    model.set_dynamics(x, u)
    model.minimize((x + u) ** 2 + 3.0 * u**2)
    model.subject_to(x + 2.0 * u - 4.0 <= 0)
    model.subject_to(u - 1.0 <= 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        source = os.path.join(tmpdir, "terminal_projection_check.cpp")
        exe = os.path.join(tmpdir, "terminal_projection_check")
        with open(source, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(
                """
                #include "terminalprojectionmodel.h"
                #include <cmath>
                #include <cstdlib>

                int main() {
                    using Model = minisolver::TerminalProjectionModel;
                    minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                    kp.set_zero();
                    kp.x(0) = 3.0;
                    kp.u(0) = 10.0;
                    Model::compute_terminal_cost_exact(kp);
                    Model::compute_terminal_qp_constraints(kp);
                    if (std::abs(kp.cost - 9.0) > 1e-12) return 1;
                    if (std::abs(kp.q(0) - 6.0) > 1e-12) return 2;
                    if (std::abs(kp.r(0)) > 1e-12) return 3;
                    if (std::abs(kp.g_val(0) - (-1.0)) > 1e-12) return 4;
                    if (std::abs(kp.D(0,0)) > 1e-12) return 5;
                    if (std::abs(kp.g_val(1) - (-1.0)) > 1e-12) return 6;
                    return 0;
                }
                """))
        subprocess.run(
            [
                "g++", "-std=c++17", "-DUSE_CUSTOM_MATRIX",
                f"-I{ROOT}/include", f"-I{tmpdir}", source, "-o", exe
            ],
            check=True,
        )
        subprocess.run([exe], check=True)


if __name__ == "__main__":
    test_implicit_riccati_pattern_keeps_inverse_fill_in()
    test_cpp_identifier_validation_rejects_keywords_and_duplicates()
    test_quad_boundary_projection_codegen_uses_unique_temps()
    test_quad_boundary_projection_generates_soc_override()
    test_quad_boundary_projection_splits_qp_and_true_residuals()
    test_quad_constraint_domain_guards()
    test_generated_terminal_stage_uses_x_only_projection()
