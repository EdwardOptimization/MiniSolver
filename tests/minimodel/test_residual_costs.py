import tempfile

import sympy as sp

from common import Dot, OptimalControlModel, compile_and_run, expect_value_error, reject, require


def test_add_residual_generates_true_gauss_newton_hessian():
    model = OptimalControlModel("ResidualCostModel")
    x = model.state("x")
    u = model.control("u")
    model.subject_to(Dot(x) == u)
    model.add_residual(sp.sin(x), weight=3.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "residual_cost_check.cpp",
            "residual_cost_check",
            """
            #include "residualcostmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::ResidualCostModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 0.4;

                const double s = std::sin(0.4);
                const double c = std::cos(0.4);
                const double expected_cost = 0.5 * 3.0 * s * s;
                const double expected_grad = 3.0 * s * c;
                const double expected_gn_hess = 3.0 * c * c;
                const double expected_exact_hess = 3.0 * (c * c - s * s);

                Model::compute_cost_gn(kp);
                if (std::abs(kp.cost - expected_cost) > 1e-12) return 1;
                if (std::abs(kp.q(0) - expected_grad) > 1e-12) return 2;
                if (std::abs(kp.Q(0,0) - expected_gn_hess) > 1e-12) return 3;

                Model::compute_cost_exact(kp);
                if (std::abs(kp.cost - expected_cost) > 1e-12) return 4;
                if (std::abs(kp.q(0) - expected_grad) > 1e-12) return 5;
                if (std::abs(kp.Q(0,0) - expected_exact_hess) > 1e-12) return 6;
                return 0;
            }
            """,
        )


def test_add_residual_mixes_general_objective_and_vector_weights():
    model = OptimalControlModel("MixedResidualCostModel")
    x = model.state("x")
    u = model.control("u")
    model.subject_to(Dot(x) == u)
    model.minimize(x**4)
    model.add_residual([sp.sin(x), u], weight=[2.0, 0.5])

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "mixed_residual_cost_check.cpp",
            "mixed_residual_cost_check",
            """
            #include "mixedresidualcostmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::MixedResidualCostModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 0.3;
                kp.u(0) = -0.2;

                const double x = 0.3;
                const double u = -0.2;
                const double s = std::sin(x);
                const double c = std::cos(x);
                const double expected_cost = std::pow(x, 4) + s * s + 0.25 * u * u;
                const double expected_q = 4.0 * std::pow(x, 3) + 2.0 * s * c;
                const double expected_r = 0.5 * u;
                const double expected_gn_Q = 12.0 * x * x + 2.0 * c * c;
                const double expected_exact_Q = 12.0 * x * x + 2.0 * (c * c - s * s);

                Model::compute_cost_gn(kp);
                if (std::abs(kp.cost - expected_cost) > 1e-12) return 1;
                if (std::abs(kp.q(0) - expected_q) > 1e-12) return 2;
                if (std::abs(kp.r(0) - expected_r) > 1e-12) return 3;
                if (std::abs(kp.Q(0,0) - expected_gn_Q) > 1e-12) return 4;
                if (std::abs(kp.R(0,0) - 0.5) > 1e-12) return 5;

                Model::compute_cost_exact(kp);
                if (std::abs(kp.Q(0,0) - expected_exact_Q) > 1e-12) return 6;
                if (std::abs(kp.R(0,0) - 0.5) > 1e-12) return 7;
                return 0;
            }
            """,
        )


def test_add_residual_terminal_stage_projects_controls():
    model = OptimalControlModel("TerminalResidualCostModel")
    x = model.state("x")
    u = model.control("u")
    model.subject_to(Dot(x) == u)
    model.add_residual(x + u, weight=2.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "terminal_residual_cost_check.cpp",
            "terminal_residual_cost_check",
            """
            #include "terminalresidualcostmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::TerminalResidualCostModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 3.0;
                kp.u(0) = 10.0;
                Model::compute_terminal_cost_gn(kp);
                if (std::abs(kp.cost - 9.0) > 1e-12) return 1;
                if (std::abs(kp.q(0) - 6.0) > 1e-12) return 2;
                if (std::abs(kp.r(0)) > 1e-12) return 3;
                if (std::abs(kp.Q(0,0) - 2.0) > 1e-12) return 4;
                if (std::abs(kp.R(0,0)) > 1e-12) return 5;
                return 0;
            }
            """,
        )


def test_add_residual_accepts_parameter_vector_weight():
    model = OptimalControlModel("ParameterResidualWeightModel")
    x = model.state("x")
    u = model.control("u")
    x_weight = model.parameter("x_weight")
    model.subject_to(Dot(x) == u)
    model.add_residual([x], weight=[x_weight])

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "parameter_residual_weight_check.cpp",
            "parameter_residual_weight_check",
            """
            #include "parameterresidualweightmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::ParameterResidualWeightModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 0.5;
                kp.p(0) = 4.0;

                Model::compute_cost_gn(kp);
                if (std::abs(kp.cost - 0.5) > 1e-12) return 1;
                if (std::abs(kp.q(0) - 2.0) > 1e-12) return 2;
                if (std::abs(kp.Q(0,0) - 4.0) > 1e-12) return 3;

                Model::compute_cost_exact(kp);
                if (std::abs(kp.cost - 0.5) > 1e-12) return 4;
                if (std::abs(kp.q(0) - 2.0) > 1e-12) return 5;
                if (std::abs(kp.Q(0,0) - 4.0) > 1e-12) return 6;
                return 0;
            }
            """,
        )


def test_add_residual_accepts_parameter_reference():
    model = OptimalControlModel("ParameterResidualReferenceModel")
    x = model.state("x")
    u = model.control("u")
    x_ref = model.parameter("x_ref")
    x_weight = model.parameter("x_weight")
    model.subject_to(Dot(x) == u)
    model.add_residual(x - x_ref, weight=x_weight)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "parameter_residual_reference_check.cpp",
            "parameter_residual_reference_check",
            """
            #include "parameterresidualreferencemodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::ParameterResidualReferenceModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 3.0;
                kp.p(0) = 1.0; // x_ref
                kp.p(1) = 5.0; // x_weight

                Model::compute_cost_gn(kp);
                if (std::abs(kp.cost - 10.0) > 1e-12) return 1;
                if (std::abs(kp.q(0) - 10.0) > 1e-12) return 2;
                if (std::abs(kp.Q(0,0) - 5.0) > 1e-12) return 3;
                return 0;
            }
            """,
        )


def test_sparse_generated_packets_zero_first_then_assign_nonzero():
    model = OptimalControlModel("SparsePacketModel")
    x0, x1 = model.state("x0", "x1")
    u0, u1 = model.control("u0", "u1")
    model.subject_to(Dot(x0) == u0)
    model.subject_to(Dot(x1) == 0)
    model.subject_to(x0 <= 1.0)
    model.subject_to(u1 <= 2.0)
    model.minimize(x0**2 + 0.5 * u1**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        with open(f"{tmpdir}/sparsepacketmodel.h", "r", encoding="utf-8") as f:
            text = f.read()

        require(text, "kp.C.setZero();")
        require(text, "kp.D.setZero();")
        require(text, "kp.Q.setZero();")
        require(text, "kp.R.setZero();")
        require(text, "kp.H.setZero();")
        require(text, "jac.Jx.setZero();")
        require(text, "jac.Ju.setZero();")
        reject(text, "kp.C(0,0) = 0;")
        reject(text, "kp.D(0,0) = 0;")
        reject(text, "kp.Q(0,1) = 0;")
        reject(text, "kp.R(0,0) = 0;")
        reject(text, "kp.H(0,0) = 0;")

        compile_and_run(
            tmpdir,
            "sparse_packet_check.cpp",
            "sparse_packet_check",
            """
            #include "sparsepacketmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::SparsePacketModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 0.25;
                kp.x(1) = -0.5;
                kp.u(0) = 0.1;
                kp.u(1) = 0.2;

                Model::compute_constraints(kp);
                if (std::abs(kp.C(0,0) - 1.0) > 1e-12) return 1;
                if (std::abs(kp.C(0,1)) > 1e-12) return 2;
                if (std::abs(kp.D(1,1) - 1.0) > 1e-12) return 3;
                if (std::abs(kp.D(1,0)) > 1e-12) return 4;

                Model::compute_cost_gn(kp);
                if (std::abs(kp.Q(0,0) - 2.0) > 1e-12) return 5;
                if (std::abs(kp.Q(0,1)) > 1e-12) return 6;
                if (std::abs(kp.R(1,1) - 1.0) > 1e-12) return 7;
                if (std::abs(kp.R(0,0)) > 1e-12) return 8;
                if (std::abs(kp.H(0,0)) > 1e-12) return 9;

                auto jac = Model::jacobian_continuous(kp.x, kp.u, kp.p);
                if (std::abs(jac.Ju(0,0) - 1.0) > 1e-12) return 10;
                if (std::abs(jac.Ju(0,1)) > 1e-12) return 11;
                if (std::abs(jac.Jx(0,0)) > 1e-12) return 12;
                return 0;
            }
            """,
        )


def test_full_generated_packets_skip_clear():
    model = OptimalControlModel("FullPacketModel")
    x0, x1 = model.state("x0", "x1")
    u0, u1 = model.control("u0", "u1")
    model.subject_to(Dot(x0) == u0)
    model.subject_to(Dot(x1) == u1)
    model.subject_to([x0 + 1.0 <= 0.0, x1 + u0 + 2.0 <= 0.0])
    model.minimize(x0**2 + 2.0 * x1**2 + 3.0 * u0**2 + 4.0 * u1**2 + x0 * u0 + x1 * u1)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        with open(f"{tmpdir}/fullpacketmodel.h", "r", encoding="utf-8") as f:
            text = f.read()

        constraints_body = text.split("// --- 2. Compute QP/IPM Constraints")[1]
        constraints_body = constraints_body.split("// --- 2.5 Compute True Constraints")[0]
        cost_body = text.split("static void compute_cost_impl")[1]
        cost_body = cost_body.split("// --- 3.5 Terminal Cost")[0]

        reject(constraints_body, "kp.g_val.setZero();")
        require(constraints_body, "kp.C.setZero();")
        require(constraints_body, "kp.D.setZero();")
        reject(cost_body, "kp.q.setZero();")
        reject(cost_body, "kp.r.setZero();")
        require(cost_body, "kp.Q.setZero();")
        require(cost_body, "kp.R.setZero();")
        require(cost_body, "kp.H.setZero();")

        compile_and_run(
            tmpdir,
            "full_packet_check.cpp",
            "full_packet_check",
            """
            #include "fullpacketmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::FullPacketModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 0.25;
                kp.x(1) = -0.5;
                kp.u(0) = 0.1;
                kp.u(1) = 0.2;

                kp.g_val.fill(99.0);
                kp.C.fill(99.0);
                kp.D.fill(99.0);
                Model::compute_constraints(kp);
                if (std::abs(kp.g_val(0) - 1.25) > 1e-12) return 1;
                if (std::abs(kp.g_val(1) - 1.6) > 1e-12) return 2;
                if (std::abs(kp.C(0,0) - 1.0) > 1e-12) return 3;
                if (std::abs(kp.C(0,1)) > 1e-12) return 4;
                if (std::abs(kp.C(1,1) - 1.0) > 1e-12) return 5;
                if (std::abs(kp.D(1,0) - 1.0) > 1e-12) return 6;
                if (std::abs(kp.D(0,1)) > 1e-12) return 7;

                kp.g_true.fill(99.0);
                Model::compute_true_constraints(kp);
                if (std::abs(kp.g_true(0) - 1.25) > 1e-12) return 15;
                if (std::abs(kp.g_true(1) - 1.6) > 1e-12) return 16;

                kp.q.fill(99.0);
                kp.r.fill(99.0);
                kp.Q.fill(99.0);
                kp.R.fill(99.0);
                kp.H.fill(99.0);
                Model::compute_cost_exact(kp);
                if (std::abs(kp.q(0) - 0.6) > 1e-12) return 8;
                if (std::abs(kp.q(1) + 1.8) > 1e-12) return 9;
                if (std::abs(kp.r(0) - 0.85) > 1e-12) return 10;
                if (std::abs(kp.r(1) - 1.1) > 1e-12) return 11;
                if (std::abs(kp.Q(0,1)) > 1e-12) return 12;
                if (std::abs(kp.R(0,1)) > 1e-12) return 13;
                if (std::abs(kp.H(1,0)) > 1e-12) return 14;
                return 0;
            }
            """,
        )


def test_add_residual_validates_weights():
    def negative_weight():
        model = OptimalControlModel("NegativeResidualWeightModel")
        x = model.state("x")
        model.add_residual(x, weight=-1.0)

    def mismatched_weights():
        model = OptimalControlModel("MismatchedResidualWeightModel")
        x, y = model.state("x", "y")
        model.add_residual([x, y], weight=[1.0])

    expect_value_error(negative_weight, "weight")
    expect_value_error(mismatched_weights, "weight")


if __name__ == "__main__":
    test_add_residual_generates_true_gauss_newton_hessian()
    test_add_residual_mixes_general_objective_and_vector_weights()
    test_add_residual_terminal_stage_projects_controls()
    test_add_residual_accepts_parameter_vector_weight()
    test_add_residual_accepts_parameter_reference()
    test_sparse_generated_packets_zero_first_then_assign_nonzero()
    test_full_generated_packets_skip_clear()
    test_add_residual_validates_weights()
