import tempfile

import sympy as sp

from common import OptimalControlModel, compile_and_run, expect_value_error


def test_add_residual_generates_true_gauss_newton_hessian():
    model = OptimalControlModel("ResidualCostModel")
    x = model.state("x")
    u = model.control("u")
    model.set_dynamics(x, u)
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
    model.set_dynamics(x, u)
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
    model.set_dynamics(x, u)
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
    model.set_dynamics(x, u)
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
    model.set_dynamics(x, u)
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
    test_add_residual_validates_weights()
