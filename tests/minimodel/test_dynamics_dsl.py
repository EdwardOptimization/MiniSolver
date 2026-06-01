import tempfile

from common import compile_and_run, expect_value_error, generate_header_text, require
from minisolver.MiniModel import Dot, Next, OptimalControlModel


def test_dot_subject_to_generates_continuous_dynamics():
    model = OptimalControlModel("DotDslModel")
    x = model.state("x")
    u = model.control("u")

    model.subject_to(Dot(x) == u)
    model.minimize(x**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "dot_dsl_check.cpp",
            "dot_dsl_check",
            """
            #include "dotdslmodel.h"
            #include <cmath>

            int main() {
                using Model = minisolver::DotDslModel;
                minisolver::MSVec<double, Model::NX> x;
                minisolver::MSVec<double, Model::NU> u;
                minisolver::MSVec<double, Model::NP> p;
                x(0) = 2.0;
                u(0) = 3.0;

                const auto xdot = Model::dynamics_continuous(x, u, p);
                if (std::abs(xdot(0) - 3.0) > 1e-12) return 1;

                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 2.0;
                kp.u(0) = 3.0;
                Model::compute_dynamics(kp, minisolver::IntegratorType::EULER_EXPLICIT, 0.25);
                if (std::abs(kp.f_resid(0) - 2.75) > 1e-12) return 2;
                if (std::abs(kp.A(0, 0) - 1.0) > 1e-12) return 3;
                if (std::abs(kp.B(0, 0) - 0.25) > 1e-12) return 4;
                return 0;
            }
            """,
        )


def test_next_subject_to_generates_discrete_dynamics_map():
    model = OptimalControlModel("NextDslModel")
    x = model.state("x")
    u = model.control("u")
    h = model.parameter("h")

    model.subject_to(Next(x) == x + h * u)
    model.minimize(x**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir)
        compile_and_run(
            tmpdir,
            "next_dsl_check.cpp",
            "next_dsl_check",
            """
            #include "nextdslmodel.h"
            #include "minisolver/integrator/implicit_integrator.h"
            #include <cmath>

            int main() {
                using Model = minisolver::NextDslModel;
                static_assert(!minisolver::detail::has_dynamics_continuous_v<Model, double>,
                    "Next-generated models must not expose continuous dynamics");
                static_assert(!minisolver::detail::has_jacobian_continuous_v<Model, double>,
                    "Next-generated models must not expose continuous Jacobians");

                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 2.0;
                kp.u(0) = 3.0;
                kp.p(0) = 0.25;

                Model::compute_dynamics(kp, minisolver::IntegratorType::DISCRETE, 99.0);
                if (std::abs(kp.f_resid(0) - 2.75) > 1e-12) return 1;
                if (std::abs(kp.A(0, 0) - 1.0) > 1e-12) return 2;
                if (std::abs(kp.B(0, 0) - 0.25) > 1e-12) return 3;
                return 0;
            }
            """,
        )


def test_next_model_rejects_non_discrete_runtime_integrators():
    model = OptimalControlModel("NextRuntimeMismatchModel")
    x = model.state("x")
    u = model.control("u")

    model.subject_to(Next(x) == x + u)
    model.minimize(x**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir)
        compile_and_run(
            tmpdir,
            "next_runtime_mismatch_check.cpp",
            "next_runtime_mismatch_check",
            """
            #include "nextruntimemismatchmodel.h"
            #include "minisolver/integrator/implicit_integrator.h"
            #include <stdexcept>

            int main() {
                using Model = minisolver::NextRuntimeMismatchModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 2.0;
                kp.u(0) = 3.0;

                try {
                    Model::compute_dynamics(kp, minisolver::IntegratorType::EULER_EXPLICIT, 0.1);
                    return 1;
                } catch (const std::invalid_argument&) {
                }

                minisolver::MSVec<double, Model::NX> x_in;
                minisolver::MSVec<double, Model::NU> u_in;
                minisolver::MSVec<double, Model::NP> p_in;
                x_in(0) = 2.0;
                u_in(0) = 3.0;
                try {
                    (void)Model::integrate(
                        x_in, u_in, p_in, 0.1, minisolver::IntegratorType::RUNGE_KUTTA_4);
                    return 2;
                } catch (const std::invalid_argument&) {
                }

                try {
                    minisolver::detail::dispatch_compute_dynamics<Model>(
                        kp, minisolver::IntegratorType::GAUSS_LEGENDRE_4, 0.1);
                    return 3;
                } catch (const std::invalid_argument&) {
                }
                return 0;
            }
            """,
        )


def test_dot_model_rejects_discrete_runtime_integrator():
    model = OptimalControlModel("DotRuntimeMismatchModel")
    x = model.state("x")
    u = model.control("u")

    model.subject_to(Dot(x) == u)
    model.minimize(x**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "dot_runtime_mismatch_check.cpp",
            "dot_runtime_mismatch_check",
            """
            #include "dotruntimemismatchmodel.h"
            #include <stdexcept>

            int main() {
                using Model = minisolver::DotRuntimeMismatchModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                try {
                    Model::compute_dynamics(kp, minisolver::IntegratorType::DISCRETE, 0.1);
                } catch (const std::invalid_argument&) {
                    return 0;
                }
                return 1;
            }
            """,
        )


def test_generated_implicit_direct_calls_fail_loud_but_dispatch_succeeds():
    model = OptimalControlModel("ImplicitDirectGuardModel")
    x = model.state("x")
    u = model.control("u")

    model.subject_to(Dot(x) == -2.0 * x + u)
    model.minimize(x**2 + u**2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="GAUSS_LEGENDRE_4")
        compile_and_run(
            tmpdir,
            "implicit_direct_guard_check.cpp",
            "implicit_direct_guard_check",
            """
            #include "implicitdirectguardmodel.h"
            #include "minisolver/integrator/implicit_integrator.h"
            #include <cmath>
            #include <stdexcept>

            int main() {
                using Model = minisolver::ImplicitDirectGuardModel;

                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 1.0;
                kp.u(0) = 0.0;

                try {
                    Model::compute_dynamics(
                        kp, minisolver::IntegratorType::GAUSS_LEGENDRE_4, 0.1);
                    return 1;
                } catch (const std::invalid_argument&) {
                }

                minisolver::MSVec<double, Model::NX> x_in;
                minisolver::MSVec<double, Model::NU> u_in;
                minisolver::MSVec<double, Model::NP> p_in;
                x_in(0) = 1.0;
                u_in(0) = 0.0;

                try {
                    (void)Model::integrate(
                        x_in, u_in, p_in, 0.1, minisolver::IntegratorType::GAUSS_LEGENDRE_4);
                    return 2;
                } catch (const std::invalid_argument&) {
                }

                minisolver::detail::dispatch_compute_dynamics<Model>(
                    kp, minisolver::IntegratorType::GAUSS_LEGENDRE_4, 0.1);
                if (!std::isfinite(kp.f_resid(0))) return 3;
                if (!std::isfinite(kp.A(0, 0))) return 4;
                if (!std::isfinite(kp.B(0, 0))) return 5;

                const auto z = minisolver::detail::dispatch_integrate<Model>(
                    x_in, u_in, p_in, 0.1, minisolver::IntegratorType::GAUSS_LEGENDRE_4);
                if (!std::isfinite(z(0))) return 6;

                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> explicit_kp;
                explicit_kp.set_zero();
                explicit_kp.x(0) = 1.0;
                explicit_kp.u(0) = 0.0;
                Model::compute_dynamics(
                    explicit_kp, minisolver::IntegratorType::RUNGE_KUTTA_4, 0.1);
                if (!std::isfinite(explicit_kp.f_resid(0))) return 7;

                return 0;
            }
            """,
        )


def test_dynamics_dsl_rejects_mixed_dot_and_next_modes():
    def mixed_modes():
        model = OptimalControlModel("MixedDynamicsModel")
        x, y = model.state("x", "y")
        u = model.control("u")
        model.subject_to(Dot(x) == u)
        model.subject_to(Next(y) == y + u)

    expect_value_error(mixed_modes, "Cannot mix")


def test_plain_equalities_are_not_supported_as_constraints():
    def plain_equality():
        model = OptimalControlModel("PlainEqualityModel")
        x, y = model.state("x", "y")
        model.subject_to(x == y)

    expect_value_error(plain_equality, "Use Dot/Next")


def test_generate_rejects_integrator_mode_mismatch():
    def dot_with_discrete_integrator():
        model = OptimalControlModel("DotDiscreteMismatchModel")
        x = model.state("x")
        u = model.control("u")
        model.subject_to(Dot(x) == u)
        model.generate(integrator_type="DISCRETE")

    def next_with_continuous_integrator():
        model = OptimalControlModel("NextContinuousMismatchModel")
        x = model.state("x")
        u = model.control("u")
        model.subject_to(Next(x) == x + u)
        model.generate(integrator_type="RUNGE_KUTTA_4")

    expect_value_error(dot_with_discrete_integrator, "DISCRETE")
    expect_value_error(next_with_continuous_integrator, "Next")


def test_set_dynamics_api_is_removed():
    model = OptimalControlModel("NoSetDynamicsModel")
    assert not hasattr(model, "set_dynamics")


def test_model_fingerprint_changes_between_dot_and_next_modes():
    def make_dot_model():
        model = OptimalControlModel("DynamicsFingerprintModel")
        x = model.state("x")
        u = model.control("u")
        model.subject_to(Dot(x) == u)
        model.minimize(x**2 + u**2)
        return model

    def make_next_model():
        model = OptimalControlModel("DynamicsFingerprintModel")
        x = model.state("x")
        u = model.control("u")
        model.subject_to(Next(x) == x + u)
        model.minimize(x**2 + u**2)
        return model

    text_dot = generate_header_text(
        make_dot_model(), "dynamicsfingerprintmodel.h", integrator_type="EULER_EXPLICIT")
    text_next = generate_header_text(
        make_next_model(), "dynamicsfingerprintmodel.h")

    require(text_dot, "dynamics_continuous")
    require(text_next, "discrete dynamics")
    require(text_next, "generated_integrator = IntegratorType::DISCRETE")
    for forbidden_api in ("dynamics_continuous", "jacobian_continuous"):
        if forbidden_api in text_next:
            raise AssertionError(
                f"Next-generated model should not expose {forbidden_api}")
    for forbidden in (
        "EULER_EXPLICIT",
        "EULER_IMPLICIT",
        "RUNGE_KUTTA_2",
        "GAUSS_LEGENDRE_2",
        "RUNGE_KUTTA_4",
        "GAUSS_LEGENDRE_4",
    ):
        if forbidden in text_next:
            raise AssertionError(f"Next-generated model should not contain {forbidden}")
    assert text_dot != text_next


if __name__ == "__main__":
    test_dot_subject_to_generates_continuous_dynamics()
    test_next_subject_to_generates_discrete_dynamics_map()
    test_next_model_rejects_non_discrete_runtime_integrators()
    test_dot_model_rejects_discrete_runtime_integrator()
    test_generated_implicit_direct_calls_fail_loud_but_dispatch_succeeds()
    test_dynamics_dsl_rejects_mixed_dot_and_next_modes()
    test_plain_equalities_are_not_supported_as_constraints()
    test_generate_rejects_integrator_mode_mismatch()
    test_set_dynamics_api_is_removed()
    test_model_fingerprint_changes_between_dot_and_next_modes()
