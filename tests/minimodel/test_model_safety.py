import re
import sympy as sp

from common import Dot, OptimalControlModel, expect_value_error, generate_header_text, require


def test_rejects_undeclared_symbols_in_model_expressions():
    def undeclared_dynamics():
        model = OptimalControlModel("BadDynamicsModel")
        x = model.state("x")
        u = model.control("u")
        ghost = sp.symbols("ghost")
        model.subject_to(Dot(x) == u + ghost)

    def undeclared_objective():
        model = OptimalControlModel("BadObjectiveModel")
        x = model.state("x")
        ghost = sp.symbols("ghost")
        model.minimize(x + ghost)

    def undeclared_constraint():
        model = OptimalControlModel("BadConstraintModel")
        x = model.state("x")
        ghost = sp.symbols("ghost")
        model.subject_to(x + ghost <= 0)

    expect_value_error(undeclared_dynamics, "undeclared")
    expect_value_error(undeclared_objective, "undeclared")
    expect_value_error(undeclared_constraint, "undeclared")


def test_generate_requires_explicit_dynamics_for_every_state():
    def missing_dynamics():
        model = OptimalControlModel("MissingDynamicsModel")
        x, y = model.state("x", "y")
        u = model.control("u")
        model.subject_to(Dot(x) == u)
        model.minimize(x**2 + y**2)
        model.generate()

    expect_value_error(missing_dynamics, "Missing dynamics")


def test_soft_constraint_weight_validation_is_explicit():
    def negative_weight():
        model = OptimalControlModel("NegativeSoftWeightModel")
        x = model.state("x")
        model.subject_to(x <= 0, weight=-1.0)

    def symbolic_weight():
        model = OptimalControlModel("SymbolicSoftWeightModel")
        x = model.state("x")
        w = model.parameter("w")
        model.subject_to(x <= 0, weight=w)

    expect_value_error(negative_weight, "non-negative")
    expect_value_error(symbolic_weight, "numeric")


def test_quad_constraint_dimension_errors_are_reported_early():
    def center_mismatch():
        model = OptimalControlModel("CenterMismatchModel")
        x0, x1 = model.state("x0", "x1")
        model.subject_to_quad([[1, 0], [0, 1]], [x0, x1], center=[0], rhs=1.0)

    def q_mismatch():
        model = OptimalControlModel("QMismatchModel")
        x0, x1 = model.state("x0", "x1")
        model.subject_to_quad([[1]], [x0, x1], rhs=1.0)

    expect_value_error(center_mismatch, "Dimension mismatch")
    expect_value_error(q_mismatch, "Dimension mismatch")


def test_model_name_validation_rejects_invalid_cpp_type_names():
    def keyword_model_name():
        OptimalControlModel("class")

    def invalid_identifier_model_name():
        OptimalControlModel("bad-model")

    expect_value_error(keyword_model_name, "C++")
    expect_value_error(invalid_identifier_model_name, "valid C++ identifier")


def test_soft_constraint_loss_validation_is_explicit():
    def invalid_loss():
        model = OptimalControlModel("InvalidSoftLossModel")
        x = model.state("x")
        model.subject_to(x <= 0, weight=1.0, loss="Huber")

    expect_value_error(invalid_loss, "L1 or L2")


def _extract_model_fingerprint(header_text):
    match = re.search(
        r"static constexpr std::uint64_t model_fingerprint = (0x[0-9a-f]+)ull;",
        header_text,
    )
    if not match:
        raise AssertionError("generated header is missing model_fingerprint")
    return match.group(1)


def test_model_fingerprint_changes_when_model_equations_change():
    def make_model(dynamics_scale):
        model = OptimalControlModel("FingerprintModel")
        x = model.state("x")
        u = model.control("u")
        model.subject_to(Dot(x) == dynamics_scale * u)
        model.minimize(x**2 + u**2)
        model.subject_to(x - 1.0 <= 0)
        return model

    text_a = generate_header_text(
        make_model(1.0), "fingerprintmodel.h", integrator_type="EULER_EXPLICIT")
    text_b = generate_header_text(
        make_model(2.0), "fingerprintmodel.h", integrator_type="EULER_EXPLICIT")

    require(text_a, "#include <cstdint>")
    assert _extract_model_fingerprint(text_a) != _extract_model_fingerprint(text_b)


if __name__ == "__main__":
    test_rejects_undeclared_symbols_in_model_expressions()
    test_generate_requires_explicit_dynamics_for_every_state()
    test_soft_constraint_weight_validation_is_explicit()
    test_quad_constraint_dimension_errors_are_reported_early()
    test_model_name_validation_rejects_invalid_cpp_type_names()
    test_soft_constraint_loss_validation_is_explicit()
    test_model_fingerprint_changes_when_model_equations_change()
