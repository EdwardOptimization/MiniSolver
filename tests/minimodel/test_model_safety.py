import sympy as sp

from common import OptimalControlModel, expect_value_error


def test_rejects_undeclared_symbols_in_model_expressions():
    def undeclared_dynamics():
        model = OptimalControlModel("BadDynamicsModel")
        x = model.state("x")
        u = model.control("u")
        ghost = sp.symbols("ghost")
        model.set_dynamics(x, u + ghost)

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
        model.set_dynamics(x, u)
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


if __name__ == "__main__":
    test_rejects_undeclared_symbols_in_model_expressions()
    test_generate_requires_explicit_dynamics_for_every_state()
    test_soft_constraint_weight_validation_is_explicit()
    test_quad_constraint_dimension_errors_are_reported_early()
