import os
import sys
import tempfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from minisolver.MiniModel import OptimalControlModel


def require(text, needle):
    if needle not in text:
        raise AssertionError(f"missing generated snippet: {needle}")


def reject(text, needle):
    if needle in text:
        raise AssertionError(f"unexpected generated snippet: {needle}")


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


if __name__ == "__main__":
    test_implicit_riccati_pattern_keeps_inverse_fill_in()
