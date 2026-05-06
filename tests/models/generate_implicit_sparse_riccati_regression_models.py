import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from minisolver.MiniModel import Dot, OptimalControlModel


def build_chain_model(name):
    model = OptimalControlModel(name)

    x0, x1, x2, x3, x4 = model.state("x0", "x1", "x2", "x3", "x4")
    u0, u1 = model.control("u0", "u1")

    model.subject_to(Dot(x0) == u0)
    model.subject_to(Dot(x1) == x0)
    model.subject_to(Dot(x2) == x1)
    model.subject_to(Dot(x3) == x2 + 0.25 * u1)
    model.subject_to(Dot(x4) == x3)

    model.minimize(8.0 * x0**2)
    model.minimize(6.0 * x1**2)
    model.minimize(4.0 * x2**2)
    model.minimize(2.0 * x3**2)
    model.minimize(1.0 * x4**2)
    model.minimize(0.05 * u0**2)
    model.minimize(0.05 * u1**2)

    model.subject_to(x0 - 100.0 <= 0)
    model.subject_to(-100.0 - x0 <= 0)
    return model


def generate_pair(output_dir, integrator_type, fused_name, generic_name):
    build_chain_model(fused_name).generate(
        output_dir, use_fused_riccati=True, integrator_type=integrator_type)
    build_chain_model(generic_name).generate(
        output_dir, use_fused_riccati=False, integrator_type=integrator_type)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tests", "generated")
    os.makedirs(output_dir, exist_ok=True)

    generate_pair(output_dir, "EULER_IMPLICIT",
        "FusedEulerImplicitRegressionModel", "GenericEulerImplicitRegressionModel")
    generate_pair(output_dir, "RK2_IMPLICIT",
        "FusedMidpointImplicitRegressionModel", "GenericMidpointImplicitRegressionModel")
    generate_pair(output_dir, "RK4_IMPLICIT",
        "FusedGaussImplicitRegressionModel", "GenericGaussImplicitRegressionModel")

    print(f"Generated implicit sparse Riccati regression models in {output_dir}")
