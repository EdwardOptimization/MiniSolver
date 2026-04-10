import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from minisolver.MiniModel import OptimalControlModel


ACC_MAX = 12.0


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tests", "generated")
    os.makedirs(output_dir, exist_ok=True)

    model = OptimalControlModel("DoubleIntegrator3DRegressionModel")

    x, y, z, vx, vy, vz = model.state("x", "y", "z", "vx", "vy", "vz")
    ax, ay, az = model.control("ax", "ay", "az")

    x_ref = model.parameter("x_ref")
    y_ref = model.parameter("y_ref")
    z_ref = model.parameter("z_ref")
    vx_ref = model.parameter("vx_ref")
    vy_ref = model.parameter("vy_ref")
    vz_ref = model.parameter("vz_ref")

    model.set_dynamics(x, vx)
    model.set_dynamics(y, vy)
    model.set_dynamics(z, vz)
    model.set_dynamics(vx, ax)
    model.set_dynamics(vy, ay)
    model.set_dynamics(vz, az)

    model.minimize(15.0 * (x - x_ref) ** 2)
    model.minimize(15.0 * (y - y_ref) ** 2)
    model.minimize(15.0 * (z - z_ref) ** 2)
    model.minimize(2.5 * (vx - vx_ref) ** 2)
    model.minimize(2.5 * (vy - vy_ref) ** 2)
    model.minimize(2.5 * (vz - vz_ref) ** 2)
    model.minimize(0.05 * ax ** 2)
    model.minimize(0.05 * ay ** 2)
    model.minimize(0.05 * az ** 2)

    model.subject_to(ax - ACC_MAX <= 0)
    model.subject_to(-ACC_MAX - ax <= 0)
    model.subject_to(ay - ACC_MAX <= 0)
    model.subject_to(-ACC_MAX - ay <= 0)
    model.subject_to(az - ACC_MAX <= 0)
    model.subject_to(-ACC_MAX - az <= 0)

    model.generate(output_dir)
    print(f"Generated DoubleIntegrator3DRegressionModel in {output_dir}")
