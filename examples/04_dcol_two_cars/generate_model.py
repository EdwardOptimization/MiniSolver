import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

import sympy as sp
from minisolver.MiniModel import Dot, OptimalControlModel


def add_car_dynamics(model, x, y, theta, v, a, omega):
    model.subject_to(Dot(x) == v * sp.cos(theta))
    model.subject_to(Dot(y) == v * sp.sin(theta))
    model.subject_to(Dot(theta) == omega)
    model.subject_to(Dot(v) == a)


def main():
    model = OptimalControlModel("DcolTwoCarsModel")

    x1, y1, theta1, v1 = model.state("x1", "y1", "theta1", "v1")
    x2, y2, theta2, v2 = model.state("x2", "y2", "theta2", "v2")
    a1, omega1, a2, omega2 = model.control("a1", "omega1", "a2", "omega2")

    x1_ref = model.parameter("x1_ref")
    y1_ref = model.parameter("y1_ref")
    theta1_ref = model.parameter("theta1_ref")
    v1_ref = model.parameter("v1_ref")
    x2_ref = model.parameter("x2_ref")
    y2_ref = model.parameter("y2_ref")
    theta2_ref = model.parameter("theta2_ref")
    v2_ref = model.parameter("v2_ref")

    q_lin = [
        model.parameter("x1_lin"),
        model.parameter("y1_lin"),
        model.parameter("theta1_lin"),
        model.parameter("x2_lin"),
        model.parameter("y2_lin"),
        model.parameter("theta2_lin"),
    ]
    q = [x1, y1, theta1, x2, y2, theta2]

    dcol_alpha = model.parameter("dcol_alpha")
    dcol_grad = [
        model.parameter("dcol_gx1"),
        model.parameter("dcol_gy1"),
        model.parameter("dcol_gtheta1"),
        model.parameter("dcol_gx2"),
        model.parameter("dcol_gy2"),
        model.parameter("dcol_gtheta2"),
    ]
    dcol_hess = []
    for i in range(6):
        row = []
        for j in range(i, 6):
            row.append(model.parameter(f"dcol_h{i}{j}"))
        dcol_hess.append(row)

    add_car_dynamics(model, x1, y1, theta1, v1, a1, omega1)
    add_car_dynamics(model, x2, y2, theta2, v2, a2, omega2)

    dq = [q[i] - q_lin[i] for i in range(6)]
    dcol_alpha_local = dcol_alpha
    for i in range(6):
        dcol_alpha_local += dcol_grad[i] * dq[i]
    quadratic = 0
    for i in range(6):
        for offset, hij in enumerate(dcol_hess[i]):
            j = i + offset
            scale = 1 if i == j else 2
            quadratic += scale * hij * dq[i] * dq[j]
    dcol_alpha_local += sp.Rational(1, 2) * quadratic

    model.minimize(
        0.08 * (x1 - x1_ref) ** 2
        + 0.002 * (y1 - y1_ref) ** 2
        + 0.45 * (theta1 - theta1_ref) ** 2
        + 0.30 * (v1 - v1_ref) ** 2
        + 0.08 * (x2 - x2_ref) ** 2
        + 0.002 * (y2 - y2_ref) ** 2
        + 0.45 * (theta2 - theta2_ref) ** 2
        + 0.30 * (v2 - v2_ref) ** 2
        + 0.04 * (a1**2 + a2**2)
        + 0.06 * (omega1**2 + omega2**2)
    )

    model.subject_to(1.0 - dcol_alpha_local <= 0)

    output_dir = os.path.join(os.path.dirname(__file__), "generated")
    model.generate(output_dir)
    print(f"Generated DcolTwoCarsModel in {output_dir}")

    inner = OptimalControlModel("InnerDcolNorm2Model")
    dummy = inner.state("dummy")
    p1x, p1y, p2x, p2y, alpha = inner.control("p1x", "p1y", "p2x", "p2y", "alpha")
    x1 = inner.parameter("x1")
    y1 = inner.parameter("y1")
    theta1 = inner.parameter("theta1")
    x2 = inner.parameter("x2")
    y2 = inner.parameter("y2")
    theta2 = inner.parameter("theta2")

    inner.subject_to(Dot(dummy) == 0)

    length = 4.5
    width = 1.9
    hx = 0.5 * length
    hy = 0.5 * width
    min_distance = 1.0

    def add_rectangle_point_constraints(px, py, cx, cy, theta):
        c = sp.cos(theta)
        s = sp.sin(theta)
        dx = px - cx
        dy = py - cy
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        inner.subject_to(local_x - alpha * hx <= 0, include_terminal=False)
        inner.subject_to(-local_x - alpha * hx <= 0, include_terminal=False)
        inner.subject_to(local_y - alpha * hy <= 0, include_terminal=False)
        inner.subject_to(-local_y - alpha * hy <= 0, include_terminal=False)

    add_rectangle_point_constraints(p1x, p1y, x1, y1, theta1)
    add_rectangle_point_constraints(p2x, p2y, x2, y2, theta2)
    inner.subject_to_quad(
        [[1, 0], [0, 1]],
        [p1x - p2x, p1y - p2y],
        rhs=alpha * min_distance,
        rhs_mode="norm2",
        type="inside",
        include_terminal=False,
    )
    inner.subject_to(-alpha <= 0, include_terminal=False)
    inner.minimize(alpha + 1.0e-8 * (p1x**2 + p1y**2 + p2x**2 + p2y**2 + alpha**2))
    inner.generate(output_dir, integrator_type="EULER_EXPLICIT")
    print(f"Generated InnerDcolNorm2Model in {output_dir}")


if __name__ == "__main__":
    main()
