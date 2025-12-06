import sympy as sp
import os


def generate_header():
    # --- 1. Define Variables ---
    # State: [x, y, theta, v]
    x = sp.Matrix([sp.symbols(f'x{i}') for i in range(4)])
    # Control: [accel, steer]
    u = sp.Matrix([sp.symbols(f'u{i}') for i in range(2)])

    # Parameters: [v_target, x_ref, y_ref, obs_x, obs_y, obs_weight]
    p = sp.Matrix([sp.symbols(f'p{i}') for i in range(6)])

    # --- 2. Dynamics (Kinematic Bicycle) ---
    dt = 0.1
    px, py, theta, v = x[0], x[1], x[2], x[3]
    acc, steer = u[0], u[1]

    f_expr = sp.Matrix([
        px + v * sp.cos(theta) * dt,
        py + v * sp.sin(theta) * dt,
        theta + v * sp.tan(steer) * dt,
        v + acc * dt
    ])

    # --- 3. Cost Function ---
    v_ref, x_ref, y_ref = p[0], p[1], p[2]
    obs_x, obs_y, obs_w = p[3], p[4], p[5]

    # Costs
    cost_track = 1.0 * (px - x_ref) ** 2 + 1.0 * (py - y_ref) ** 2 + 0.5 * (v - v_ref) ** 2
    cost_control = 0.1 * acc ** 2 + 1.0 * steer ** 2

    # Obstacle Cost (Exponential Barrier)
    dist_sq = (px - obs_x) ** 2 + (py - obs_y) ** 2
    cost_obs = obs_w * sp.exp(-dist_sq / 4.0)

    total_cost = cost_track + cost_control + cost_obs

    # --- 4. Derivatives ---
    A_expr = f_expr.jacobian(x)
    B_expr = f_expr.jacobian(u)

    xu = sp.Matrix.vstack(x, u)
    hessian = sp.hessian(total_cost, xu)

    Q = hessian[:4, :4]
    R = hessian[4:, 4:]
    H = hessian[4:, :4]
    q_grad = sp.Matrix([total_cost]).jacobian(x).T
    r_grad = sp.Matrix([total_cost]).jacobian(u).T

    # --- 5. Generate C++ ---
    path = "include/model/car_model.h"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as file:
        file.write("#pragma once\n#include \"core/types.h\"\n#include <cmath>\n")
        file.write("namespace roboopt {\nstruct CarModel {\n")

        # === FIX IS HERE: Added NC=0 ===
        file.write("    static const int NX=4; static const int NU=2; static const int NC=0; static const int NP=6;\n")

        # === FIX IS HERE: Added 0 to template arguments ===
        file.write("    template<typename T> static void compute(KnotPoint<T,4,2,0,6>& kp) {\n")

        # Unpack
        for i in range(4): file.write(f"        T x{i} = kp.x({i});\n")
        for i in range(2): file.write(f"        T u{i} = kp.u({i});\n")
        for i in range(6): file.write(f"        T p{i} = kp.p({i});\n")

        # CSE
        replacements, reduced = sp.cse([A_expr, B_expr, f_expr, q_grad, r_grad, Q, R, H])
        for name, val in replacements:
            file.write(f"        T {name} = {sp.ccode(val)};\n")

        # Assignments
        targets = [("A", 4, 4), ("B", 4, 2), ("f_resid", 4, 1),
                   ("q", 4, 1), ("r", 2, 1), ("Q", 4, 4), ("R", 2, 2), ("H", 2, 4)]

        idx = 0
        for name, rows, cols in targets:
            mat = reduced[idx];
            idx += 1
            for r in range(rows):
                for c in range(cols):
                    if rows == 1 or cols == 1:
                        val = mat[r] if rows > 1 else mat[c]
                    else:
                        val = mat[r, c]
                    if val != 0: file.write(f"        kp.{name}({r},{c}) = {sp.ccode(val)};\n")

        file.write("    }\n};\n}")


if __name__ == "__main__":
    generate_header()
    print("Generated include/model/car_model.h with FIXED template arguments (NC=0).")