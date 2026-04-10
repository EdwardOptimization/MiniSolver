import math
import os
from dataclasses import dataclass

import casadi as ca


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_HEADER = os.path.join(THIS_DIR, "asset_regression_reference_data.h")


def wrap_angle(x):
    return ca.atan2(ca.sin(x), ca.cos(x))


def setup_straight_track(stage_offset, horizon, dt):
    refs = []
    ref_speed = 2.0
    track_half_width = 1.0
    for k in range(horizon + 1):
        t = (stage_offset + k) * dt
        refs.append(
            {
                "x_ref": ref_speed * t,
                "y_ref": 0.0,
                "psi_ref": 0.0,
                "v_ref": ref_speed,
                "n_x": 0.0,
                "n_y": 1.0,
                "w_left": track_half_width,
                "w_right": track_half_width,
            }
        )
    return refs


def setup_curved_track(stage_offset, horizon, dt):
    refs = []
    ref_speed = 1.8
    amp = 0.28
    freq = 0.55
    for k in range(horizon + 1):
        t = (stage_offset + k) * dt
        x_ref = ref_speed * t
        y_ref = amp * math.sin(freq * x_ref)
        dy_dx = amp * freq * math.cos(freq * x_ref)
        psi_ref = math.atan(dy_dx)
        width = 0.55 - 0.08 * math.sin(0.35 * x_ref)
        refs.append(
            {
                "x_ref": x_ref,
                "y_ref": y_ref,
                "psi_ref": psi_ref,
                "v_ref": ref_speed,
                "n_x": -math.sin(psi_ref),
                "n_y": math.cos(psi_ref),
                "w_left": width,
                "w_right": width,
            }
        )
    return refs


def setup_3d_reference(horizon, dt, phase=0.0, amplitude_scale=1.0):
    refs = []
    for k in range(horizon + 1):
        t = phase + k * dt
        refs.append(
            {
                "x_ref": 0.5 + 0.35 * t,
                "y_ref": amplitude_scale * 0.15 * math.sin(0.9 * t),
                "z_ref": 1.0 + amplitude_scale * 0.12 * math.cos(0.7 * t),
                "vx_ref": 0.35,
                "vy_ref": amplitude_scale * 0.15 * 0.9 * math.cos(0.9 * t),
                "vz_ref": -amplitude_scale * 0.12 * 0.7 * math.sin(0.7 * t),
            }
        )
    return refs


def solve_kinematic_bicycle(initial_state, refs, dt):
    nx, nu = 5, 2
    horizon = len(refs) - 1
    wheelbase = 0.33

    X = [ca.MX.sym(f"X_{k}", nx) for k in range(horizon + 1)]
    U = [ca.MX.sym(f"U_{k}", nu) for k in range(horizon + 1)]
    w = ca.vertcat(*X, *U)

    g = [X[0] - ca.DM(initial_state)]
    obj = 0

    def f(x, u):
        return ca.vertcat(
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            x[3] * ca.tan(x[4]) / wheelbase,
            u[0],
            u[1],
        )

    for k in range(horizon + 1):
        r = refs[k]
        xk = X[k]
        uk = U[k]
        lateral = r["n_x"] * (xk[0] - r["x_ref"]) + r["n_y"] * (xk[1] - r["y_ref"])
        psi_error = wrap_angle(xk[2] - r["psi_ref"])
        obj += 10.0 * (xk[0] - r["x_ref"]) ** 2
        obj += 10.0 * (xk[1] - r["y_ref"]) ** 2
        obj += 2.5 * psi_error**2
        obj += 1.0 * (xk[3] - r["v_ref"]) ** 2
        obj += 0.25 * xk[4] ** 2
        obj += 0.025 * uk[0] ** 2
        obj += 0.025 * uk[1] ** 2
        g.extend(
            [
                lateral - r["w_left"],
                -lateral - r["w_right"],
                xk[3] - 8.0,
                0.1 - xk[3],
                xk[4] - 0.5,
                -0.5 - xk[4],
                uk[0] - 4.0,
                -4.0 - uk[0],
                uk[1] - 2.5,
                -2.5 - uk[1],
            ]
        )
        if k < horizon:
            k1 = f(xk, uk)
            k2 = f(xk + 0.5 * dt * k1, uk)
            g.append(X[k + 1] - (xk + dt * k2))

    nlp = {"x": w, "f": obj, "g": ca.vertcat(*g)}
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-10,
        "ipopt.acceptable_tol": 1e-10,
        "ipopt.max_iter": 3000,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    x0 = []
    for r in refs:
        x0.extend([r["x_ref"], r["y_ref"], r["psi_ref"], max(0.5, r["v_ref"]), 0.0])
    for _ in refs:
        x0.extend([0.0, 0.0])

    lbg = []
    ubg = []
    lbg.extend([0.0] * nx)
    ubg.extend([0.0] * nx)
    for k in range(horizon + 1):
        lbg.extend([-ca.inf] * 10)
        ubg.extend([0.0] * 10)
        if k < horizon:
            lbg.extend([0.0] * nx)
            ubg.extend([0.0] * nx)

    sol = solver(x0=ca.DM(x0), lbg=ca.DM(lbg), ubg=ca.DM(ubg))
    vec = sol["x"].full().ravel()
    states = [vec[k * nx : (k + 1) * nx].tolist() for k in range(horizon + 1)]
    controls_offset = (horizon + 1) * nx
    controls = [
        vec[controls_offset + k * nu : controls_offset + (k + 1) * nu].tolist()
        for k in range(horizon + 1)
    ]
    return {
        "states": states,
        "controls": controls,
        "objective": float(sol["f"]),
    }


def solve_double_integrator(initial_state, refs, dt):
    nx, nu = 6, 3
    horizon = len(refs) - 1

    X = [ca.MX.sym(f"X_{k}", nx) for k in range(horizon + 1)]
    U = [ca.MX.sym(f"U_{k}", nu) for k in range(horizon + 1)]
    w = ca.vertcat(*X, *U)

    g = [X[0] - ca.DM(initial_state)]
    obj = 0

    def f(x, u):
        return ca.vertcat(x[3], x[4], x[5], u[0], u[1], u[2])

    for k in range(horizon + 1):
        r = refs[k]
        xk = X[k]
        uk = U[k]
        obj += 15.0 * (xk[0] - r["x_ref"]) ** 2
        obj += 15.0 * (xk[1] - r["y_ref"]) ** 2
        obj += 15.0 * (xk[2] - r["z_ref"]) ** 2
        obj += 2.5 * (xk[3] - r["vx_ref"]) ** 2
        obj += 2.5 * (xk[4] - r["vy_ref"]) ** 2
        obj += 2.5 * (xk[5] - r["vz_ref"]) ** 2
        obj += 0.05 * uk[0] ** 2
        obj += 0.05 * uk[1] ** 2
        obj += 0.05 * uk[2] ** 2
        g.extend(
            [
                uk[0] - 12.0,
                -12.0 - uk[0],
                uk[1] - 12.0,
                -12.0 - uk[1],
                uk[2] - 12.0,
                -12.0 - uk[2],
            ]
        )
        if k < horizon:
            k1 = f(xk, uk)
            k2 = f(xk + 0.5 * dt * k1, uk)
            g.append(X[k + 1] - (xk + dt * k2))

    nlp = {"x": w, "f": obj, "g": ca.vertcat(*g)}
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-10,
        "ipopt.acceptable_tol": 1e-10,
        "ipopt.max_iter": 3000,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    x0 = []
    for r in refs:
        x0.extend([r["x_ref"], r["y_ref"], r["z_ref"], r["vx_ref"], r["vy_ref"], r["vz_ref"]])
    for _ in refs:
        x0.extend([0.0, 0.0, 0.0])

    lbg = []
    ubg = []
    lbg.extend([0.0] * nx)
    ubg.extend([0.0] * nx)
    for k in range(horizon + 1):
        lbg.extend([-ca.inf] * 6)
        ubg.extend([0.0] * 6)
        if k < horizon:
            lbg.extend([0.0] * nx)
            ubg.extend([0.0] * nx)

    sol = solver(x0=ca.DM(x0), lbg=ca.DM(lbg), ubg=ca.DM(ubg))
    vec = sol["x"].full().ravel()
    states = [vec[k * nx : (k + 1) * nx].tolist() for k in range(horizon + 1)]
    controls_offset = (horizon + 1) * nx
    controls = [
        vec[controls_offset + k * nu : controls_offset + (k + 1) * nu].tolist()
        for k in range(horizon + 1)
    ]
    return {
        "states": states,
        "controls": controls,
        "objective": float(sol["f"]),
    }


@dataclass
class ReferenceCase:
    name: str
    objective: float
    terminal_state: list
    first_control: list


def solve_curved_closed_loop():
    dt = 0.1
    horizon = 14
    mpc_steps = 6
    sim_state = [0.0, 0.24, 0.18, 1.0, 0.02]
    wheelbase = 0.33
    for step in range(mpc_steps):
        refs = setup_curved_track(float(step), horizon, dt)
        sol = solve_kinematic_bicycle(sim_state, refs, dt)
        accel, delta_rate = sol["controls"][0]
        sim_x, sim_y, sim_psi, sim_v, sim_delta = sim_state
        sim_x += sim_v * math.cos(sim_psi) * dt
        sim_y += sim_v * math.sin(sim_psi) * dt
        sim_psi += sim_v * math.tan(sim_delta) / wheelbase * dt
        sim_v += accel * dt
        sim_delta += delta_rate * dt
        sim_delta = max(-0.5, min(0.5, sim_delta))
        sim_state = [sim_x, sim_y, sim_psi, sim_v, sim_delta]
    return sim_state


def as_cpp_array(values):
    return ", ".join(f"{v:.17g}" for v in values)


def main():
    import sys

    output_header = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_HEADER

    straight = solve_kinematic_bicycle(
        initial_state=[0.0, 0.35, 0.12, 0.8, 0.0],
        refs=setup_straight_track(0.0, 12, 0.1),
        dt=0.1,
    )
    simple_3d = solve_double_integrator(
        initial_state=[-0.35, 0.25, 0.75, 0.0, 0.0, 0.0],
        refs=setup_3d_reference(15, 0.1, 0.0, 1.0),
        dt=0.1,
    )
    shifted_3d = solve_double_integrator(
        initial_state=[-0.20, 0.28, 0.80, 0.04, -0.01, 0.02],
        refs=setup_3d_reference(16, 0.1, 0.15, 1.15),
        dt=0.1,
    )
    curved_final = solve_curved_closed_loop()

    os.makedirs(os.path.dirname(output_header), exist_ok=True)
    with open(output_header, "w", encoding="ascii") as f:
        f.write("#pragma once\n")
        f.write("#include <array>\n\n")
        f.write("namespace minisolver::testdata {\n\n")
        f.write("struct OpenLoopReference5x2 {\n")
        f.write("    double objective;\n")
        f.write("    std::array<double, 5> terminal_state;\n")
        f.write("    std::array<double, 2> first_control;\n")
        f.write("};\n\n")
        f.write("struct OpenLoopReference6x3 {\n")
        f.write("    double objective;\n")
        f.write("    std::array<double, 6> terminal_state;\n")
        f.write("    std::array<double, 3> first_control;\n")
        f.write("};\n\n")
        f.write("inline constexpr OpenLoopReference5x2 kKinematicBicycleStraightReference{\n")
        f.write(f"    {straight['objective']:.17g},\n")
        f.write(f"    {{{as_cpp_array(straight['states'][-1])}}},\n")
        f.write(f"    {{{as_cpp_array(straight['controls'][0])}}},\n")
        f.write("};\n\n")
        f.write("inline constexpr std::array<double, 5> kKinematicBicycleCurvedClosedLoopFinalState{\n")
        f.write(f"    {as_cpp_array(curved_final)}\n")
        f.write("};\n\n")
        f.write("inline constexpr OpenLoopReference6x3 kDoubleIntegrator3DTrackingReference{\n")
        f.write(f"    {simple_3d['objective']:.17g},\n")
        f.write(f"    {{{as_cpp_array(simple_3d['states'][-1])}}},\n")
        f.write(f"    {{{as_cpp_array(simple_3d['controls'][0])}}},\n")
        f.write("};\n\n")
        f.write("inline constexpr OpenLoopReference6x3 kDoubleIntegrator3DShiftedReference{\n")
        f.write(f"    {shifted_3d['objective']:.17g},\n")
        f.write(f"    {{{as_cpp_array(shifted_3d['states'][-1])}}},\n")
        f.write(f"    {{{as_cpp_array(shifted_3d['controls'][0])}}},\n")
        f.write("};\n\n")
        f.write("}  // namespace minisolver::testdata\n")

    print(f"Wrote {output_header}")


if __name__ == "__main__":
    main()
