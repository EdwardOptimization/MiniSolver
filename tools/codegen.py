import sympy as sp
import os

def generate_header():
    # --- 1. Define Variables ---
    x = sp.Matrix([sp.symbols(f'x{i}') for i in range(4)])
    u = sp.Matrix([sp.symbols(f'u{i}') for i in range(2)])
    p = sp.Matrix([sp.symbols(f'p{i}') for i in range(6)])
    
    dt_sym = sp.symbols('dt')

    px, py, theta, v = x[0], x[1], x[2], x[3]
    acc, steer = u[0], u[1]

    # --- 2. Continuous Dynamics ---
    L = 2.5
    f_cont = sp.Matrix([
        v * sp.cos(theta),
        v * sp.sin(theta),
        (v / L) * sp.tan(steer),
        acc
    ])

    # --- 3. RK4 Integration ---
    def get_f(state, ctrl):
        subs_map = {x[i]: state[i] for i in range(4)}
        return f_cont.subs(subs_map)

    k1 = f_cont
    k2 = get_f(x + 0.5 * dt_sym * k1, u)
    k3 = get_f(x + 0.5 * dt_sym * k2, u)
    k4 = get_f(x + dt_sym * k3, u)

    x_next = x + (dt_sym / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # --- 4. Analytical Derivatives ---
    A_expr = x_next.jacobian(x)
    B_expr = x_next.jacobian(u)

    # --- 5. Cost Function ---
    v_ref, x_ref, y_ref = p[0], p[1], p[2]
    
    cost_track = 1.0 * (px - x_ref) ** 2 + 1.0 * (py - y_ref) ** 2 + 0.1 * theta**2 + 1.0 * (v - v_ref) ** 2
    cost_control = 0.1 * acc ** 2 + 1.0 * steer ** 2
    total_cost = cost_track + cost_control

    xu = sp.Matrix.vstack(x, u)
    hessian = sp.hessian(total_cost, xu)
    Q = hessian[:4, :4]
    R = hessian[4:, 4:]
    H = hessian[4:, :4]
    q_grad = sp.Matrix([total_cost]).jacobian(x).T
    r_grad = sp.Matrix([total_cost]).jacobian(u).T

    # --- 6. Constraints ---
    obs_x, obs_y, obs_rad = p[3], p[4], p[5]
    car_rad = 1.0
    min_dist_sq = (obs_rad + car_rad)**2
    dist_sq = (px - obs_x)**2 + (py - obs_y)**2

    g_expr = sp.Matrix([
        acc - 3.0,
        -acc - 3.0,
        steer - 0.5,
        -steer - 0.5,
        min_dist_sq - dist_sq
    ])

    C_expr = g_expr.jacobian(x)
    D_expr = g_expr.jacobian(u)

    # --- 7. Generate C++ Code ---
    replacements, reduced = sp.cse([
        x_next, A_expr, B_expr,
        q_grad, r_grad, Q, R, H,
        g_expr, C_expr, D_expr,
        total_cost 
    ])

    path = "include/model/car_model.h"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as file:
        file.write(f"""#pragma once
#include "core/types.h"
#include "core/solver_options.h"
#include "core/matrix_defs.h"
#include <cmath>

namespace minisolver {{

struct CarModel {{
    static const int NX=4; 
    static const int NU=2; 
    static const int NC=5; 
    static const int NP=6;

    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& u) 
    {{
        T x2=x(2); T x3=x(3);
        T u0=u(0); T u1=u(1);
        MSVec<T, NX> xdot;
        xdot(0) = x3 * cos(x2);
        xdot(1) = x3 * sin(x2);
        xdot(2) = (x3 / 2.5) * tan(u1);
        xdot(3) = u0;
        return xdot;
    }}

    // --- Integrator Interface ---
    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& u,
        double dt,
        IntegratorType type)
    {{
        switch(type) {{
            case IntegratorType::EULER_EXPLICIT: return x + dynamics_continuous(x, u) * dt;
            default: // RK4 Explicit
            {{
               auto k1 = dynamics_continuous(x, u);
               auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u);
               auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u);
               auto k4 = dynamics_continuous<T>(x + k3 * dt, u);
               return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }}
        }}
    }}

    // --- Main Compute Function ---
    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {{
        T x0=kp.x(0); T x1=kp.x(1); T x2=kp.x(2); T x3=kp.x(3);
        T u0=kp.u(0); T u1=kp.u(1);
        T p0=kp.p(0); T p1=kp.p(1); T p2=kp.p(2); T p3=kp.p(3); T p4=kp.p(4); T p5=kp.p(5);

""")
        for name, val in replacements:
            file.write(f"        T {name} = {sp.ccode(val)};\n")
        
        # Map: 0=x_next, 1=A, 2=B, 3=q, 4=r, 5=Q, 6=R, 7=H, 8=g, 9=C, 10=D, 11=cost
        assignments = [
            ("f_resid", 0, 4, 1), 
            ("A", 1, 4, 4), ("B", 2, 4, 2),
            ("q", 3, 4, 1), ("r", 4, 2, 1),
            ("Q", 5, 4, 4), ("R", 6, 2, 2), ("H", 7, 2, 4),
            ("g_val", 8, 5, 1), ("C", 9, 5, 4), ("D", 10, 5, 2)
        ]

        for name, idx, rows, cols in assignments:
            mat = reduced[idx]
            file.write(f"        // {name}\n")
            for r in range(rows):
                for c in range(cols):
                    if rows == 1 or cols == 1:
                        val = mat[r] if rows > 1 else mat[c]
                    else:
                        val = mat[r, c]
                    
                    if val != 0:
                        file.write(f"        kp.{name}({r},{c}) = {sp.ccode(val)};\n")
                    else:
                        file.write(f"        kp.{name}({r},{c}) = 0;\n")
        
        # Cost assignment
        cost_val = reduced[11]
        file.write(f"        kp.cost = {sp.ccode(cost_val)};\n")

        file.write("    }\n};\n}\n")

if __name__ == "__main__":
    generate_header()
    print("Generated MSMat-compatible Model.")
