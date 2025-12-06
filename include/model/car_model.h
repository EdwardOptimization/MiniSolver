#pragma once
#include "core/types.h"
#include <cmath>

namespace roboopt {

struct CarModel {
    // NX=4 (x, y, theta, v)
    // NU=2 (accel, steer)
    // NC=4 (Control limits: min/max for acc and steer)
    // NP=6 (Target parameters)
    static const int NX = 4;
    static const int NU = 2;
    static const int NC = 4;
    static const int NP = 6;

    // Parameters map:
    // p[0] = v_target
    // p[1] = x_ref
    // p[2] = y_ref
    // p[3] = obs_x
    // p[4] = obs_y
    // p[5] = obs_weight

    template<typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp) {
        // --- 1. Unpack Variables ---
        T px = kp.x(0);
        T py = kp.x(1);
        T th = kp.x(2);
        T v  = kp.x(3);

        T acc = kp.u(0);
        T delta = kp.u(1);

        T dt = 0.1;
        T L  = 2.5; // Wheelbase

        // --- 2. Dynamics & Derivatives (Kinematic Bicycle) ---
        // f(x,u) = [x + v*cos(th)*dt, y + v*sin(th)*dt, th + v/L*tan(delta)*dt, v + acc*dt]
        // Note: The solver expects Linearized Dynamics: x_{k+1} = A x_k + B u_k + f_resid
        // So f_resid = f(x_linear, u_linear) - A x_linear - B u_linear roughly,
        // OR standardly: f_resid = f(x,u). The solver usually does A dx + B du + f(x,u) - x_{k+1} = 0.
        // Let's assume standard DDP convention:
        // A = df/dx, B = df/du at the current linearization point.
        // f_resid = f(current_x, current_u) - x_current (Wait, usually f_resid is the gap to the NEXT state in shooting)
        // Let's stick to: A, B are Jacobians. f_resid is the explicit next state f(x,u).
        // (The solver will compute defect d = f_resid - x_next)

        T c_th = cos(th);
        T s_th = sin(th);
        T tan_d = tan(delta);
        T sec2_d = 1.0 / (cos(delta) * cos(delta));

        // Next State
        kp.f_resid(0) = px + v * c_th * dt;
        kp.f_resid(1) = py + v * s_th * dt;
        kp.f_resid(2) = th + (v / L) * tan_d * dt;
        kp.f_resid(3) = v + acc * dt;

        // A Matrix (df/dx)
        kp.A.setIdentity();
        kp.A(0, 2) = -v * s_th * dt; kp.A(0, 3) = c_th * dt;
        kp.A(1, 2) =  v * c_th * dt; kp.A(1, 3) = s_th * dt;
        kp.A(2, 3) = (tan_d / L) * dt;

        // B Matrix (df/du)
        kp.B.setZero();
        kp.B(2, 1) = (v / L) * sec2_d * dt;
        kp.B(3, 0) = dt;

        // --- 3. Costs (Quadratic Approximation) ---
        T v_target = kp.p(0);
        T x_ref    = kp.p(1);
        T y_ref    = kp.p(2);
        T obs_x    = kp.p(3);
        T obs_y    = kp.p(4);
        T obs_w    = kp.p(5);

        // State Cost: W_x * (x - x_ref)^2 + ...
        // We accumulate into Q, q.
        // Reset Cost
        kp.Q.setZero(); kp.q.setZero();
        kp.R.setZero(); kp.r.setZero();
        kp.H.setZero();

        // 3.1 Tracking Cost
        T w_pos = 1.0;
        T w_vel = 1.0;
        T w_ang = 0.1;

        // (px - x_ref)^2
        kp.Q(0,0) += w_pos; kp.q(0) += w_pos * (px - x_ref);
        // (py - y_ref)^2
        kp.Q(1,1) += w_pos; kp.q(1) += w_pos * (py - y_ref);
        // th^2 (penalize heading deviation from 0)
        kp.Q(2,2) += w_ang; kp.q(2) += w_ang * th;
        // (v - v_target)^2
        kp.Q(3,3) += w_vel; kp.q(3) += w_vel * (v - v_target);

        // 3.2 Control Cost
        T w_acc = 0.1;
        T w_steer = 1.0;
        kp.R(0,0) += w_acc;   kp.r(0) += w_acc * acc;
        kp.R(1,1) += w_steer; kp.r(1) += w_steer * delta;

        // 3.3 Obstacle Avoidance (Soft Cost -> Gaussian)
        // Cost = W * exp( -((x-ox)^2 + (y-oy)^2) / sigma )
        // Let's rely on Hard Constraints for Obstacles in advanced usage,
        // but for now, keep the soft cost to help the solver not get stuck locally.
        T dx = px - obs_x;
        T dy = py - obs_y;
        T dist2 = dx*dx + dy*dy;
        T sigma = 4.0; 
        T exp_val = exp(-dist2 / sigma);
        T cost_obs = obs_w * exp_val;

        // Gradient of Obstacle Cost
        // dC/dx = cost_obs * (-2*dx/sigma)
        T grad_factor = cost_obs * (-2.0 / sigma);
        T g_ox = grad_factor * dx;
        T g_oy = grad_factor * dy;

        kp.q(0) += g_ox;
        kp.q(1) += g_oy;

        // Hessian of Obstacle Cost (Gauss-Newton approx or Full Hessian)
        // Approximate: Q += grad * grad^T (Positive Semi-Definite)
        // Or analytical. Let's do simple diagonal approx to keep it convex-ish locally
        // Or just trust the soft cost Gradient to push it away.
        // Let's add a small Hessian term to stabilize
        kp.Q(0,0) += abs(g_ox); // Heuristic
        kp.Q(1,1) += abs(g_oy);

        // --- 4. Constraints (Inequality) ---
        // g(x,u) <= 0
        // 0: acc - 3.0 <= 0
        // 1: -acc - 3.0 <= 0  => acc >= -3.0
        // 2: delta - 0.5 <= 0
        // 3: -delta - 0.5 <= 0

        T max_acc = 3.0;
        T max_steer = 0.5;

        // Values
        kp.g_val(0) = acc - max_acc;
        kp.g_val(1) = -acc - max_acc;
        kp.g_val(2) = delta - max_steer;
        kp.g_val(3) = -delta - max_steer;

        // Jacobians (C = dg/dx, D = dg/du)
        kp.C.setZero();
        kp.D.setZero();

        // 0: d/du0 = 1
        kp.D(0, 0) = 1.0;
        // 1: d/du0 = -1
        kp.D(1, 0) = -1.0;
        // 2: d/du1 = 1
        kp.D(2, 1) = 1.0;
        // 3: d/du1 = -1
        kp.D(3, 1) = -1.0;
    }
};

}
