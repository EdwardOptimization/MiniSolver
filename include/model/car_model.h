#pragma once
#include "core/types.h"
#include <cmath>
#include <Eigen/Dense>

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

    // --- Continuous Dynamics: x_dot = f(x, u) ---
    template<typename T>
    static Eigen::Matrix<T, NX, 1> dynamics_continuous(
        const Eigen::Matrix<T, NX, 1>& x,
        const Eigen::Matrix<T, NU, 1>& u) 
    {
        T th = x(2);
        T v  = x(3);
        T acc = u(0);
        T delta = u(1);
        T L = 2.5; // Wheelbase

        Eigen::Matrix<T, NX, 1> xdot;
        xdot(0) = v * cos(th);
        xdot(1) = v * sin(th);
        xdot(2) = (v / L) * tan(delta);
        xdot(3) = acc;

        return xdot;
    }

    // --- Integrator ---
    template<typename T>
    static Eigen::Matrix<T, NX, 1> integrate(
        const Eigen::Matrix<T, NX, 1>& x,
        const Eigen::Matrix<T, NU, 1>& u,
        double dt,
        IntegratorType type)
    {
        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
                return x + dynamics_continuous(x, u) * dt;

            case IntegratorType::RK2_EXPLICIT: {
                // Heun's Method (Explicit Trapezoidal)
                auto k1 = dynamics_continuous(x, u);
                auto k2 = dynamics_continuous<T>(x + k1 * dt, u);
                return x + (k1 + k2) * 0.5 * dt;
            }

            case IntegratorType::RK4_EXPLICIT: {
                auto k1 = dynamics_continuous(x, u);
                auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u);
                auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u);
                auto k4 = dynamics_continuous<T>(x + k3 * dt, u);
                return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }

            case IntegratorType::EULER_IMPLICIT: {
                // x_{k+1} = x_k + dt * f(x_{k+1}, u)
                // Solve R(z) = z - x_k - dt * f(z, u) = 0
                return solve_implicit_step(x, u, dt, [](const Eigen::Matrix<T, NX, 1>& z, const Eigen::Matrix<T, NU, 1>& u_in) {
                    return dynamics_continuous(z, u_in);
                });
            }

            case IntegratorType::RK2_IMPLICIT: {
                // Implicit Midpoint (Gauss-Legendre 2)
                // x_{k+1} = x_k + dt * f( (x_k + x_{k+1})/2, u )
                // Solve R(z) = z - x_k - dt * f( (x_k + z)/2, u ) = 0
                return solve_implicit_step(x, u, dt, [x](const Eigen::Matrix<T, NX, 1>& z, const Eigen::Matrix<T, NU, 1>& u_in) {
                    return dynamics_continuous<T>((x + z) * 0.5, u_in);
                });
            }

            case IntegratorType::RK4_IMPLICIT: {
                // Gauss-Legendre 4 (Two-stage IRK)
                // This requires solving for two internal stages (k1, k2) of size NX each.
                // 2*NX nonlinear system.
                // For demonstration, we approximate or fallback to RK4 Explicit if too complex to inline,
                // BUT user requested it. Let's implement a simplified GL4 or rigorous one.
                // Rigorous GL4 requires solving 2*NX system.
                // To keep this "header-only" and simple without external nonlinear solvers,
                // we will use a fixed-point iteration or simple Newton for the stages.
                // However, given the code structure, implementing a robust 2*NX Newton solver here is heavy.
                // FALLBACK for now: Use RK4 Explicit but warn, OR implement a simplified version.
                // Let's implement the FULL GL4 using a specialized Newton solver for the stages.
                return solve_gl4_step(x, u, dt);
            }

            default:
                return x + dynamics_continuous(x, u) * dt;
        }
    }

    // --- Implicit Solver Helper (Newton-Raphson) ---
    template<typename T, typename Func>
    static Eigen::Matrix<T, NX, 1> solve_implicit_step(
        const Eigen::Matrix<T, NX, 1>& x0,
        const Eigen::Matrix<T, NU, 1>& u,
        double dt,
        Func flux_func)
    {
        // Guess z = x0 (Explicit Euler guess)
        Eigen::Matrix<T, NX, 1> z = x0 + dynamics_continuous(x0, u) * dt;
        
        const int max_iter = 10;
        const double tol = 1e-6;

        for(int iter=0; iter<max_iter; ++iter) {
            // Residual: R(z) = z - x0 - dt * flux(z)
            // Jacobian: J = I - dt * d(flux)/dz
            // Update: z = z - J^{-1} * R
            
            // 1. Evaluate Residual
            Eigen::Matrix<T, NX, 1> f_val = flux_func(z, u);
            Eigen::Matrix<T, NX, 1> R = z - x0 - f_val * dt;

            if(R.norm() < tol) break;

            // 2. Compute Jacobian via Finite Difference
            Eigen::Matrix<T, NX, NX> J;
            double eps = 1e-7;
            for(int i=0; i<NX; ++i) {
                Eigen::Matrix<T, NX, 1> z_p = z;
                z_p(i) += eps;
                Eigen::Matrix<T, NX, 1> f_p = flux_func(z_p, u);
                // df/dz approx
                Eigen::Matrix<T, NX, 1> df = (f_p - f_val) / eps;
                // dR/dz col
                J.col(i) = Eigen::Matrix<T, NX, 1>::Unit(i) - df * dt;
            }

            // 3. Update
            z = z - J.inverse() * R;
        }
        return z;
    }

    // --- Gauss-Legendre 4 Solver ---
    template<typename T>
    static Eigen::Matrix<T, NX, 1> solve_gl4_step(
        const Eigen::Matrix<T, NX, 1>& x0,
        const Eigen::Matrix<T, NU, 1>& u,
        double dt)
    {
        // Constants
        const double sqrt3 = 1.73205080757;
        // c1 = 1/2 - sqrt3/6, c2 = 1/2 + sqrt3/6
        // A matrix for RK: 
        // a11 = 1/4, a12 = 1/4 - sqrt3/6
        // a21 = 1/4 + sqrt3/6, a22 = 1/4
        // b1 = 1/2, b2 = 1/2
        
        const double a11 = 0.25;
        const double a12 = 0.25 - sqrt3/6.0;
        const double a21 = 0.25 + sqrt3/6.0;
        const double a22 = 0.25;

        // Variables: K = [k1; k2] (Size 2*NX)
        // Initial guess: f(x0, u) for both
        Eigen::Matrix<T, NX, 1> f0 = dynamics_continuous(x0, u);
        Eigen::Matrix<T, NX, 1> k1 = f0;
        Eigen::Matrix<T, NX, 1> k2 = f0;

        const int max_iter = 10;
        const double tol = 1e-6;

        for(int iter=0; iter<max_iter; ++iter) {
            // Calculate states at stages
            // Y1 = x0 + dt * (a11*k1 + a12*k2)
            // Y2 = x0 + dt * (a21*k1 + a22*k2)
            Eigen::Matrix<T, NX, 1> Y1 = x0 + dt * (a11 * k1 + a12 * k2);
            Eigen::Matrix<T, NX, 1> Y2 = x0 + dt * (a21 * k1 + a22 * k2);

            // Function evaluations
            Eigen::Matrix<T, NX, 1> f1 = dynamics_continuous(Y1, u);
            Eigen::Matrix<T, NX, 1> f2 = dynamics_continuous(Y2, u);

            // Residuals
            Eigen::Matrix<T, NX, 1> R1 = k1 - f1;
            Eigen::Matrix<T, NX, 1> R2 = k2 - f2;

            double err = R1.norm() + R2.norm();
            if(err < tol) break;

            // Jacobian J of the system R(K) = 0 w.r.t K
            // R1 = k1 - f(Y1(k1, k2))
            // dR1/dk1 = I - df/dY * dY1/dk1 = I - df/dY * (dt * a11)
            // dR1/dk2 =   - df/dY * dY1/dk2 =   - df/dY * (dt * a12)
            // etc.
            // We need Jacobian of f at Y1 and Y2.
            // Let's use FD for df/dY
            Eigen::Matrix<T, NX, NX> J_Y1 = finite_diff_jacobian(Y1, u);
            Eigen::Matrix<T, NX, NX> J_Y2 = finite_diff_jacobian(Y2, u);

            // Construct 2*NX x 2*NX system
            Eigen::Matrix<T, 2*NX, 2*NX> BigJ;
            Eigen::Matrix<T, NX, NX> I = Eigen::Matrix<T, NX, NX>::Identity();

            BigJ.template block<NX, NX>(0, 0)  = I - J_Y1 * (dt * a11);
            BigJ.template block<NX, NX>(0, NX) =   - J_Y1 * (dt * a12);
            BigJ.template block<NX, NX>(NX, 0) =   - J_Y2 * (dt * a21);
            BigJ.template block<NX, NX>(NX, NX)= I - J_Y2 * (dt * a22);

            Eigen::Matrix<T, 2*NX, 1> BigR;
            BigR << R1, R2;

            Eigen::Matrix<T, 2*NX, 1> DeltaK = BigJ.inverse() * BigR;
            
            k1 -= DeltaK.template head<NX>();
            k2 -= DeltaK.template tail<NX>();
        }

        // Final update: x_{n+1} = x_n + dt/2 * (k1 + k2)
        return x0 + 0.5 * dt * (k1 + k2);
    }

    // Helper for Jacobian
    template<typename T>
    static Eigen::Matrix<T, NX, NX> finite_diff_jacobian(const Eigen::Matrix<T, NX, 1>& x, const Eigen::Matrix<T, NU, 1>& u) {
        Eigen::Matrix<T, NX, NX> J;
        double eps = 1e-7;
        Eigen::Matrix<T, NX, 1> f0 = dynamics_continuous(x, u);
        for(int i=0; i<NX; ++i) {
            Eigen::Matrix<T, NX, 1> xp = x;
            xp(i) += eps;
            J.col(i) = (dynamics_continuous(xp, u) - f0) / eps;
        }
        return J;
    }

    // --- Compute with Integrator Selection ---
    template<typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt) {
        // --- 1. Unpack Variables ---
        T px = kp.x(0);
        T py = kp.x(1);
        T th = kp.x(2);
        T v  = kp.x(3);

        T acc = kp.u(0);
        T delta = kp.u(1);

        // --- 2. Dynamics & Derivatives (Finite Difference) ---
        // Explicitly calculate next state based on selected integrator
        Eigen::Matrix<T, NX, 1> x_next = integrate(kp.x, kp.u, dt, type);
        
        // f_resid stores the PREDICTED next state (f(x,u))
        kp.f_resid = x_next;

        // Compute A = df/dx, B = df/du using Finite Difference
        // This makes it work for ANY integrator (implicit or explicit)
        T eps = 1e-6;

        // A Matrix
        for(int i=0; i<NX; ++i) {
            Eigen::Matrix<T, NX, 1> x_p = kp.x;
            x_p(i) += eps;
            Eigen::Matrix<T, NX, 1> x_next_p = integrate(x_p, kp.u, dt, type);
            kp.A.col(i) = (x_next_p - x_next) / eps;
        }

        // B Matrix
        for(int i=0; i<NU; ++i) {
            Eigen::Matrix<T, NU, 1> u_p = kp.u;
            u_p(i) += eps;
            Eigen::Matrix<T, NX, 1> x_next_p = integrate(kp.x, u_p, dt, type);
            kp.B.col(i) = (x_next_p - x_next) / eps;
        }

        // --- 3. Costs (Quadratic Approximation) ---
        T v_target = kp.p(0);
        T x_ref    = kp.p(1);
        T y_ref    = kp.p(2);
        T obs_x    = kp.p(3);
        T obs_y    = kp.p(4);
        T obs_w    = kp.p(5);

        // Reset Cost
        kp.Q.setZero(); kp.q.setZero();
        kp.R.setZero(); kp.r.setZero();
        kp.H.setZero();

        // 3.1 Tracking Cost
        T w_pos = 1.0;
        T w_vel = 1.0;
        T w_ang = 0.1;

        kp.Q(0,0) += w_pos; kp.q(0) += w_pos * (px - x_ref);
        kp.Q(1,1) += w_pos; kp.q(1) += w_pos * (py - y_ref);
        kp.Q(2,2) += w_ang; kp.q(2) += w_ang * th;
        kp.Q(3,3) += w_vel; kp.q(3) += w_vel * (v - v_target);

        // 3.2 Control Cost
        T w_acc = 0.1;
        T w_steer = 1.0;
        kp.R(0,0) += w_acc;   kp.r(0) += w_acc * acc;
        kp.R(1,1) += w_steer; kp.r(1) += w_steer * delta;

        // 3.3 Obstacle Avoidance
        T dx = px - obs_x;
        T dy = py - obs_y;
        T dist2 = dx*dx + dy*dy;
        T sigma = 4.0; 
        T exp_val = exp(-dist2 / sigma);
        T cost_obs = obs_w * exp_val;

        T grad_factor = cost_obs * (-2.0 / sigma);
        T g_ox = grad_factor * dx;
        T g_oy = grad_factor * dy;

        kp.q(0) += g_ox;
        kp.q(1) += g_oy;
        kp.Q(0,0) += abs(g_ox); 
        kp.Q(1,1) += abs(g_oy);

        // --- 4. Constraints (Inequality) ---
        T max_acc = 3.0;
        T max_steer = 0.5;

        kp.g_val(0) = acc - max_acc;
        kp.g_val(1) = -acc - max_acc;
        kp.g_val(2) = delta - max_steer;
        kp.g_val(3) = -delta - max_steer;

        kp.C.setZero();
        kp.D.setZero();

        kp.D(0, 0) = 1.0;
        kp.D(1, 0) = -1.0;
        kp.D(2, 1) = 1.0;
        kp.D(3, 1) = -1.0;
    }
};

}
