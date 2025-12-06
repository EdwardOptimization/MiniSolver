#pragma once
#include "core/types.h"
#include "core/solver_options.h"
#include "core/matrix_defs.h"
#include <cmath>

namespace minisolver {

struct CarModel {
    static const int NX=4; 
    static const int NU=2; 
    static const int NC=5; 
    static const int NP=6;

    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& u) 
    {
        T x2=x(2); T x3=x(3);
        T u0=u(0); T u1=u(1);
        MSVec<T, NX> xdot;
        xdot(0) = x3 * cos(x2);
        xdot(1) = x3 * sin(x2);
        xdot(2) = (x3 / 2.5) * tan(u1);
        xdot(3) = u0;
        return xdot;
    }

    // --- Integrator Interface ---
    template<typename T>
    static MSVec<T, NX> integrate(
        const MSVec<T, NX>& x,
        const MSVec<T, NU>& u,
        double dt,
        IntegratorType type)
    {
        switch(type) {
            case IntegratorType::EULER_EXPLICIT: return x + dynamics_continuous(x, u) * dt;
            default: // RK4 Explicit
            {
               auto k1 = dynamics_continuous(x, u);
               auto k2 = dynamics_continuous<T>(x + k1 * (0.5 * dt), u);
               auto k3 = dynamics_continuous<T>(x + k2 * (0.5 * dt), u);
               auto k4 = dynamics_continuous<T>(x + k3 * dt, u);
               return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }
        }
    }

    // --- Main Compute Function ---
    template<typename T>
    static void compute(KnotPoint<T,NX,NU,NC,NP>& kp, IntegratorType type, double dt) {
        T x0=kp.x(0); T x1=kp.x(1); T x2=kp.x(2); T x3=kp.x(3);
        T u0=kp.u(0); T u1=kp.u(1);
        T p0=kp.p(0); T p1=kp.p(1); T p2=kp.p(2); T p3=kp.p(3); T p4=kp.p(4); T p5=kp.p(5);

        T x4 = cos(x2);
        T x5 = dt*u0;
        T x6 = x3 + x5;
        T x7 = x3 + 1.5*x5;
        T x8 = tan(u1);
        T x9 = 0.40000000000000002*x8;
        T x10 = dt*x9;
        T x11 = x10*x7 + x2;
        T x12 = cos(x11);
        T x13 = x12*x6;
        T x14 = x3 + 0.5*x5;
        T x15 = 0.20000000000000001*x8;
        T x16 = dt*x15;
        T x17 = x14*x16 + x2;
        T x18 = cos(x17);
        T x19 = 2*x18;
        T x20 = x3 + 1.0*x5;
        T x21 = x16*x20 + x2;
        T x22 = cos(x21);
        T x23 = 2*x22;
        T x24 = 0.16666666666666666*dt;
        T x25 = x24*(x13 + x14*x19 + x14*x23 + x3*x4);
        T x26 = sin(x2);
        T x27 = sin(x11);
        T x28 = x27*x6;
        T x29 = sin(x17);
        T x30 = 2*x29;
        T x31 = sin(x21);
        T x32 = 2*x31;
        T x33 = x14*x30 + x14*x32 + x26*x3 + x28;
        T x34 = 1.6000000000000001*x14;
        T x35 = x10*x14;
        T x36 = pow(dt, 2);
        T x37 = 0.60000000000000009*x36*x8;
        T x38 = x15*x36;
        T x39 = x14*x38;
        T x40 = x14*x31;
        T x41 = x36*x9;
        T x42 = pow(x8, 2) + 1;
        T x43 = 0.40000000000000002*x42;
        T x44 = dt*x43;
        T x45 = pow(x14, 2)*x44;
        T x46 = x44*x7;
        T x47 = x20*x44;
        T x48 = 1.0*dt;
        T x49 = x14*x22;
        T x50 = -x0;
        T x51 = -x1;
        // f_resid
        kp.f_resid(0,0) = x0 + x25;
        kp.f_resid(1,0) = x1 + x24*x33;
        kp.f_resid(2,0) = x2 + x24*(x3*x9 + x34*x8 + x6*x9);
        kp.f_resid(3,0) = x20;
        // A
        kp.A(0,0) = 1;
        kp.A(0,1) = 0;
        kp.A(0,2) = -x24*x33;
        kp.A(0,3) = x24*(-x10*x28 + x12 + x19 + x23 - x29*x35 - x31*x35 + x4);
        kp.A(1,0) = 0;
        kp.A(1,1) = 1;
        kp.A(1,2) = x25;
        kp.A(1,3) = x24*(x10*x13 + x18*x35 + x22*x35 + x26 + x27 + x30 + x32);
        kp.A(2,0) = 0;
        kp.A(2,1) = 0;
        kp.A(2,2) = 1;
        kp.A(2,3) = x10;
        kp.A(3,0) = 0;
        kp.A(3,1) = 0;
        kp.A(3,2) = 0;
        kp.A(3,3) = 1;
        // B
        kp.B(0,0) = x24*(dt*x12 + 1.0*dt*x18 + 1.0*dt*x22 - x28*x37 - x29*x39 - x40*x41);
        kp.B(0,1) = x24*(-x28*x46 - x29*x45 - x40*x47);
        kp.B(1,0) = x24*(dt*x27 + x13*x37 + x18*x39 + x29*x48 + x31*x48 + x41*x49);
        kp.B(1,1) = x24*(x13*x46 + x18*x45 + x47*x49);
        kp.B(2,0) = x38;
        kp.B(2,1) = x24*(x3*x43 + x34*x42 + x43*x6);
        kp.B(3,0) = x48;
        kp.B(3,1) = 0;
        // q
        kp.q(0,0) = -2.0*p1 + 2.0*x0;
        kp.q(1,0) = -2.0*p2 + 2.0*x1;
        kp.q(2,0) = 0.20000000000000001*x2;
        kp.q(3,0) = -2.0*p0 + 2.0*x3;
        // r
        kp.r(0,0) = 0.20000000000000001*u0;
        kp.r(1,0) = 2.0*u1;
        // Q
        kp.Q(0,0) = 2.0;
        kp.Q(0,1) = 0;
        kp.Q(0,2) = 0;
        kp.Q(0,3) = 0;
        kp.Q(1,0) = 0;
        kp.Q(1,1) = 2.0;
        kp.Q(1,2) = 0;
        kp.Q(1,3) = 0;
        kp.Q(2,0) = 0;
        kp.Q(2,1) = 0;
        kp.Q(2,2) = 0.20000000000000001;
        kp.Q(2,3) = 0;
        kp.Q(3,0) = 0;
        kp.Q(3,1) = 0;
        kp.Q(3,2) = 0;
        kp.Q(3,3) = 2.0;
        // R
        kp.R(0,0) = 0.20000000000000001;
        kp.R(0,1) = 0;
        kp.R(1,0) = 0;
        kp.R(1,1) = 2.0;
        // H
        kp.H(0,0) = 0;
        kp.H(0,1) = 0;
        kp.H(0,2) = 0;
        kp.H(0,3) = 0;
        kp.H(1,0) = 0;
        kp.H(1,1) = 0;
        kp.H(1,2) = 0;
        kp.H(1,3) = 0;
        // g_val
        kp.g_val(0,0) = u0 - 3.0;
        kp.g_val(1,0) = -u0 - 3.0;
        kp.g_val(2,0) = u1 - 0.5;
        kp.g_val(3,0) = -u1 - 0.5;
        kp.g_val(4,0) = -pow(-p3 - x50, 2) - pow(-p4 - x51, 2) + pow(p5 + 1.0, 2);
        // C
        kp.C(0,0) = 0;
        kp.C(0,1) = 0;
        kp.C(0,2) = 0;
        kp.C(0,3) = 0;
        kp.C(1,0) = 0;
        kp.C(1,1) = 0;
        kp.C(1,2) = 0;
        kp.C(1,3) = 0;
        kp.C(2,0) = 0;
        kp.C(2,1) = 0;
        kp.C(2,2) = 0;
        kp.C(2,3) = 0;
        kp.C(3,0) = 0;
        kp.C(3,1) = 0;
        kp.C(3,2) = 0;
        kp.C(3,3) = 0;
        kp.C(4,0) = 2*p3 - 2*x0;
        kp.C(4,1) = 2*p4 - 2*x1;
        kp.C(4,2) = 0;
        kp.C(4,3) = 0;
        // D
        kp.D(0,0) = 1;
        kp.D(0,1) = 0;
        kp.D(1,0) = -1;
        kp.D(1,1) = 0;
        kp.D(2,0) = 0;
        kp.D(2,1) = 1;
        kp.D(3,0) = 0;
        kp.D(3,1) = -1;
        kp.D(4,0) = 0;
        kp.D(4,1) = 0;
        kp.cost = 0.10000000000000001*pow(u0, 2) + 1.0*pow(u1, 2) + 0.10000000000000001*pow(x2, 2) + 1.0*pow(-p0 + x3, 2) + 1.0*pow(-p1 - x50, 2) + 1.0*pow(-p2 - x51, 2);
    }
};
}
