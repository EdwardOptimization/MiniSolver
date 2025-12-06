#pragma once
#include "core/types.h"
#include <cmath>
namespace generated {
struct CarModel {
    static const int NX = 4; static const int NU = 2; static const int NP = 3; static const int NC = 0;
    template<typename T>
    static void compute(roboopt::KnotPoint<T,4,2,0,3>& kp) {
        T x_0 = kp.x(0);
        T x_1 = kp.x(1);
        T x_2 = kp.x(2);
        T x_3 = kp.x(3);
        T u_0 = kp.u(0);
        T u_1 = kp.u(1);
        T p_0 = kp.p(0);
        T p_1 = kp.p(1);
        T p_2 = kp.p(2);
        T x0 = 0.10000000000000001*cos(x_2);
        T x1 = x0*x_3;
        T x2 = 0.10000000000000001*sin(x_2);
        T x3 = x2*x_3;
        T x4 = tan(u_1);
        T x5 = 0.10000000000000001*x4;
        kp.f_resid(0,0) = x1 + x_0;
        kp.f_resid(1,0) = x3 + x_1;
        kp.f_resid(2,0) = x5*x_3 + x_2;
        kp.f_resid(3,0) = 0.10000000000000001*u_0 + x_3;
        kp.A(0,0) = 1;
        kp.A(0,2) = -x3;
        kp.A(0,3) = x0;
        kp.A(1,1) = 1;
        kp.A(1,2) = x1;
        kp.A(1,3) = x2;
        kp.A(2,2) = 1;
        kp.A(2,3) = x5;
        kp.A(3,3) = 1;
        kp.B(2,1) = 0.10000000000000001*x_3*(pow(x4, 2) + 1);
        kp.B(3,0) = 0.10000000000000001;
        kp.q(0,0) = -2.0*p_1 + 2.0*x_0;
        kp.q(1,0) = -2.0*p_2 + 2.0*x_1;
        kp.q(3,0) = -0.20000000000000001*p_0 + 0.20000000000000001*x_3;
        kp.r(0,0) = 0.02*u_0;
        kp.r(1,0) = 0.20000000000000001*u_1;
        kp.Q(0,0) = 2.0;
        kp.Q(1,1) = 2.0;
        kp.Q(3,3) = 0.20000000000000001;
        kp.R(0,0) = 0.02;
        kp.R(1,1) = 0.20000000000000001;
    }
};
}