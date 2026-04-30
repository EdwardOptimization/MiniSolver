#pragma once

#include "minisolver/matrix/matrix_defs.h"

namespace minisolver {

// Continuous Jacobians of f(x, u, p) — the ODE right-hand side.
template <typename T, int NX, int NU>
struct ContinuousJacobians {
    MSMat<T, NX, NX> Jx; // df/dx
    MSMat<T, NX, NU> Ju; // df/du
};

// Compute continuous Jacobians via centered finite differences.
// Model must provide: static MSVec<T,NX> dynamics_continuous(x, u, p)
template <typename Model, typename T>
ContinuousJacobians<T, Model::NX, Model::NU> compute_numerical_jacobian(
    const MSVec<T, Model::NX>& x,
    const MSVec<T, Model::NU>& u,
    const MSVec<T, Model::NP>& p,
    double eps = 1e-7)
{
    constexpr int NX = Model::NX;
    constexpr int NU = Model::NU;
    ContinuousJacobians<T, NX, NU> jac;

    MSVec<T, NX> f_plus, f_minus;

    // df/dx via centered differences
    MSVec<T, NX> x_work = x;
    for (int j = 0; j < NX; ++j) {
        double saved = x_work(j);
        x_work(j) = saved + eps;
        f_plus = Model::dynamics_continuous(x_work, u, p);
        x_work(j) = saved - eps;
        f_minus = Model::dynamics_continuous(x_work, u, p);
        x_work(j) = saved;
        jac.Jx.col(j) = (f_plus - f_minus) / (2.0 * eps);
    }

    // df/du via centered differences
    MSVec<T, NU> u_work = u;
    for (int j = 0; j < NU; ++j) {
        double saved = u_work(j);
        u_work(j) = saved + eps;
        f_plus = Model::dynamics_continuous(x, u_work, p);
        u_work(j) = saved - eps;
        f_minus = Model::dynamics_continuous(x, u_work, p);
        u_work(j) = saved;
        jac.Ju.col(j) = (f_plus - f_minus) / (2.0 * eps);
    }

    return jac;
}

} // namespace minisolver
