#include "minisolver/integrator/implicit_integrator.h"
#include "minisolver/matrix/matrix_defs.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace minisolver;

namespace {

volatile double sink = 0.0;

template <int R, int C, typename Mat> void fill_general(Mat& m, double seed)
{
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            const double base = std::sin(seed + 0.17 * i + 0.31 * j);
            m(i, j) = base * 0.02 + (i == j ? 2.0 + 0.05 * i : 0.0);
        }
    }
}

template <int R, int C, typename Mat> void fill_rhs(Mat& m, double seed)
{
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            m(i, j) = std::cos(seed + 0.13 * i - 0.19 * j);
        }
    }
}

template <int R, int C, typename Mat> double checksum(const Mat& m)
{
    double sum = 0.0;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            sum += m(i, j) * (1.0 + 0.01 * i + 0.02 * j);
        }
    }
    return sum;
}

struct GaussLegendreBenchModel {
    static const int NX = 6;
    static const int NU = 3;
    static const int NC = 0;
    static const int NP = 0;

    static inline long long dynamics_calls = 0;
    static inline long long jacobian_calls = 0;

    static void reset_counters()
    {
        dynamics_calls = 0;
        jacobian_calls = 0;
    }

    template <typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x, const MSVec<T, NU>& u, const MSVec<T, NP>& /*p*/)
    {
        ++dynamics_calls;

        MSVec<T, NX> xdot;
        for (int i = 0; i < NX; ++i) {
            const int next = (i + 1) % NX;
            const double rate = 0.45 + 0.08 * i;
            T control = T(0);
            for (int j = 0; j < NU; ++j) {
                control += static_cast<T>(0.03 * (i + 1) * (j + 1)) * u(j);
            }

            xdot(i) = -static_cast<T>(rate) * x(i) + static_cast<T>(0.05) * x(next) * x(next)
                + static_cast<T>(0.1 * std::sin(static_cast<double>(x(i)))) + control;
        }
        return xdot;
    }

    template <typename T>
    static ContinuousJacobians<T, NX, NU> jacobian_continuous(
        const MSVec<T, NX>& x, const MSVec<T, NU>& /*u*/, const MSVec<T, NP>& /*p*/)
    {
        ++jacobian_calls;

        ContinuousJacobians<T, NX, NU> jac;
        jac.Jx.setZero();
        jac.Ju.setZero();
        for (int i = 0; i < NX; ++i) {
            const int next = (i + 1) % NX;
            const double rate = 0.45 + 0.08 * i;
            jac.Jx(i, i)
                = -static_cast<T>(rate) + static_cast<T>(0.1 * std::cos(static_cast<double>(x(i))));
            jac.Jx(i, next) = static_cast<T>(0.1) * x(next);
            for (int j = 0; j < NU; ++j) {
                jac.Ju(i, j) = static_cast<T>(0.03 * (i + 1) * (j + 1));
            }
        }
        return jac;
    }
};

template <int R, int C, typename MatA, typename MatB>
double max_abs_diff(const MatA& a, const MatB& b)
{
    double max_diff = 0.0;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            max_diff = std::max(max_diff, std::abs(a(i, j) - b(i, j)));
        }
    }
    return max_diff;
}

template <int N, int NRHS>
bool solve_columns_baseline(
    const MSMat<double, N, N>& a, const MSMat<double, N, NRHS>& b, MSMat<double, N, NRHS>& x)
{
    for (int col_idx = 0; col_idx < NRHS; ++col_idx) {
        MSVec<double, N> col;
        if (!MatOps::lu_solve(a, b.col(col_idx), col)) {
            return false;
        }
        x.col(col_idx) = col;
    }
    return true;
}

template <int N, int NRHS>
bool solve_multi_rhs_local(
    const MSMat<double, N, N>& a, const MSMat<double, N, NRHS>& b, MSMat<double, N, NRHS>& x)
{
    return MatOps::lu_solve_matrix(a, b, x);
}

template <int N, int NRHS, typename Func>
double time_case(const std::string& label, int iters, Func&& func)
{
    MSMat<double, N, N> a;
    MSMat<double, N, NRHS> b;
    MSMat<double, N, NRHS> x;
    fill_general<N, N>(a, 0.3 + 0.01 * N);
    fill_rhs<N, NRHS>(b, 0.7 + 0.02 * NRHS);

    if (!func(a, b, x)) {
        std::cerr << label << " failed\n";
        std::abort();
    }
    sink += checksum<N, NRHS>(x);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        func(a, b, x);
        sink += checksum<N, NRHS>(x);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);

    std::cout << std::left << std::setw(18) << label << " N=" << std::setw(2) << N
              << " RHS=" << std::setw(2) << NRHS << " ns/iter=" << std::fixed
              << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int N, int NRHS> void validate_case()
{
    MSMat<double, N, N> a;
    MSMat<double, N, NRHS> b;
    MSMat<double, N, NRHS> baseline;
    MSMat<double, N, NRHS> multi;
    fill_general<N, N>(a, 0.5 + 0.01 * N);
    fill_rhs<N, NRHS>(b, 0.9 + 0.02 * NRHS);
    if (!solve_columns_baseline<N, NRHS>(a, b, baseline)
        || !solve_multi_rhs_local<N, NRHS>(a, b, multi)) {
        std::cerr << "validation solve failed N=" << N << " RHS=" << NRHS << "\n";
        std::abort();
    }
    const double diff = max_abs_diff<N, NRHS>(baseline, multi);
    if (diff > 1e-12) {
        std::cerr << "validation mismatch N=" << N << " RHS=" << NRHS << " diff=" << diff << "\n";
        std::abort();
    }
}

template <int N, int NRHS> void run_case(int iters)
{
    validate_case<N, NRHS>();
    time_case<N, NRHS>("lu_columns", iters, solve_columns_baseline<N, NRHS>);
    time_case<N, NRHS>("lu_multi_rhs", iters, solve_multi_rhs_local<N, NRHS>);
}

template <typename Model> void fill_gauss_knot(KnotPoint<double, Model::NX, Model::NU, 0, 0>& kp)
{
    kp.set_zero();
    for (int i = 0; i < Model::NX; ++i) {
        kp.x(i) = 0.2 + 0.04 * i;
    }
    for (int i = 0; i < Model::NU; ++i) {
        kp.u(i) = -0.1 + 0.03 * i;
    }
}

template <typename Model> void run_gauss_compute_case(int iters)
{
    using Knot = KnotPoint<double, Model::NX, Model::NU, 0, 0>;

    Knot kp;
    fill_gauss_knot<Model>(kp);

    NewtonConfig config;
    config.max_iters = 8;
    config.tol = 1e-12;
    config.regularization = 1e-12;

    ImplicitIntegrator<Model>::compute_dynamics(kp, IntegratorType::RK4_IMPLICIT, 0.05, config);
    sink += checksum<Model::NX, 1>(kp.f_resid) + checksum<Model::NX, Model::NX>(kp.A)
        + checksum<Model::NX, Model::NU>(kp.B);

    Model::reset_counters();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        fill_gauss_knot<Model>(kp);
        ImplicitIntegrator<Model>::compute_dynamics(kp, IntegratorType::RK4_IMPLICIT, 0.05, config);
        sink += checksum<Model::NX, 1>(kp.f_resid) + checksum<Model::NX, Model::NX>(kp.A)
            + checksum<Model::NX, Model::NU>(kp.B);
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);
    const double dyn_per_iter = static_cast<double>(Model::dynamics_calls) / iters;
    const double jac_per_iter = static_cast<double>(Model::jacobian_calls) / iters;

    std::cout << std::left << std::setw(18) << "gauss_compute"
              << " NX=" << std::setw(2) << Model::NX << " NU=" << std::setw(2) << Model::NU
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter
              << " dyn/iter=" << std::setprecision(2) << dyn_per_iter
              << " jac/iter=" << jac_per_iter << "\n";
}

} // namespace

int main()
{
    run_gauss_compute_case<GaussLegendreBenchModel>(100000);

    run_case<4, 4>(300000);
    run_case<6, 6>(200000);
    run_case<8, 4>(200000);
    run_case<8, 2>(250000);
    run_case<12, 6>(100000);
    run_case<20, 10>(40000);

    if (sink == 123456789.0) {
        std::cout << "sink\n";
    }
    return 0;
}
