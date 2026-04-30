#include "minisolver/matrix/matrix_defs.h"

#include <algorithm>
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
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            m(i, j) = std::cos(seed + 0.13 * i - 0.19 * j);
}

template <int R, int C, typename Mat> double checksum(const Mat& m)
{
    double sum = 0.0;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            sum += m(i, j) * (1.0 + 0.01 * i + 0.02 * j);
    return sum;
}

template <int R, int C, typename MatA, typename MatB>
double max_abs_diff(const MatA& a, const MatB& b)
{
    double max_diff = 0.0;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            max_diff = std::max(max_diff, std::abs(a(i, j) - b(i, j)));
    return max_diff;
}

template <int N, int NRHS>
bool solve_columns_baseline(
    const MSMat<double, N, N>& a, const MSMat<double, N, NRHS>& b, MSMat<double, N, NRHS>& x)
{
    for (int col_idx = 0; col_idx < NRHS; ++col_idx) {
        MSVec<double, N> col;
        if (!MatOps::lu_solve(a, b.col(col_idx), col))
            return false;
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

} // namespace

int main()
{
    run_case<4, 4>(300000);
    run_case<6, 6>(200000);
    run_case<8, 4>(200000);
    run_case<8, 2>(250000);
    run_case<12, 6>(100000);
    run_case<20, 10>(40000);

    if (sink == 123456789.0)
        std::cout << "sink\n";
    return 0;
}
