// MiniMatrix vs Eigen microbenchmark for the inner kernels exercised by
// MiniSolver's hot path (Riccati GEMM blocks, LDLT factorization, triangular
// solve).
//
// Build with both backends from the same source by toggling the existing
// USE_EIGEN / USE_CUSTOM_MATRIX CMake options. This bench is intentionally a
// *measurement tool*, not a regression: it prints ns/iter for each kernel so
// engineers can decide where to invest further unrolling or SIMD work.
//
// The kernels measured here are the ones that dominate Riccati's backward
// pass at the small fixed sizes typical of NMPC (4x4, 6x6, 8x8, 12x12). They
// are deliberately not Eigen-vs-MiniMatrix in the sense of "swap one for the
// other in solver.h" -- both implementations live behind MatOps so the
// benchmark exercises whichever one the build is configured for.

#include "minisolver/matrix/matrix_defs.h"
#include "minisolver/matrix/mini_matrix.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef USE_EIGEN
#include <Eigen/Cholesky>
#include <Eigen/Core>
#endif

using namespace minisolver;

namespace {

volatile double sink = 0.0;

template <int R, int C> void fill_mini(MiniMatrix<double, R, C>& m, double seed)
{
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            m(i, j) = std::sin(seed + 0.17 * i + 0.23 * j + 0.31 * (i * C + j));
        }
    }
}

template <int N> void fill_spd_mini(MiniMatrix<double, N, N>& A, double seed)
{
    MiniMatrix<double, N, N> X;
    fill_mini(X, seed);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += X(i, k) * X(j, k);
            }
            A(i, j) = sum + (i == j ? 1.0 : 0.0);
        }
    }
}

template <int R, int C> double checksum_mini(const MiniMatrix<double, R, C>& m)
{
    double sum = 0.0;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            sum += m(i, j) * (1.0 + 0.01 * i + 0.03 * j);
        }
    }
    return sum;
}

template <int R, int K, int C> double bench_minimatrix_gemm(int iters, const std::string& label)
{
    MiniMatrix<double, R, K> A;
    MiniMatrix<double, K, C> B;
    MiniMatrix<double, R, C> Out;
    fill_mini(A, 0.13);
    fill_mini(B, 0.71);

    matrix::matmul(Out, A, B);
    sink += checksum_mini(Out);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        matrix::matmul(Out, A, B);
        sink += checksum_mini(Out);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);
    std::cout << std::left << std::setw(28) << label << " " << R << "x" << K << "*" << K << "x" << C
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int N> double bench_minimatrix_ldlt(int iters, const std::string& label)
{
    MiniMatrix<double, N, N> A;
    fill_spd_mini(A, 0.91);

    MiniLDLT<double, N> ldlt(A);
    sink += checksum_mini(A);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        MiniLDLT<double, N> local(A);
        sink += static_cast<double>(local.info());
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);
    std::cout << std::left << std::setw(28) << label << " " << N << "x" << N
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}

#ifdef USE_EIGEN
template <int R, int C> void fill_eigen(Eigen::Matrix<double, R, C>& m, double seed)
{
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            m(i, j) = std::sin(seed + 0.17 * i + 0.23 * j + 0.31 * (i * C + j));
        }
    }
}

template <int N> void fill_spd_eigen(Eigen::Matrix<double, N, N>& A, double seed)
{
    Eigen::Matrix<double, N, N> X;
    fill_eigen(X, seed);
    A = X * X.transpose();
    for (int i = 0; i < N; ++i) {
        A(i, i) += 1.0;
    }
}

template <int R, int C> double checksum_eigen(const Eigen::Matrix<double, R, C>& m)
{
    double sum = 0.0;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            sum += m(i, j) * (1.0 + 0.01 * i + 0.03 * j);
        }
    }
    return sum;
}

template <int R, int K, int C> double bench_eigen_gemm(int iters, const std::string& label)
{
    Eigen::Matrix<double, R, K> A;
    Eigen::Matrix<double, K, C> B;
    Eigen::Matrix<double, R, C> Out;
    fill_eigen(A, 0.13);
    fill_eigen(B, 0.71);

    Out.noalias() = A * B;
    sink += checksum_eigen(Out);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        Out.noalias() = A * B;
        sink += checksum_eigen(Out);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);
    std::cout << std::left << std::setw(28) << label << " " << R << "x" << K << "*" << K << "x" << C
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int N> double bench_eigen_ldlt(int iters, const std::string& label)
{
    Eigen::Matrix<double, N, N> A;
    fill_spd_eigen(A, 0.91);

    Eigen::LDLT<Eigen::Matrix<double, N, N>> ldlt(A);
    sink += checksum_eigen(A);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        Eigen::LDLT<Eigen::Matrix<double, N, N>> local(A);
        sink += local.info() == Eigen::Success ? 1.0 : 0.0;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);
    std::cout << std::left << std::setw(28) << label << " " << N << "x" << N
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}
#endif

} // namespace

int main()
{
    const int gemm_iters = 2000000;
    const int ldlt_iters = 500000;

#ifdef USE_EIGEN
    std::cout << "Backend: Eigen3\n";
    bench_eigen_gemm<4, 4, 4>(gemm_iters, "eigen_gemm_4");
    bench_eigen_gemm<6, 6, 6>(gemm_iters, "eigen_gemm_6");
    bench_eigen_gemm<8, 8, 8>(gemm_iters, "eigen_gemm_8");
    bench_eigen_gemm<12, 12, 12>(gemm_iters, "eigen_gemm_12");
    bench_eigen_gemm<8, 4, 8>(gemm_iters, "eigen_gemm_8x4_4x8");
    bench_eigen_ldlt<4>(ldlt_iters, "eigen_ldlt_4");
    bench_eigen_ldlt<6>(ldlt_iters, "eigen_ldlt_6");
    bench_eigen_ldlt<8>(ldlt_iters, "eigen_ldlt_8");
    bench_eigen_ldlt<12>(ldlt_iters, "eigen_ldlt_12");
#else
    std::cout << "Backend: MiniMatrix (custom)\n";
    bench_minimatrix_gemm<4, 4, 4>(gemm_iters, "mini_gemm_4");
    bench_minimatrix_gemm<6, 6, 6>(gemm_iters, "mini_gemm_6");
    bench_minimatrix_gemm<8, 8, 8>(gemm_iters, "mini_gemm_8");
    bench_minimatrix_gemm<12, 12, 12>(gemm_iters, "mini_gemm_12");
    bench_minimatrix_gemm<8, 4, 8>(gemm_iters, "mini_gemm_8x4_4x8");
    bench_minimatrix_ldlt<4>(ldlt_iters, "mini_ldlt_4");
    bench_minimatrix_ldlt<6>(ldlt_iters, "mini_ldlt_6");
    bench_minimatrix_ldlt<8>(ldlt_iters, "mini_ldlt_8");
    bench_minimatrix_ldlt<12>(ldlt_iters, "mini_ldlt_12");
#endif

    if (sink == 1234567.0) {
        std::cout << "sink\n";
    }
    return 0;
}
