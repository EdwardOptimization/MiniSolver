#include "minisolver/matrix/mini_matrix.h"
#include <array>
#include <chrono>
#include <iostream>
#include <string>

using minisolver::MiniMatrix;

namespace {

volatile double sink = 0.0;

template <typename Mat> double checksum(const Mat& m)
{
    double sum = 0.0;
    for (int i = 0; i < Mat::Rows * Mat::Cols; ++i)
        sum += m.data[i] * static_cast<double>(i + 1);
    return sum;
}

template <typename Fn> double time_ns_per_iter(const std::string& name, int iters, Fn fn)
{
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        fn(i);
    const auto end = std::chrono::high_resolution_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(end - start).count();
    const double per_iter = ns / static_cast<double>(iters);
    std::cout << name << "," << per_iter << "\n";
    return per_iter;
}

template <int R, int K, int C> void bench_matmul(int iters)
{
    const int batch = 64;
    std::array<MiniMatrix<double, R, K>, batch> As;
    std::array<MiniMatrix<double, K, C>, batch> Bs;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < R * K; ++i)
            As[b].data[i] = 0.01 * static_cast<double>((b + 1) * (i + 1));
        for (int i = 0; i < K * C; ++i)
            Bs[b].data[i] = 0.02 * static_cast<double>((b + 3) * (i + 1));
    }

    time_ns_per_iter("matmul_" + std::to_string(R) + "x" + std::to_string(K) + "_"
            + std::to_string(K) + "x" + std::to_string(C),
        iters, [&](int i) {
            const MiniMatrix<double, R, K>& A = As[i & (batch - 1)];
            const MiniMatrix<double, K, C>& B = Bs[(i * 7) & (batch - 1)];
            MiniMatrix<double, R, C> Cmat = A * B;
            sink += checksum(Cmat);
        });
}

template <int RA, int R, int C> void bench_add_at_mul_b(int iters)
{
    const int batch = 64;
    std::array<MiniMatrix<double, RA, R>, batch> As;
    std::array<MiniMatrix<double, RA, C>, batch> Bs;
    MiniMatrix<double, R, C> D;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < RA * R; ++i)
            As[b].data[i] = 0.01 * static_cast<double>((b + 1) * (i + 1));
        for (int i = 0; i < RA * C; ++i)
            Bs[b].data[i] = 0.02 * static_cast<double>((b + 3) * (i + 1));
    }

    time_ns_per_iter("add_At_mul_B_" + std::to_string(RA) + "x" + std::to_string(R) + "_"
            + std::to_string(RA) + "x" + std::to_string(C),
        iters, [&](int i) {
            const MiniMatrix<double, RA, R>& A = As[i & (batch - 1)];
            const MiniMatrix<double, RA, C>& B = Bs[(i * 7) & (batch - 1)];
            D.add_At_mul_B(A, B);
            sink += checksum(D);
        });
}

}

int main()
{
    const int iters = 1000000;
    std::cout << "kernel,ns_per_iter\n";
    bench_matmul<4, 4, 4>(iters);
    bench_matmul<8, 8, 8>(iters);
    bench_matmul<12, 12, 12>(iters / 10);
    bench_add_at_mul_b<8, 4, 4>(iters);
    bench_add_at_mul_b<16, 8, 8>(iters / 10);
    std::cerr << "sink=" << sink << "\n";
    return 0;
}
