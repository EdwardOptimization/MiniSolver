#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "minisolver/solver/riccati.h"

namespace {

// Minimal xorshift RNG for reproducible, low-overhead matrix generation.
struct XorShift64 {
    uint64_t s = 0x123456789abcdef0ULL;
    uint64_t next_u64()
    {
        uint64_t x = s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s = x;
        return x;
    }
    double uniform(double lo, double hi)
    {
        // 53-bit mantissa
        const uint64_t r = next_u64() >> 11;
        const double u = static_cast<double>(r) * (1.0 / 9007199254740992.0);
        return lo + (hi - lo) * u;
    }
};

template <typename MatrixType>
bool legacy_fast_inverse(const MatrixType& A, MatrixType& A_inv, double epsilon = 1e-9)
{
    constexpr int ROWS = MatrixType::RowsAtCompileTime;

    if constexpr (ROWS == 1) {
        const double val = A(0, 0);
        if (std::abs(val) < epsilon)
            return false;
        A_inv(0, 0) = 1.0 / val;
        return true;
    } else if constexpr (ROWS == 2) {
        const double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        if (std::abs(det) < epsilon)
            return false;

        const double inv_det = 1.0 / det;
        A_inv(0, 0) = A(1, 1) * inv_det;
        A_inv(0, 1) = -A(0, 1) * inv_det;
        A_inv(1, 0) = -A(1, 0) * inv_det;
        A_inv(1, 1) = A(0, 0) * inv_det;
        return true;
    } else if constexpr (ROWS == 3) {
        const double A00 = A(0, 0), A01 = A(0, 1), A02 = A(0, 2);
        const double A10 = A(1, 0), A11 = A(1, 1), A12 = A(1, 2);
        const double A20 = A(2, 0), A21 = A(2, 1), A22 = A(2, 2);

        const double det = A00 * (A11 * A22 - A12 * A21) - A01 * (A10 * A22 - A12 * A20)
            + A02 * (A10 * A21 - A11 * A20);

        if (std::abs(det) < epsilon)
            return false;
        const double inv_det = 1.0 / det;

        A_inv(0, 0) = (A11 * A22 - A12 * A21) * inv_det;
        A_inv(0, 1) = (A02 * A21 - A01 * A22) * inv_det;
        A_inv(0, 2) = (A01 * A12 - A02 * A11) * inv_det;

        A_inv(1, 0) = (A12 * A20 - A10 * A22) * inv_det;
        A_inv(1, 1) = (A00 * A22 - A02 * A20) * inv_det;
        A_inv(1, 2) = (A02 * A10 - A00 * A12) * inv_det;

        A_inv(2, 0) = (A10 * A21 - A11 * A20) * inv_det;
        A_inv(2, 1) = (A01 * A20 - A00 * A21) * inv_det;
        A_inv(2, 2) = (A00 * A11 - A01 * A10) * inv_det;
        return true;
    } else {
        // Legacy path used Cholesky solve as a fallback for N>3.
        return minisolver::MatOps::cholesky_solve(A, MatrixType::Identity(), A_inv);
    }
}

template <typename MatrixType> void fill_spd_1x1(MatrixType& A, XorShift64& rng)
{
    A(0, 0) = rng.uniform(0.5, 3.0);
}

template <typename MatrixType> void fill_spd_2x2(MatrixType& A, XorShift64& rng)
{
    const double a = rng.uniform(0.5, 3.0);
    const double c = rng.uniform(0.5, 3.0);
    double b = rng.uniform(-0.5, 0.5);

    // Enforce SPD: det = a*c - b^2 > 0.
    const double max_b = 0.8 * std::sqrt(a * c);
    if (b > max_b)
        b = max_b;
    if (b < -max_b)
        b = -max_b;

    A(0, 0) = a;
    A(0, 1) = b;
    A(1, 0) = b;
    A(1, 1) = c;
}

template <typename MatrixType> void fill_spd_3x3(MatrixType& A, XorShift64& rng)
{
    // Construct SPD via A = L * L^T with positive diagonal.
    minisolver::MSMat<double, 3, 3> L;
    L.setZero();
    L(0, 0) = rng.uniform(0.5, 3.0);
    L(1, 0) = rng.uniform(-0.5, 0.5);
    L(1, 1) = rng.uniform(0.5, 3.0);
    L(2, 0) = rng.uniform(-0.5, 0.5);
    L(2, 1) = rng.uniform(-0.5, 0.5);
    L(2, 2) = rng.uniform(0.5, 3.0);

    A = L * L.transpose();
}

struct BenchArgs {
    int64_t iters = 20'000'000;
    int mats = 4096;
    int repeats = 5;
    double epsilon = 1e-9;
};

BenchArgs parse_args(int argc, char** argv)
{
    BenchArgs a;
    for (int i = 1; i < argc; ++i) {
        const std::string s(argv[i]);
        auto next_val = [&](int& idx) -> const char* {
            if (idx + 1 >= argc)
                return nullptr;
            return argv[++idx];
        };

        if (s == "--iters") {
            const char* v = next_val(i);
            if (v)
                a.iters = std::stoll(v);
        } else if (s == "--mats") {
            const char* v = next_val(i);
            if (v)
                a.mats = std::stoi(v);
        } else if (s == "--repeats") {
            const char* v = next_val(i);
            if (v)
                a.repeats = std::stoi(v);
        } else if (s == "--epsilon") {
            const char* v = next_val(i);
            if (v)
                a.epsilon = std::stod(v);
        } else if (s == "--help" || s == "-h") {
            std::cout
                << "Usage: fast_inverse_bench [--iters N] [--mats M] [--repeats R] [--epsilon E]\n"
                << "Default: --iters 20000000 --mats 4096 --repeats 5 --epsilon 1e-9\n";
            std::exit(0);
        }
    }
    if (a.mats < 1)
        a.mats = 1;
    if (a.repeats < 1)
        a.repeats = 1;
    return a;
}

template <typename MatrixType, typename FillFn, typename InvFn>
double time_inv_ns_per_call(const BenchArgs& args, FillFn fill, InvFn inv_fn)
{
    // Allocate as an array (over-aligned types are handled by operator new[]).
    std::unique_ptr<MatrixType[]> mats(new MatrixType[static_cast<size_t>(args.mats)]);
    XorShift64 rng;

    for (int i = 0; i < args.mats; ++i) {
        fill(mats[static_cast<size_t>(i)], rng);
    }

    // Warmup
    MatrixType inv;
    volatile double sink = 0.0;
    for (int i = 0; i < 1000; ++i) {
        const MatrixType& A = mats[static_cast<size_t>(i % args.mats)];
        const bool ok = inv_fn(A, inv, args.epsilon);
        sink += ok ? inv(0, 0) : 0.0;
    }

    double best_ns = 1e300;
    for (int r = 0; r < args.repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < args.iters; ++i) {
            const MatrixType& A = mats[static_cast<size_t>(i % args.mats)];
            const bool ok = inv_fn(A, inv, args.epsilon);
            sink += ok ? inv(0, 0) : 0.0;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        const double ns_per = ns / static_cast<double>(args.iters);
        if (ns_per < best_ns)
            best_ns = ns_per;
    }

    // Prevent optimizing away.
    if (sink == 123456.789)
        std::cerr << "sink=" << sink << "\n";
    return best_ns;
}

template <typename MatrixType, typename FillFn>
void bench_case(const BenchArgs& args, const std::string& label, FillFn fill_fn)
{
    const double legacy_ns = time_inv_ns_per_call<MatrixType>(
        args, fill_fn, [](const MatrixType& A, MatrixType& A_inv, double eps) {
            return legacy_fast_inverse(A, A_inv, eps);
        });

    const double new_ns = time_inv_ns_per_call<MatrixType>(
        args, fill_fn, [](const MatrixType& A, MatrixType& A_inv, double eps) {
            return minisolver::fast_inverse(A, A_inv, eps);
        });

    const double delta_pct = (new_ns - legacy_ns) / legacy_ns * 100.0;

    std::cout << std::left << std::setw(10) << label << " legacy " << std::fixed
              << std::setprecision(3) << legacy_ns << " ns/call"
              << " | new " << std::fixed << std::setprecision(3) << new_ns << " ns/call"
              << " | delta " << std::showpos << std::fixed << std::setprecision(2) << delta_pct
              << "%\n"
              << std::noshowpos;
}

} // namespace

int main(int argc, char** argv)
{
    const BenchArgs args = parse_args(argc, argv);

    std::cout << "fast_inverse microbenchmark (before/after SPD checks)\n";
    std::cout << "iters=" << args.iters << " mats=" << args.mats << " repeats=" << args.repeats
              << " epsilon=" << std::scientific << args.epsilon << std::defaultfloat << "\n\n";

    bench_case<minisolver::MSMat<double, 1, 1>>(
        args, "1x1 SPD", fill_spd_1x1<minisolver::MSMat<double, 1, 1>>);
    bench_case<minisolver::MSMat<double, 2, 2>>(
        args, "2x2 SPD", fill_spd_2x2<minisolver::MSMat<double, 2, 2>>);
    bench_case<minisolver::MSMat<double, 3, 3>>(
        args, "3x3 SPD", fill_spd_3x3<minisolver::MSMat<double, 3, 3>>);

    return 0;
}
