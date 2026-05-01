#include "minisolver/matrix/mini_matrix.h"
#include "minisolver/matrix/static_for.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace minisolver;

namespace {

volatile double sink = 0.0;

template <typename Mat> void fill_matrix(Mat& m, double seed)
{
    for (int i = 0; i < Mat::Rows; ++i) {
        for (int j = 0; j < Mat::Cols; ++j) {
            m(i, j) = std::sin(seed + 0.17 * i + 0.23 * j);
        }
    }
}

template <typename Mat> double checksum(const Mat& m)
{
    double sum = 0.0;
    for (int i = 0; i < Mat::Rows; ++i) {
        for (int j = 0; j < Mat::Cols; ++j) {
            sum += m(i, j) * (1.0 + 0.01 * i + 0.03 * j);
        }
    }
    return sum;
}

template <int ParentR, int ParentC, int BR, int BC>
void assign_block_baseline(MiniMatrix<double, ParentR, ParentC>& dst,
    const MiniMatrix<double, BR, BC>& src, int row0, int col0)
{
    dst.template block<BR, BC>(row0, col0) = src;
}

template <int ParentR, int ParentC, int BR, int BC>
void assign_block_static(MiniMatrix<double, ParentR, ParentC>& dst,
    const MiniMatrix<double, BR, BC>& src, int row0, int col0)
{
    struct Body {
        MiniMatrix<double, ParentR, ParentC>& dst;
        const MiniMatrix<double, BR, BC>& src;
        int row0;
        int col0;
        inline void operator()(int index)
        {
            const int r = index / BC;
            const int c = index - r * BC;
            dst(row0 + r, col0 + c) = src(r, c);
        }
    } body = { dst, src, row0, col0 };
    matrix::StaticFor<0, BR * BC>::run(body);
}

template <int ParentR, int ParentC, int BR, int BC>
void extract_block_baseline(const MiniMatrix<double, ParentR, ParentC>& src,
    MiniMatrix<double, BR, BC>& dst, int row0, int col0)
{
    dst = src.template block<BR, BC>(row0, col0);
}

template <int ParentR, int ParentC, int BR, int BC>
void extract_block_static(const MiniMatrix<double, ParentR, ParentC>& src,
    MiniMatrix<double, BR, BC>& dst, int row0, int col0)
{
    struct Body {
        const MiniMatrix<double, ParentR, ParentC>& src;
        MiniMatrix<double, BR, BC>& dst;
        int row0;
        int col0;
        inline void operator()(int index)
        {
            const int r = index / BC;
            const int c = index - r * BC;
            dst(r, c) = src(row0 + r, col0 + c);
        }
    } body = { src, dst, row0, col0 };
    matrix::StaticFor<0, BR * BC>::run(body);
}

template <int ParentR, int ParentC, int BR, int BC, typename Func>
double time_assign(const std::string& label, int iters, Func&& func)
{
    MiniMatrix<double, ParentR, ParentC> dst;
    MiniMatrix<double, BR, BC> src;
    fill_matrix(dst, 0.3);
    fill_matrix(src, 0.8);
    const int row0 = ParentR - BR;
    const int col0 = ParentC - BC;

    func(dst, src, row0, col0);
    sink += checksum(dst);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        func(dst, src, row0, col0);
        sink += checksum(dst);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);
    std::cout << std::left << std::setw(18) << label << " P=" << ParentR << "x" << ParentC
              << " B=" << BR << "x" << BC << " ns/iter=" << std::fixed << std::setprecision(3)
              << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int ParentR, int ParentC, int BR, int BC, typename Func>
double time_extract(const std::string& label, int iters, Func&& func)
{
    MiniMatrix<double, ParentR, ParentC> src;
    MiniMatrix<double, BR, BC> dst;
    fill_matrix(src, 1.1);
    const int row0 = ParentR - BR;
    const int col0 = ParentC - BC;

    func(src, dst, row0, col0);
    sink += checksum(dst);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        func(src, dst, row0, col0);
        sink += checksum(dst);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);
    std::cout << std::left << std::setw(18) << label << " P=" << ParentR << "x" << ParentC
              << " B=" << BR << "x" << BC << " ns/iter=" << std::fixed << std::setprecision(3)
              << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int ParentR, int ParentC, int BR, int BC> void run_case(int iters)
{
    time_assign<ParentR, ParentC, BR, BC>(
        "assign_base", iters, assign_block_baseline<ParentR, ParentC, BR, BC>);
    time_assign<ParentR, ParentC, BR, BC>(
        "assign_static", iters, assign_block_static<ParentR, ParentC, BR, BC>);
    time_extract<ParentR, ParentC, BR, BC>(
        "extract_base", iters, extract_block_baseline<ParentR, ParentC, BR, BC>);
    time_extract<ParentR, ParentC, BR, BC>(
        "extract_static", iters, extract_block_static<ParentR, ParentC, BR, BC>);
}

} // namespace

int main()
{
    const int iters = 1000000;
    run_case<4, 4, 2, 2>(iters);
    run_case<8, 8, 4, 4>(iters);
    run_case<12, 12, 6, 6>(iters);
    run_case<8, 4, 4, 2>(iters);
    run_case<12, 6, 6, 2>(iters);

    if (sink == 123456789.0) {
        std::cout << "sink\n";
    }
    return 0;
}
