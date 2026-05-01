#include "minisolver/matrix/matrix_defs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace minisolver;

namespace {

volatile double sink = 0.0;

template <int R, int C, typename Mat> double checksum(const Mat& m)
{
    double sum = 0.0;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            sum += m(r, c) * static_cast<double>(r * C + c + 1);
        }
    }
    return sum;
}

template <int R, int C, typename MatA, typename MatB>
double max_abs_diff(const MatA& a, const MatB& b)
{
    double max_diff = 0.0;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            max_diff = std::max(max_diff, std::abs(a(r, c) - b(r, c)));
        }
    }
    return max_diff;
}

template <int R, int C> void fill_matrix(MSMat<double, R, C>& m, double base)
{
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            const double sign = ((r + c) % 2 == 0) ? 1.0 : -1.0;
            m(r, c) = sign * (base + 0.013 * (r + 1) + 0.007 * (c + 1));
        }
    }
}

template <int N> void fill_vector(MSVec<double, N>& v, double base)
{
    for (int i = 0; i < N; ++i) {
        v(i) = base + 0.011 * static_cast<double>(i + 1);
    }
}

template <int NX, int NU, int NC> struct Inputs {
    MSMat<double, NX, NX> Q;
    MSMat<double, NU, NU> R;
    MSMat<double, NU, NX> H;
    MSVec<double, NX> q;
    MSVec<double, NU> r;
    MSMat<double, NC, NX> C;
    MSMat<double, NC, NU> D;
    MSVec<double, NC> sigma;
    MSVec<double, NC> grad;
    MSVec<double, NX> dx;
    MSVec<double, NU> du;
};

template <int NX, int NU, int NC> struct Outputs {
    MSMat<double, NX, NX> Q_bar;
    MSMat<double, NU, NU> R_bar;
    MSMat<double, NU, NX> H_bar;
    MSVec<double, NX> q_bar;
    MSVec<double, NU> r_bar;
};

template <int NX, int NU, int NC> Inputs<NX, NU, NC> make_inputs()
{
    Inputs<NX, NU, NC> in;
    fill_matrix(in.Q, 1.0);
    fill_matrix(in.R, 1.3);
    fill_matrix(in.H, 0.2);
    fill_matrix(in.C, 0.4);
    fill_matrix(in.D, 0.6);
    fill_vector(in.q, 0.7);
    fill_vector(in.r, 0.9);
    fill_vector(in.sigma, 1.1);
    fill_vector(in.grad, 0.3);
    fill_vector(in.dx, 0.5);
    fill_vector(in.du, 0.8);
    return in;
}

template <int NX, int NU, int NC>
void barrier_baseline(const Inputs<NX, NU, NC>& in, Outputs<NX, NU, NC>& out)
{
    MSDiag<double, NC> SigmaMat(in.sigma);

    MSMat<double, NC, NX> tempC = SigmaMat * in.C;
    MSMat<double, NC, NU> tempD = SigmaMat * in.D;

    out.Q_bar.noalias() = in.Q + in.C.transpose() * tempC;
    out.R_bar.noalias() = in.R + in.D.transpose() * tempD;
    out.H_bar.noalias() = in.H + in.D.transpose() * tempC;

    out.q_bar.noalias() = in.q + in.C.transpose() * in.grad;
    out.r_bar.noalias() = in.r + in.D.transpose() * in.grad;
}

template <int NX, int NU, int NC>
void barrier_fused(const Inputs<NX, NU, NC>& in, Outputs<NX, NU, NC>& out)
{
    out.Q_bar = in.Q;
    MatOps::weighted_mult_add_transA(out.Q_bar, in.C, in.sigma, in.C);

    out.R_bar = in.R;
    MatOps::weighted_mult_add_transA(out.R_bar, in.D, in.sigma, in.D);

    out.H_bar = in.H;
    MatOps::weighted_mult_add_transA(out.H_bar, in.D, in.sigma, in.C);

    out.q_bar = in.q;
    MatOps::mult_add_transA_v(out.q_bar, in.C, in.grad);

    out.r_bar = in.r;
    MatOps::mult_add_transA_v(out.r_bar, in.D, in.grad);
}

template <int NX, int NU, int NC> void validate_fused(const Inputs<NX, NU, NC>& in)
{
    Outputs<NX, NU, NC> baseline;
    Outputs<NX, NU, NC> fused;
    barrier_baseline(in, baseline);
    barrier_fused(in, fused);

    const double max_diff = std::max({
        max_abs_diff<NX, NX>(baseline.Q_bar, fused.Q_bar),
        max_abs_diff<NU, NU>(baseline.R_bar, fused.R_bar),
        max_abs_diff<NU, NX>(baseline.H_bar, fused.H_bar),
        max_abs_diff<NX, 1>(baseline.q_bar, fused.q_bar),
        max_abs_diff<NU, 1>(baseline.r_bar, fused.r_bar),
    });
    if (max_diff > 1e-12) {
        std::cerr << "barrier fused validation failed: NX=" << NX << " NU=" << NU << " NC=" << NC
                  << " max_diff=" << max_diff << "\n";
        std::abort();
    }
}

template <int NX, int NU, int NC>
void dual_step_baseline(const Inputs<NX, NU, NC>& in, MSVec<double, NC>& out)
{
    out.noalias() = in.C * in.dx + in.D * in.du;
}

template <int N>
void axpy_baseline(
    const MSVec<double, N>& base, const MSVec<double, N>& step, double alpha, MSVec<double, N>& out)
{
    out = base + step * alpha;
}

template <int N>
void axpy_fused_local(
    const MSVec<double, N>& base, const MSVec<double, N>& step, double alpha, MSVec<double, N>& out)
{
    for (int i = 0; i < N; ++i) {
        out(i) = base(i) + step(i) * alpha;
    }
}

template <int NX, int NU, int NC, typename Func>
double time_case(const std::string& label, int iters, const Inputs<NX, NU, NC>& in, Func&& func)
{
    Outputs<NX, NU, NC> out;
    func(in, out);
    sink += checksum<NX, NX>(out.Q_bar) + checksum<NU, NU>(out.R_bar) + checksum<NU, NX>(out.H_bar)
        + checksum<NX, 1>(out.q_bar) + checksum<NU, 1>(out.r_bar);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        func(in, out);
        sink += out.Q_bar(0, 0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);

    std::cout << std::left << std::setw(18) << label << " NX=" << std::setw(2) << NX
              << " NU=" << std::setw(2) << NU << " NC=" << std::setw(2) << NC
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int NX, int NU, int NC, typename Func>
double time_dual_step(
    const std::string& label, int iters, const Inputs<NX, NU, NC>& in, Func&& func)
{
    MSVec<double, NC> out;
    func(in, out);
    sink += checksum<NC, 1>(out);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        func(in, out);
        sink += out(0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);

    std::cout << std::left << std::setw(18) << label << " NX=" << std::setw(2) << NX
              << " NU=" << std::setw(2) << NU << " NC=" << std::setw(2) << NC
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int N, typename Func> double time_axpy(const std::string& label, int iters, Func&& func)
{
    MSVec<double, N> base;
    MSVec<double, N> step;
    MSVec<double, N> out;
    fill_vector(base, 0.6);
    fill_vector(step, -0.2);

    const double alpha = 0.731;
    func(base, step, alpha, out);
    sink += checksum<N, 1>(out);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        func(base, step, alpha, out);
        sink += checksum<N, 1>(out);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_iter = ns / static_cast<double>(iters);

    std::cout << std::left << std::setw(18) << label << " N=" << std::setw(2) << N
              << " ns/iter=" << std::fixed << std::setprecision(3) << ns_per_iter << "\n";
    return ns_per_iter;
}

template <int NX, int NU, int NC> void run_shape(int iters)
{
    const auto inputs = make_inputs<NX, NU, NC>();
    validate_fused(inputs);
    time_case<NX, NU, NC>("barrier_baseline", iters, inputs, barrier_baseline<NX, NU, NC>);
    time_case<NX, NU, NC>("barrier_fused", iters, inputs, barrier_fused<NX, NU, NC>);
    time_dual_step<NX, NU, NC>("dual_step_base", iters, inputs, dual_step_baseline<NX, NU, NC>);
    time_axpy<NX>("axpy_base_x", iters, axpy_baseline<NX>);
    time_axpy<NX>("axpy_fused_x", iters, axpy_fused_local<NX>);
    time_axpy<NU>("axpy_base_u", iters, axpy_baseline<NU>);
    time_axpy<NU>("axpy_fused_u", iters, axpy_fused_local<NU>);
    time_axpy<NC>("axpy_base_c", iters, axpy_baseline<NC>);
    time_axpy<NC>("axpy_fused_c", iters, axpy_fused_local<NC>);
}

} // namespace

int main()
{
    const int iters = 1000000;

    run_shape<4, 2, 5>(iters);
    run_shape<6, 2, 10>(iters);
    run_shape<10, 4, 16>(iters);

    if (sink == 123456789.0) {
        std::cerr << sink << "\n";
    }
    return 0;
}
