#include "minisolver/matrix/mini_matrix.h"
#include "minisolver/matrix/static_for.h"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using minisolver::MiniMatrix;

namespace {

volatile double sink = 0.0;

template <typename Mat> double checksum_minimatrix(const Mat& m)
{
    double sum = 0.0;
    for (int i = 0; i < Mat::Rows * Mat::Cols; ++i)
        sum += m.data[i] * static_cast<double>(i + 1);
    return sum;
}

template <typename Mat> double checksum_eigen(const Mat& m)
{
    double sum = 0.0;
    for (int i = 0; i < m.size(); ++i)
        sum += m.data()[i] * static_cast<double>(i + 1);
    return sum;
}

template <bool Unroll, int End> struct ForRange {
    template <typename Body> static inline void run(Body& body)
    {
        for (int i = 0; i < End; ++i)
            body(i);
    }
};

template <int End> struct ForRange<true, End> {
    template <typename Body> static inline void run(Body& body)
    {
        minisolver::matrix::StaticFor<0, End>::run(body);
    }
};

template <bool Unroll, int End> struct PrefixRange {
    template <typename Body> static inline void run(int count, Body& body)
    {
        for (int i = 0; i < count; ++i)
            body(i);
    }
};

template <int End> struct PrefixRange<true, End> {
    template <typename Body> static inline void run(int count, Body& body)
    {
        auto guarded = [&](int i) {
            if (i < count)
                body(i);
        };
        minisolver::matrix::StaticFor<0, End>::run(guarded);
    }
};

template <bool Unroll, int End> struct SuffixRange {
    template <typename Body> static inline void run(int begin, Body& body)
    {
        for (int i = begin; i < End; ++i)
            body(i);
    }
};

template <int End> struct SuffixRange<true, End> {
    template <typename Body> static inline void run(int begin, Body& body)
    {
        auto guarded = [&](int i) {
            if (i >= begin)
                body(i);
        };
        minisolver::matrix::StaticFor<0, End>::run(guarded);
    }
};

template <bool U0, bool U1, bool U2> std::string unroll_variant_name(const std::string& base)
{
    if (!U0 && !U1 && !U2)
        return base + "_loop";
    std::string name = base + "_unroll";
    if (U0)
        name += "_outer";
    if (U1)
        name += "_row";
    if (U2)
        name += "_inner";
    return name;
}

template <typename Fn>
double time_ns_per_iter(
    const std::string& kernel, int n, const std::string& variant, int iters, Fn fn)
{
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        fn(i);
    const auto end = std::chrono::high_resolution_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(end - start).count();
    const double per_iter = ns / static_cast<double>(iters);
    std::cout << kernel << "," << n << "," << variant << "," << per_iter << "\n";
    return per_iter;
}

template <bool UI, bool UJ, bool UK, int R, int K, int C>
inline void matmul_variant(MiniMatrix<double, R, C>& out, const MiniMatrix<double, R, K>& a,
    const MiniMatrix<double, K, C>& b)
{
    auto body_i = [&](int i) {
        auto body_j = [&](int j) {
            double sum = 0.0;
            auto body_k = [&](int k) { sum += a(i, k) * b(k, j); };
            ForRange<UK, K>::run(body_k);
            out(i, j) = sum;
        };
        ForRange<UJ, C>::run(body_j);
    };
    ForRange<UI, R>::run(body_i);
}

template <bool UI, bool UJ, bool UK, int RA, int R, int C>
inline void add_at_mul_b_variant(MiniMatrix<double, R, C>& out, const MiniMatrix<double, RA, R>& a,
    const MiniMatrix<double, RA, C>& b)
{
    auto body_i = [&](int i) {
        auto body_j = [&](int j) {
            double sum = 0.0;
            auto body_k = [&](int k) { sum += a(k, i) * b(k, j); };
            ForRange<UK, RA>::run(body_k);
            out(i, j) += sum;
        };
        ForRange<UJ, C>::run(body_j);
    };
    ForRange<UI, R>::run(body_i);
}

template <int R, int K, int C>
void fill_minimatrix_inputs(
    std::array<MiniMatrix<double, R, K>, 64>& as, std::array<MiniMatrix<double, K, C>, 64>& bs)
{
    for (int batch = 0; batch < 64; ++batch) {
        for (int i = 0; i < R * K; ++i)
            as[batch].data[i] = 0.01 * static_cast<double>((batch + 1) * (i + 1));
        for (int i = 0; i < K * C; ++i)
            bs[batch].data[i] = 0.02 * static_cast<double>((batch + 3) * (i + 1));
    }
}

template <int R, int K, int C>
void fill_eigen_inputs(std::array<Eigen::Matrix<double, R, K>, 64>& as,
    std::array<Eigen::Matrix<double, K, C>, 64>& bs)
{
    for (int batch = 0; batch < 64; ++batch) {
        for (int r = 0; r < R; ++r)
            for (int k = 0; k < K; ++k)
                as[batch](r, k) = 0.01 * static_cast<double>((batch + 1) * (r * K + k + 1));
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < C; ++c)
                bs[batch](k, c) = 0.02 * static_cast<double>((batch + 3) * (k * C + c + 1));
    }
}

inline double lower_seed(int batch, int row, int col)
{
    if (col > row)
        return 0.0;
    if (row == col)
        return 2.0 + 0.03 * static_cast<double>((batch + row) % 7);
    return 0.01 * static_cast<double>((batch + 1) * (row + 1) + (col + 1));
}

inline double general_seed(int batch, int row, int col, int n)
{
    if (row == col)
        return 2.0 * static_cast<double>(n) + 0.01 * static_cast<double>(batch + row + 1);
    const double sign = ((row + col + batch) & 1) ? -1.0 : 1.0;
    return sign * 0.02 * static_cast<double>((batch + 1) + (row + 1) * (col + 2));
}

template <int N> void fill_minimatrix_spd(std::array<MiniMatrix<double, N, N>, 64>& mats)
{
    for (int batch = 0; batch < 64; ++batch) {
        std::array<double, N * N> l;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                l[i * N + j] = lower_seed(batch, i, j);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k)
                    sum += l[i * N + k] * l[j * N + k];
                if (i == j)
                    sum += 0.25;
                mats[batch](i, j) = sum;
            }
        }
    }
}

template <int N> void fill_eigen_spd(std::array<Eigen::Matrix<double, N, N>, 64>& mats)
{
    for (int batch = 0; batch < 64; ++batch) {
        Eigen::Matrix<double, N, N> l = Eigen::Matrix<double, N, N>::Zero();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                l(i, j) = lower_seed(batch, i, j);
        mats[batch].noalias() = l * l.transpose();
        mats[batch].diagonal().array() += 0.25;
    }
}

template <int N> void fill_minimatrix_general(std::array<MiniMatrix<double, N, N>, 64>& mats)
{
    for (int batch = 0; batch < 64; ++batch) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                mats[batch](i, j) = general_seed(batch, i, j, N);
    }
}

template <int N> void fill_eigen_general(std::array<Eigen::Matrix<double, N, N>, 64>& mats)
{
    for (int batch = 0; batch < 64; ++batch) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                mats[batch](i, j) = general_seed(batch, i, j, N);
    }
}

inline double rhs_seed(int batch, int row, int col)
{
    return 0.03 * static_cast<double>((batch + 2) * (row + 1))
        + 0.01 * static_cast<double>((col + 1) * (row + 2));
}

template <int N> void fill_minimatrix_rhs_vec(std::array<MiniMatrix<double, N, 1>, 64>& rhs)
{
    for (int batch = 0; batch < 64; ++batch)
        for (int i = 0; i < N; ++i)
            rhs[batch](i) = rhs_seed(batch, i, 0);
}

template <int N> void fill_eigen_rhs_vec(std::array<Eigen::Matrix<double, N, 1>, 64>& rhs)
{
    for (int batch = 0; batch < 64; ++batch)
        for (int i = 0; i < N; ++i)
            rhs[batch](i) = rhs_seed(batch, i, 0);
}

template <int N> void fill_minimatrix_rhs_mat(std::array<MiniMatrix<double, N, N>, 64>& rhs)
{
    for (int batch = 0; batch < 64; ++batch)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                rhs[batch](i, j) = rhs_seed(batch, i, j);
}

template <int N> void fill_eigen_rhs_mat(std::array<Eigen::Matrix<double, N, N>, 64>& rhs)
{
    for (int batch = 0; batch < 64; ++batch)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                rhs[batch](i, j) = rhs_seed(batch, i, j);
}

template <int N>
double residual_inf_vec(const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, 1>& x,
    const MiniMatrix<double, N, 1>& b)
{
    double max_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j)
            sum += a(i, j) * x(j);
        max_abs = std::max(max_abs, std::abs(sum - b(i)));
    }
    return max_abs;
}

template <int N>
double residual_inf_mat(const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, N>& x,
    const MiniMatrix<double, N, N>& b)
{
    double max_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < N; ++c) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j)
                sum += a(i, j) * x(j, c);
            max_abs = std::max(max_abs, std::abs(sum - b(i, c)));
        }
    }
    return max_abs;
}

template <typename EigenMat, typename EigenRhs>
double residual_inf_eigen(const EigenMat& a, const EigenRhs& x, const EigenRhs& b)
{
    return (a * x - b).cwiseAbs().maxCoeff();
}

inline void require_residual_ok(
    const std::string& kernel, int n, const std::string& variant, double residual, double tol)
{
    if (!(residual <= tol)) {
        std::cerr << "residual check failed: kernel=" << kernel << " n=" << n
                  << " variant=" << variant << " residual=" << residual << " tol=" << tol << "\n";
        std::abort();
    }
}

template <int N> double minimatrix_llt_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> l;
    l.fill(0.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k)
                sum += l[i * N + k] * l[j * N + k];
            if (i == j) {
                const double val = a(i, i) - sum;
                if (val <= 0.0)
                    return -1.0;
                l[i * N + j] = std::sqrt(val);
            } else {
                l[i * N + j] = (a(i, j) - sum) / l[j * N + j];
            }
        }
    }

    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += l[i] * static_cast<double>(i + 1);
    return sum;
}

template <int N> double minimatrix_ldlt_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> l;
    std::array<double, N> d;
    l.fill(0.0);
    d.fill(0.0);

    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < j; ++k)
            l[j * N + j] += l[j * N + k] * l[j * N + k] * d[k];
        d[j] = a(j, j) - l[j * N + j];
        if (std::abs(d[j]) < 1e-14)
            return -1.0;
        l[j * N + j] = 1.0;

        for (int i = j + 1; i < N; ++i) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k)
                sum += l[i * N + k] * l[j * N + k] * d[k];
            l[i * N + j] = (a(i, j) - sum) / d[j];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += l[i] * static_cast<double>(i + 1);
    for (int i = 0; i < N; ++i)
        sum += d[i] * static_cast<double>(i + 1);
    return sum;
}

template <int N> double minimatrix_lu_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> lu;
    std::array<int, N> perm;
    for (int i = 0; i < N * N; ++i)
        lu[i] = a.data[i];
    for (int i = 0; i < N; ++i)
        perm[i] = i;

    for (int k = 0; k < N; ++k) {
        int max_row = k;
        double max_val = std::abs(lu[k * N + k]);
        for (int i = k + 1; i < N; ++i) {
            const double v = std::abs(lu[i * N + k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }
        if (max_val < 1e-14)
            return -1.0;
        if (max_row != k) {
            std::swap(perm[k], perm[max_row]);
            for (int j = 0; j < N; ++j)
                std::swap(lu[k * N + j], lu[max_row * N + j]);
        }
        for (int i = k + 1; i < N; ++i) {
            const double factor = lu[i * N + k] / lu[k * N + k];
            lu[i * N + k] = factor;
            for (int j = k + 1; j < N; ++j)
                lu[i * N + j] -= factor * lu[k * N + j];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += lu[i] * static_cast<double>(i + 1);
    for (int i = 0; i < N; ++i)
        sum += static_cast<double>(perm[i] + 1);
    return sum;
}

template <int N> double minimatrix_qr_mgs_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> q;
    std::array<double, N * N> r;
    std::array<double, N> v;
    q.fill(0.0);
    r.fill(0.0);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i)
            v[i] = a(i, j);

        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < N; ++i)
                dot += q[i * N + k] * v[i];
            r[k * N + j] = dot;
            for (int i = 0; i < N; ++i)
                v[i] -= dot * q[i * N + k];
        }

        double norm_sq = 0.0;
        for (int i = 0; i < N; ++i)
            norm_sq += v[i] * v[i];
        const double norm = std::sqrt(norm_sq);
        if (norm < 1e-14)
            return -1.0;
        r[j * N + j] = norm;
        for (int i = 0; i < N; ++i)
            q[i * N + j] = v[i] / norm;
    }

    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += (q[i] + r[i]) * static_cast<double>(i + 1);
    return sum;
}

template <bool UO, bool UR, bool UI, int N>
double minimatrix_llt_variant_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> l;
    l.fill(0.0);
    bool success = true;

    auto body_outer = [&](int i) {
        if (!success)
            return;
        auto body_row = [&](int j) {
            if (!success)
                return;
            double sum = 0.0;
            auto body_inner = [&](int k) { sum += l[i * N + k] * l[j * N + k]; };
            PrefixRange<UI, N>::run(j, body_inner);

            if (i == j) {
                const double val = a(i, i) - sum;
                if (val <= 0.0) {
                    success = false;
                    return;
                }
                l[i * N + j] = std::sqrt(val);
            } else {
                l[i * N + j] = (a(i, j) - sum) / l[j * N + j];
            }
        };
        PrefixRange<UR, N>::run(i + 1, body_row);
    };
    ForRange<UO, N>::run(body_outer);

    if (!success)
        return -1.0;
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += l[i] * static_cast<double>(i + 1);
    return sum;
}

template <bool UO, bool UR, bool UI, int N>
double minimatrix_ldlt_variant_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> l;
    std::array<double, N> d;
    l.fill(0.0);
    d.fill(0.0);
    bool success = true;

    auto body_outer = [&](int j) {
        if (!success)
            return;

        double diag_sum = 0.0;
        auto body_diag = [&](int k) { diag_sum += l[j * N + k] * l[j * N + k] * d[k]; };
        PrefixRange<UI, N>::run(j, body_diag);
        d[j] = a(j, j) - diag_sum;
        if (std::abs(d[j]) < 1e-14) {
            success = false;
            return;
        }
        l[j * N + j] = 1.0;

        auto body_row = [&](int i) {
            double sum = 0.0;
            auto body_inner = [&](int k) { sum += l[i * N + k] * l[j * N + k] * d[k]; };
            PrefixRange<UI, N>::run(j, body_inner);
            l[i * N + j] = (a(i, j) - sum) / d[j];
        };
        SuffixRange<UR, N>::run(j + 1, body_row);
    };
    ForRange<UO, N>::run(body_outer);

    if (!success)
        return -1.0;
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += l[i] * static_cast<double>(i + 1);
    for (int i = 0; i < N; ++i)
        sum += d[i] * static_cast<double>(i + 1);
    return sum;
}

template <bool UO, bool UR, bool UI, int N>
double minimatrix_lu_variant_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> lu;
    std::array<int, N> perm;
    for (int i = 0; i < N * N; ++i)
        lu[i] = a.data[i];
    for (int i = 0; i < N; ++i)
        perm[i] = i;

    bool success = true;
    auto body_outer = [&](int k) {
        if (!success)
            return;

        int max_row = k;
        double max_val = std::abs(lu[k * N + k]);
        auto body_pivot = [&](int i) {
            const double v = std::abs(lu[i * N + k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        };
        SuffixRange<UR, N>::run(k + 1, body_pivot);

        if (max_val < 1e-14) {
            success = false;
            return;
        }
        if (max_row != k) {
            std::swap(perm[k], perm[max_row]);
            auto body_swap_col = [&](int j) { std::swap(lu[k * N + j], lu[max_row * N + j]); };
            ForRange<UI, N>::run(body_swap_col);
        }

        auto body_elim_row = [&](int i) {
            const double factor = lu[i * N + k] / lu[k * N + k];
            lu[i * N + k] = factor;
            auto body_update_col = [&](int j) { lu[i * N + j] -= factor * lu[k * N + j]; };
            SuffixRange<UI, N>::run(k + 1, body_update_col);
        };
        SuffixRange<UR, N>::run(k + 1, body_elim_row);
    };
    ForRange<UO, N>::run(body_outer);

    if (!success)
        return -1.0;
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += lu[i] * static_cast<double>(i + 1);
    for (int i = 0; i < N; ++i)
        sum += static_cast<double>(perm[i] + 1);
    return sum;
}

template <bool UO, bool UR, bool UI, int N>
double minimatrix_qr_mgs_variant_checksum(const MiniMatrix<double, N, N>& a)
{
    std::array<double, N * N> q;
    std::array<double, N * N> r;
    std::array<double, N> v;
    q.fill(0.0);
    r.fill(0.0);
    bool success = true;

    auto body_outer = [&](int j) {
        if (!success)
            return;

        auto body_copy = [&](int i) { v[i] = a(i, j); };
        ForRange<UR, N>::run(body_copy);

        auto body_prev_col = [&](int k) {
            double dot = 0.0;
            auto body_dot = [&](int i) { dot += q[i * N + k] * v[i]; };
            ForRange<UI, N>::run(body_dot);
            r[k * N + j] = dot;
            auto body_update = [&](int i) { v[i] -= dot * q[i * N + k]; };
            ForRange<UI, N>::run(body_update);
        };
        PrefixRange<UR, N>::run(j, body_prev_col);

        double norm_sq = 0.0;
        auto body_norm = [&](int i) { norm_sq += v[i] * v[i]; };
        ForRange<UI, N>::run(body_norm);
        const double norm = std::sqrt(norm_sq);
        if (norm < 1e-14) {
            success = false;
            return;
        }
        r[j * N + j] = norm;
        auto body_assign = [&](int i) { q[i * N + j] = v[i] / norm; };
        ForRange<UI, N>::run(body_assign);
    };
    ForRange<UO, N>::run(body_outer);

    if (!success)
        return -1.0;
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i)
        sum += (q[i] + r[i]) * static_cast<double>(i + 1);
    return sum;
}

template <bool UO, bool UR, bool UI, int N>
bool minimatrix_ldlt_factor_variant(
    const MiniMatrix<double, N, N>& a, std::array<double, N * N>& l, std::array<double, N>& d)
{
    l.fill(0.0);
    d.fill(0.0);
    bool success = true;

    auto body_outer = [&](int j) {
        if (!success)
            return;

        double diag_sum = 0.0;
        auto body_diag = [&](int k) { diag_sum += l[j * N + k] * l[j * N + k] * d[k]; };
        PrefixRange<UI, N>::run(j, body_diag);
        d[j] = a(j, j) - diag_sum;
        if (std::abs(d[j]) < 1e-14) {
            success = false;
            return;
        }
        l[j * N + j] = 1.0;

        auto body_row = [&](int i) {
            double sum = 0.0;
            auto body_inner = [&](int k) { sum += l[i * N + k] * l[j * N + k] * d[k]; };
            PrefixRange<UI, N>::run(j, body_inner);
            l[i * N + j] = (a(i, j) - sum) / d[j];
        };
        SuffixRange<UR, N>::run(j + 1, body_row);
    };
    ForRange<UO, N>::run(body_outer);
    return success;
}

template <int N>
void minimatrix_ldlt_solve_vec_from_factor(const std::array<double, N * N>& l,
    const std::array<double, N>& d, const MiniMatrix<double, N, 1>& b, MiniMatrix<double, N, 1>& x)
{
    std::array<double, N> y;
    std::array<double, N> z;

    for (int i = 0; i < N; ++i) {
        double sum = b(i);
        for (int k = 0; k < i; ++k)
            sum -= l[i * N + k] * y[k];
        y[i] = sum;
        z[i] = y[i] / d[i];
    }

    for (int i = N - 1; i >= 0; --i) {
        double sum = z[i];
        for (int k = i + 1; k < N; ++k)
            sum -= l[k * N + i] * x(k);
        x(i) = sum;
    }
}

template <int N>
void minimatrix_ldlt_solve_mat_from_factor(const std::array<double, N * N>& l,
    const std::array<double, N>& d, const MiniMatrix<double, N, N>& b, MiniMatrix<double, N, N>& x)
{
    std::array<double, N * N> y;
    std::array<double, N * N> z;

    for (int i = 0; i < N; ++i) {
        for (int col = 0; col < N; ++col) {
            double sum = b(i, col);
            for (int k = 0; k < i; ++k)
                sum -= l[i * N + k] * y[k * N + col];
            y[i * N + col] = sum;
            z[i * N + col] = sum / d[i];
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        for (int col = 0; col < N; ++col) {
            double sum = z[i * N + col];
            for (int k = i + 1; k < N; ++k)
                sum -= l[k * N + i] * x(k, col);
            x(i, col) = sum;
        }
    }
}

template <bool UO, bool UR, bool UI, int N>
bool minimatrix_ldlt_solve_vec_variant(const MiniMatrix<double, N, N>& a,
    const MiniMatrix<double, N, 1>& b, MiniMatrix<double, N, 1>& x)
{
    std::array<double, N * N> l;
    std::array<double, N> d;
    if (!minimatrix_ldlt_factor_variant<UO, UR, UI, N>(a, l, d))
        return false;
    minimatrix_ldlt_solve_vec_from_factor<N>(l, d, b, x);
    return true;
}

template <bool UO, bool UR, bool UI, int N>
bool minimatrix_ldlt_solve_mat_variant(const MiniMatrix<double, N, N>& a,
    const MiniMatrix<double, N, N>& b, MiniMatrix<double, N, N>& x)
{
    std::array<double, N * N> l;
    std::array<double, N> d;
    if (!minimatrix_ldlt_factor_variant<UO, UR, UI, N>(a, l, d))
        return false;
    minimatrix_ldlt_solve_mat_from_factor<N>(l, d, b, x);
    return true;
}

template <bool UO, bool UR, bool UI, int N>
bool minimatrix_lu_factor_variant(
    const MiniMatrix<double, N, N>& a, std::array<double, N * N>& lu, std::array<int, N>& perm)
{
    for (int i = 0; i < N * N; ++i)
        lu[i] = a.data[i];
    for (int i = 0; i < N; ++i)
        perm[i] = i;

    bool success = true;
    auto body_outer = [&](int k) {
        if (!success)
            return;

        int max_row = k;
        double max_val = std::abs(lu[k * N + k]);
        auto body_pivot = [&](int i) {
            const double v = std::abs(lu[i * N + k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        };
        SuffixRange<UR, N>::run(k + 1, body_pivot);

        if (max_val < 1e-14) {
            success = false;
            return;
        }
        if (max_row != k) {
            std::swap(perm[k], perm[max_row]);
            auto body_swap_col = [&](int j) { std::swap(lu[k * N + j], lu[max_row * N + j]); };
            ForRange<UI, N>::run(body_swap_col);
        }

        auto body_elim_row = [&](int i) {
            const double factor = lu[i * N + k] / lu[k * N + k];
            lu[i * N + k] = factor;
            auto body_update_col = [&](int j) { lu[i * N + j] -= factor * lu[k * N + j]; };
            SuffixRange<UI, N>::run(k + 1, body_update_col);
        };
        SuffixRange<UR, N>::run(k + 1, body_elim_row);
    };
    ForRange<UO, N>::run(body_outer);
    return success;
}

template <int N>
void minimatrix_lu_solve_vec_from_factor(const std::array<double, N * N>& lu,
    const std::array<int, N>& perm, const MiniMatrix<double, N, 1>& b, MiniMatrix<double, N, 1>& x)
{
    std::array<double, N> y;
    for (int i = 0; i < N; ++i) {
        double sum = b(perm[i]);
        for (int j = 0; j < i; ++j)
            sum -= lu[i * N + j] * y[j];
        y[i] = sum;
    }

    for (int i = N - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < N; ++j)
            sum -= lu[i * N + j] * x(j);
        x(i) = sum / lu[i * N + i];
    }
}

template <int N>
void minimatrix_lu_solve_mat_from_factor(const std::array<double, N * N>& lu,
    const std::array<int, N>& perm, const MiniMatrix<double, N, N>& b, MiniMatrix<double, N, N>& x)
{
    std::array<double, N * N> y;
    for (int i = 0; i < N; ++i) {
        for (int col = 0; col < N; ++col) {
            double sum = b(perm[i], col);
            for (int j = 0; j < i; ++j)
                sum -= lu[i * N + j] * y[j * N + col];
            y[i * N + col] = sum;
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        for (int col = 0; col < N; ++col) {
            double sum = y[i * N + col];
            for (int j = i + 1; j < N; ++j)
                sum -= lu[i * N + j] * x(j, col);
            x(i, col) = sum / lu[i * N + i];
        }
    }
}

template <bool UO, bool UR, bool UI, int N>
bool minimatrix_lu_solve_vec_variant(const MiniMatrix<double, N, N>& a,
    const MiniMatrix<double, N, 1>& b, MiniMatrix<double, N, 1>& x)
{
    std::array<double, N * N> lu;
    std::array<int, N> perm;
    if (!minimatrix_lu_factor_variant<UO, UR, UI, N>(a, lu, perm))
        return false;
    minimatrix_lu_solve_vec_from_factor<N>(lu, perm, b, x);
    return true;
}

template <bool UO, bool UR, bool UI, int N>
bool minimatrix_lu_solve_mat_variant(const MiniMatrix<double, N, N>& a,
    const MiniMatrix<double, N, N>& b, MiniMatrix<double, N, N>& x)
{
    std::array<double, N * N> lu;
    std::array<int, N> perm;
    if (!minimatrix_lu_factor_variant<UO, UR, UI, N>(a, lu, perm))
        return false;
    minimatrix_lu_solve_mat_from_factor<N>(lu, perm, b, x);
    return true;
}

template <int N>
bool minimatrix_qr_mgs_factor(
    const MiniMatrix<double, N, N>& a, std::array<double, N * N>& q, std::array<double, N * N>& r)
{
    std::array<double, N> v;
    q.fill(0.0);
    r.fill(0.0);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i)
            v[i] = a(i, j);

        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < N; ++i)
                dot += q[i * N + k] * v[i];
            r[k * N + j] = dot;
            for (int i = 0; i < N; ++i)
                v[i] -= dot * q[i * N + k];
        }

        double norm_sq = 0.0;
        for (int i = 0; i < N; ++i)
            norm_sq += v[i] * v[i];
        const double norm = std::sqrt(norm_sq);
        if (norm < 1e-14)
            return false;
        r[j * N + j] = norm;
        for (int i = 0; i < N; ++i)
            q[i * N + j] = v[i] / norm;
    }
    return true;
}

template <int N>
void minimatrix_qr_mgs_solve_vec_from_factor(const std::array<double, N * N>& q,
    const std::array<double, N * N>& r, const MiniMatrix<double, N, 1>& b,
    MiniMatrix<double, N, 1>& x)
{
    std::array<double, N> y;
    for (int j = 0; j < N; ++j) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
            sum += q[i * N + j] * b(i);
        y[j] = sum;
    }

    for (int i = N - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < N; ++j)
            sum -= r[i * N + j] * x(j);
        x(i) = sum / r[i * N + i];
    }
}

template <int N>
void minimatrix_qr_mgs_solve_mat_from_factor(const std::array<double, N * N>& q,
    const std::array<double, N * N>& r, const MiniMatrix<double, N, N>& b,
    MiniMatrix<double, N, N>& x)
{
    std::array<double, N * N> y;
    for (int j = 0; j < N; ++j) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int i = 0; i < N; ++i)
                sum += q[i * N + j] * b(i, col);
            y[j * N + col] = sum;
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        for (int col = 0; col < N; ++col) {
            double sum = y[i * N + col];
            for (int j = i + 1; j < N; ++j)
                sum -= r[i * N + j] * x(j, col);
            x(i, col) = sum / r[i * N + i];
        }
    }
}

template <int N>
bool minimatrix_qr_mgs_solve_vec(const MiniMatrix<double, N, N>& a,
    const MiniMatrix<double, N, 1>& b, MiniMatrix<double, N, 1>& x)
{
    std::array<double, N * N> q;
    std::array<double, N * N> r;
    if (!minimatrix_qr_mgs_factor<N>(a, q, r))
        return false;
    minimatrix_qr_mgs_solve_vec_from_factor<N>(q, r, b, x);
    return true;
}

template <int N>
bool minimatrix_qr_mgs_solve_mat(const MiniMatrix<double, N, N>& a,
    const MiniMatrix<double, N, N>& b, MiniMatrix<double, N, N>& x)
{
    std::array<double, N * N> q;
    std::array<double, N * N> r;
    if (!minimatrix_qr_mgs_factor<N>(a, q, r))
        return false;
    minimatrix_qr_mgs_solve_mat_from_factor<N>(q, r, b, x);
    return true;
}

template <bool UI, bool UJ, bool UK, int N>
void bench_matmul_variant(const std::string& variant, int iters)
{
    std::array<MiniMatrix<double, N, N>, 64> as;
    std::array<MiniMatrix<double, N, N>, 64> bs;
    fill_minimatrix_inputs<N, N, N>(as, bs);

    time_ns_per_iter("matmul", N, variant, iters, [&](int i) {
        const MiniMatrix<double, N, N>& a = as[i & 63];
        const MiniMatrix<double, N, N>& b = bs[(i * 7) & 63];
        MiniMatrix<double, N, N> out;
        matmul_variant<UI, UJ, UK>(out, a, b);
        sink += checksum_minimatrix(out);
    });
}

template <int N> void bench_matmul_current(int iters)
{
    std::array<MiniMatrix<double, N, N>, 64> as;
    std::array<MiniMatrix<double, N, N>, 64> bs;
    fill_minimatrix_inputs<N, N, N>(as, bs);

    time_ns_per_iter("matmul", N, "minimatrix_current", iters, [&](int i) {
        const MiniMatrix<double, N, N>& a = as[i & 63];
        const MiniMatrix<double, N, N>& b = bs[(i * 7) & 63];
        MiniMatrix<double, N, N> out = a * b;
        sink += checksum_minimatrix(out);
    });
}

template <int N> void bench_matmul_eigen(int iters)
{
    typedef Eigen::Matrix<double, N, N> Mat;
    std::array<Mat, 64> as;
    std::array<Mat, 64> bs;
    fill_eigen_inputs<N, N, N>(as, bs);

    time_ns_per_iter("matmul", N, "eigen", iters, [&](int i) {
        const Mat& a = as[i & 63];
        const Mat& b = bs[(i * 7) & 63];
        Mat out;
        out.noalias() = a * b;
        sink += checksum_eigen(out);
    });
}

template <bool UI, bool UJ, bool UK, int N>
void bench_add_at_mul_b_variant(const std::string& variant, int iters)
{
    std::array<MiniMatrix<double, N, N>, 64> as;
    std::array<MiniMatrix<double, N, N>, 64> bs;
    MiniMatrix<double, N, N> out;
    fill_minimatrix_inputs<N, N, N>(as, bs);

    time_ns_per_iter("add_At_mul_B", N, variant, iters, [&](int i) {
        const MiniMatrix<double, N, N>& a = as[i & 63];
        const MiniMatrix<double, N, N>& b = bs[(i * 7) & 63];
        add_at_mul_b_variant<UI, UJ, UK>(out, a, b);
        sink += checksum_minimatrix(out);
    });
}

template <int N> void bench_add_at_mul_b_current(int iters)
{
    std::array<MiniMatrix<double, N, N>, 64> as;
    std::array<MiniMatrix<double, N, N>, 64> bs;
    MiniMatrix<double, N, N> out;
    fill_minimatrix_inputs<N, N, N>(as, bs);

    time_ns_per_iter("add_At_mul_B", N, "minimatrix_current", iters, [&](int i) {
        const MiniMatrix<double, N, N>& a = as[i & 63];
        const MiniMatrix<double, N, N>& b = bs[(i * 7) & 63];
        out.add_At_mul_B(a, b);
        sink += checksum_minimatrix(out);
    });
}

template <int N> void bench_add_at_mul_b_eigen(int iters)
{
    typedef Eigen::Matrix<double, N, N> Mat;
    std::array<Mat, 64> as;
    std::array<Mat, 64> bs;
    Mat out = Mat::Zero();
    fill_eigen_inputs<N, N, N>(as, bs);

    time_ns_per_iter("add_At_mul_B", N, "eigen", iters, [&](int i) {
        const Mat& a = as[i & 63];
        const Mat& b = bs[(i * 7) & 63];
        out.noalias() += a.transpose() * b;
        sink += checksum_eigen(out);
    });
}

template <bool UO, bool UR, bool UI, int N>
void bench_decomp_llt_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_llt");
    time_ns_per_iter("decomp_spd", N, variant, iters,
        [&](int i) { sink += minimatrix_llt_variant_checksum<UO, UR, UI, N>(mats[i & 63]); });
}

template <bool UO, bool UR, bool UI, int N>
void bench_decomp_ldlt_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_ldlt");
    time_ns_per_iter("decomp_spd", N, variant, iters,
        [&](int i) { sink += minimatrix_ldlt_variant_checksum<UO, UR, UI, N>(mats[i & 63]); });
}

template <bool UO, bool UR, bool UI, int N>
void bench_decomp_lu_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_lu_partial");
    time_ns_per_iter("decomp_general", N, variant, iters,
        [&](int i) { sink += minimatrix_lu_variant_checksum<UO, UR, UI, N>(mats[i & 63]); });
}

template <bool UO, bool UR, bool UI, int N>
void bench_decomp_qr_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_qr_mgs");
    time_ns_per_iter("decomp_general", N, variant, iters,
        [&](int i) { sink += minimatrix_qr_mgs_variant_checksum<UO, UR, UI, N>(mats[i & 63]); });
}

template <int N>
void bench_decomp_minimatrix_spd_variants(
    const std::array<MiniMatrix<double, N, N>, 64>& mats, int iters)
{
    bench_decomp_llt_variant<false, false, false, N>(mats, iters);
    bench_decomp_llt_variant<true, false, false, N>(mats, iters);
    bench_decomp_llt_variant<false, true, false, N>(mats, iters);
    bench_decomp_llt_variant<false, false, true, N>(mats, iters);
    bench_decomp_llt_variant<true, true, false, N>(mats, iters);
    bench_decomp_llt_variant<true, false, true, N>(mats, iters);
    bench_decomp_llt_variant<false, true, true, N>(mats, iters);
    bench_decomp_llt_variant<true, true, true, N>(mats, iters);

    bench_decomp_ldlt_variant<false, false, false, N>(mats, iters);
    bench_decomp_ldlt_variant<true, false, false, N>(mats, iters);
    bench_decomp_ldlt_variant<false, true, false, N>(mats, iters);
    bench_decomp_ldlt_variant<false, false, true, N>(mats, iters);
    bench_decomp_ldlt_variant<true, true, false, N>(mats, iters);
    bench_decomp_ldlt_variant<true, false, true, N>(mats, iters);
    bench_decomp_ldlt_variant<false, true, true, N>(mats, iters);
    bench_decomp_ldlt_variant<true, true, true, N>(mats, iters);
}

template <int N>
void bench_decomp_minimatrix_general_variants(
    const std::array<MiniMatrix<double, N, N>, 64>& mats, int iters)
{
    bench_decomp_lu_variant<false, false, false, N>(mats, iters);
    bench_decomp_lu_variant<true, false, false, N>(mats, iters);
    bench_decomp_lu_variant<false, true, false, N>(mats, iters);
    bench_decomp_lu_variant<false, false, true, N>(mats, iters);
    bench_decomp_lu_variant<true, true, false, N>(mats, iters);
    bench_decomp_lu_variant<true, false, true, N>(mats, iters);
    bench_decomp_lu_variant<false, true, true, N>(mats, iters);
    bench_decomp_lu_variant<true, true, true, N>(mats, iters);

    bench_decomp_qr_variant<false, false, false, N>(mats, iters);
    bench_decomp_qr_variant<true, false, false, N>(mats, iters);
    bench_decomp_qr_variant<false, true, false, N>(mats, iters);
    bench_decomp_qr_variant<false, false, true, N>(mats, iters);
    bench_decomp_qr_variant<true, true, false, N>(mats, iters);
    bench_decomp_qr_variant<true, false, true, N>(mats, iters);
    bench_decomp_qr_variant<false, true, true, N>(mats, iters);
    bench_decomp_qr_variant<true, true, true, N>(mats, iters);
}

template <int N> void bench_decomp_spd_all(int iters)
{
    typedef Eigen::Matrix<double, N, N> EigenMat;
    std::array<MiniMatrix<double, N, N>, 64> mini_mats;
    std::array<EigenMat, 64> eigen_mats;
    fill_minimatrix_spd<N>(mini_mats);
    fill_eigen_spd<N>(eigen_mats);

    time_ns_per_iter("decomp_spd", N, "minimatrix_llt_current", iters, [&](int i) {
        minisolver::MiniLLT<double, N> llt(mini_mats[i & 63]);
        sink += static_cast<double>(llt.info());
    });

    bench_decomp_minimatrix_spd_variants<N>(mini_mats, iters);

    time_ns_per_iter("decomp_spd", N, "eigen_llt", iters, [&](int i) {
        Eigen::LLT<EigenMat> llt(eigen_mats[i & 63]);
        sink += checksum_eigen(llt.matrixLLT());
    });

    time_ns_per_iter("decomp_spd", N, "eigen_ldlt", iters, [&](int i) {
        Eigen::LDLT<EigenMat> ldlt(eigen_mats[i & 63]);
        sink += checksum_eigen(ldlt.matrixLDLT()) + checksum_eigen(ldlt.vectorD());
    });

    time_ns_per_iter("decomp_spd", N, "eigen_partial_piv_lu", iters, [&](int i) {
        Eigen::PartialPivLU<EigenMat> lu(eigen_mats[i & 63]);
        sink += checksum_eigen(lu.matrixLU());
    });

    time_ns_per_iter("decomp_spd", N, "eigen_householder_qr", iters, [&](int i) {
        Eigen::HouseholderQR<EigenMat> qr(eigen_mats[i & 63]);
        sink += checksum_eigen(qr.matrixQR());
    });
}

template <int N> void bench_decomp_general_all(int iters)
{
    typedef Eigen::Matrix<double, N, N> EigenMat;
    std::array<MiniMatrix<double, N, N>, 64> mini_mats;
    std::array<EigenMat, 64> eigen_mats;
    fill_minimatrix_general<N>(mini_mats);
    fill_eigen_general<N>(eigen_mats);

    bench_decomp_minimatrix_general_variants<N>(mini_mats, iters);

    time_ns_per_iter("decomp_general", N, "eigen_partial_piv_lu", iters, [&](int i) {
        Eigen::PartialPivLU<EigenMat> lu(eigen_mats[i & 63]);
        sink += checksum_eigen(lu.matrixLU());
    });

    time_ns_per_iter("decomp_general", N, "eigen_full_piv_lu", iters, [&](int i) {
        Eigen::FullPivLU<EigenMat> lu(eigen_mats[i & 63]);
        sink += checksum_eigen(lu.matrixLU());
    });

    time_ns_per_iter("decomp_general", N, "eigen_householder_qr", iters, [&](int i) {
        Eigen::HouseholderQR<EigenMat> qr(eigen_mats[i & 63]);
        sink += checksum_eigen(qr.matrixQR());
    });

    time_ns_per_iter("decomp_general", N, "eigen_col_piv_householder_qr", iters, [&](int i) {
        Eigen::ColPivHouseholderQR<EigenMat> qr(eigen_mats[i & 63]);
        sink += checksum_eigen(qr.matrixQR());
    });
}

template <int N, typename Solver>
void validate_minimatrix_vec_solve(const std::string& kernel, const std::string& variant,
    const std::array<MiniMatrix<double, N, N>, 64>& mats,
    const std::array<MiniMatrix<double, N, 1>, 64>& rhs, Solver solve)
{
    for (int i = 0; i < 64; ++i) {
        MiniMatrix<double, N, 1> x;
        if (!solve(mats[i], rhs[i], x)) {
            std::cerr << "solve failed: kernel=" << kernel << " n=" << N << " variant=" << variant
                      << "\n";
            std::abort();
        }
        require_residual_ok(kernel, N, variant, residual_inf_vec<N>(mats[i], x, rhs[i]), 1e-8);
    }
}

template <int N, typename Solver>
void validate_minimatrix_mat_solve(const std::string& kernel, const std::string& variant,
    const std::array<MiniMatrix<double, N, N>, 64>& mats,
    const std::array<MiniMatrix<double, N, N>, 64>& rhs, Solver solve)
{
    for (int i = 0; i < 64; ++i) {
        MiniMatrix<double, N, N> x;
        if (!solve(mats[i], rhs[i], x)) {
            std::cerr << "solve failed: kernel=" << kernel << " n=" << N << " variant=" << variant
                      << "\n";
            std::abort();
        }
        require_residual_ok(kernel, N, variant, residual_inf_mat<N>(mats[i], x, rhs[i]), 1e-8);
    }
}

template <int N, typename Solver>
void validate_eigen_vec_solve(const std::string& kernel, const std::string& variant,
    const std::array<Eigen::Matrix<double, N, N>, 64>& mats,
    const std::array<Eigen::Matrix<double, N, 1>, 64>& rhs, Solver solve)
{
    for (int i = 0; i < 64; ++i) {
        Eigen::Matrix<double, N, 1> x;
        if (!solve(mats[i], rhs[i], x)) {
            std::cerr << "solve failed: kernel=" << kernel << " n=" << N << " variant=" << variant
                      << "\n";
            std::abort();
        }
        require_residual_ok(kernel, N, variant, residual_inf_eigen(mats[i], x, rhs[i]), 1e-8);
    }
}

template <int N, typename Solver>
void validate_eigen_mat_solve(const std::string& kernel, const std::string& variant,
    const std::array<Eigen::Matrix<double, N, N>, 64>& mats,
    const std::array<Eigen::Matrix<double, N, N>, 64>& rhs, Solver solve)
{
    for (int i = 0; i < 64; ++i) {
        Eigen::Matrix<double, N, N> x;
        if (!solve(mats[i], rhs[i], x)) {
            std::cerr << "solve failed: kernel=" << kernel << " n=" << N << " variant=" << variant
                      << "\n";
            std::abort();
        }
        require_residual_ok(kernel, N, variant, residual_inf_eigen(mats[i], x, rhs[i]), 1e-8);
    }
}

template <bool UO, bool UR, bool UI, int N>
void bench_solve_spd_ldlt_vec_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats,
    const std::array<MiniMatrix<double, N, 1>, 64>& rhs, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_ldlt");
    validate_minimatrix_vec_solve<N>("solve_spd_vec", variant, mats, rhs,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, 1>& b,
            MiniMatrix<double, N, 1>& x) {
            return minimatrix_ldlt_solve_vec_variant<UO, UR, UI, N>(a, b, x);
        });
    time_ns_per_iter("solve_spd_vec", N, variant, iters, [&](int i) {
        MiniMatrix<double, N, 1> x;
        minimatrix_ldlt_solve_vec_variant<UO, UR, UI, N>(mats[i & 63], rhs[i & 63], x);
        sink += checksum_minimatrix(x);
    });
}

template <bool UO, bool UR, bool UI, int N>
void bench_solve_spd_ldlt_mat_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats,
    const std::array<MiniMatrix<double, N, N>, 64>& rhs, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_ldlt");
    validate_minimatrix_mat_solve<N>("solve_spd_mat", variant, mats, rhs,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, N>& b,
            MiniMatrix<double, N, N>& x) {
            return minimatrix_ldlt_solve_mat_variant<UO, UR, UI, N>(a, b, x);
        });
    time_ns_per_iter("solve_spd_mat", N, variant, iters, [&](int i) {
        MiniMatrix<double, N, N> x;
        minimatrix_ldlt_solve_mat_variant<UO, UR, UI, N>(mats[i & 63], rhs[i & 63], x);
        sink += checksum_minimatrix(x);
    });
}

template <bool UO, bool UR, bool UI, int N>
void bench_solve_general_lu_vec_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats,
    const std::array<MiniMatrix<double, N, 1>, 64>& rhs, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_lu_partial");
    validate_minimatrix_vec_solve<N>("solve_general_vec", variant, mats, rhs,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, 1>& b,
            MiniMatrix<double, N, 1>& x) {
            return minimatrix_lu_solve_vec_variant<UO, UR, UI, N>(a, b, x);
        });
    time_ns_per_iter("solve_general_vec", N, variant, iters, [&](int i) {
        MiniMatrix<double, N, 1> x;
        minimatrix_lu_solve_vec_variant<UO, UR, UI, N>(mats[i & 63], rhs[i & 63], x);
        sink += checksum_minimatrix(x);
    });
}

template <bool UO, bool UR, bool UI, int N>
void bench_solve_general_lu_mat_variant(const std::array<MiniMatrix<double, N, N>, 64>& mats,
    const std::array<MiniMatrix<double, N, N>, 64>& rhs, int iters)
{
    const std::string variant = unroll_variant_name<UO, UR, UI>("minimatrix_lu_partial");
    validate_minimatrix_mat_solve<N>("solve_general_mat", variant, mats, rhs,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, N>& b,
            MiniMatrix<double, N, N>& x) {
            return minimatrix_lu_solve_mat_variant<UO, UR, UI, N>(a, b, x);
        });
    time_ns_per_iter("solve_general_mat", N, variant, iters, [&](int i) {
        MiniMatrix<double, N, N> x;
        minimatrix_lu_solve_mat_variant<UO, UR, UI, N>(mats[i & 63], rhs[i & 63], x);
        sink += checksum_minimatrix(x);
    });
}

template <int N> void bench_solve_spd_all(int iters)
{
    typedef Eigen::Matrix<double, N, N> EigenMat;
    typedef Eigen::Matrix<double, N, 1> EigenVec;
    std::array<MiniMatrix<double, N, N>, 64> mini_mats;
    std::array<MiniMatrix<double, N, 1>, 64> mini_rhs_vec;
    std::array<MiniMatrix<double, N, N>, 64> mini_rhs_mat;
    std::array<EigenMat, 64> eigen_mats;
    std::array<EigenVec, 64> eigen_rhs_vec;
    std::array<EigenMat, 64> eigen_rhs_mat;
    fill_minimatrix_spd<N>(mini_mats);
    fill_minimatrix_rhs_vec<N>(mini_rhs_vec);
    fill_minimatrix_rhs_mat<N>(mini_rhs_mat);
    fill_eigen_spd<N>(eigen_mats);
    fill_eigen_rhs_vec<N>(eigen_rhs_vec);
    fill_eigen_rhs_mat<N>(eigen_rhs_mat);

    validate_minimatrix_vec_solve<N>("solve_spd_vec", "minimatrix_llt_current", mini_mats,
        mini_rhs_vec,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, 1>& b,
            MiniMatrix<double, N, 1>& x) {
            minisolver::MiniLLT<double, N> llt(a);
            if (llt.info() != 0)
                return false;
            x = llt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_vec", N, "minimatrix_llt_current", iters, [&](int i) {
        minisolver::MiniLLT<double, N> llt(mini_mats[i & 63]);
        MiniMatrix<double, N, 1> x = llt.solve(mini_rhs_vec[i & 63]);
        sink += checksum_minimatrix(x);
    });

    validate_minimatrix_vec_solve<N>("solve_spd_vec", "minimatrix_ldlt_current", mini_mats,
        mini_rhs_vec,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, 1>& b,
            MiniMatrix<double, N, 1>& x) {
            minisolver::MiniLDLT<double, N> ldlt(a);
            if (ldlt.info() != 0)
                return false;
            x = ldlt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_vec", N, "minimatrix_ldlt_current", iters, [&](int i) {
        minisolver::MiniLDLT<double, N> ldlt(mini_mats[i & 63]);
        MiniMatrix<double, N, 1> x = ldlt.solve(mini_rhs_vec[i & 63]);
        sink += checksum_minimatrix(x);
    });

    validate_minimatrix_mat_solve<N>("solve_spd_mat", "minimatrix_llt_current", mini_mats,
        mini_rhs_mat,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, N>& b,
            MiniMatrix<double, N, N>& x) {
            minisolver::MiniLLT<double, N> llt(a);
            if (llt.info() != 0)
                return false;
            x = llt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_mat", N, "minimatrix_llt_current", iters, [&](int i) {
        minisolver::MiniLLT<double, N> llt(mini_mats[i & 63]);
        MiniMatrix<double, N, N> x = llt.solve(mini_rhs_mat[i & 63]);
        sink += checksum_minimatrix(x);
    });

    validate_minimatrix_mat_solve<N>("solve_spd_mat", "minimatrix_ldlt_current", mini_mats,
        mini_rhs_mat,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, N>& b,
            MiniMatrix<double, N, N>& x) {
            minisolver::MiniLDLT<double, N> ldlt(a);
            if (ldlt.info() != 0)
                return false;
            x = ldlt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_mat", N, "minimatrix_ldlt_current", iters, [&](int i) {
        minisolver::MiniLDLT<double, N> ldlt(mini_mats[i & 63]);
        MiniMatrix<double, N, N> x = ldlt.solve(mini_rhs_mat[i & 63]);
        sink += checksum_minimatrix(x);
    });

    bench_solve_spd_ldlt_vec_variant<false, false, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_spd_ldlt_vec_variant<true, false, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_spd_ldlt_vec_variant<false, true, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_spd_ldlt_vec_variant<false, false, true, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_spd_ldlt_vec_variant<true, true, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_spd_ldlt_vec_variant<true, false, true, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_spd_ldlt_vec_variant<false, true, true, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_spd_ldlt_vec_variant<true, true, true, N>(mini_mats, mini_rhs_vec, iters);

    bench_solve_spd_ldlt_mat_variant<false, false, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_spd_ldlt_mat_variant<true, false, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_spd_ldlt_mat_variant<false, true, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_spd_ldlt_mat_variant<false, false, true, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_spd_ldlt_mat_variant<true, true, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_spd_ldlt_mat_variant<true, false, true, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_spd_ldlt_mat_variant<false, true, true, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_spd_ldlt_mat_variant<true, true, true, N>(mini_mats, mini_rhs_mat, iters);

    validate_eigen_vec_solve<N>("solve_spd_vec", "eigen_llt", eigen_mats, eigen_rhs_vec,
        [](const EigenMat& a, const EigenVec& b, EigenVec& x) {
            Eigen::LLT<EigenMat> llt(a);
            if (llt.info() != Eigen::Success)
                return false;
            x = llt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_vec", N, "eigen_llt", iters, [&](int i) {
        Eigen::LLT<EigenMat> llt(eigen_mats[i & 63]);
        EigenVec x = llt.solve(eigen_rhs_vec[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_mat_solve<N>("solve_spd_mat", "eigen_llt", eigen_mats, eigen_rhs_mat,
        [](const EigenMat& a, const EigenMat& b, EigenMat& x) {
            Eigen::LLT<EigenMat> llt(a);
            if (llt.info() != Eigen::Success)
                return false;
            x = llt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_mat", N, "eigen_llt", iters, [&](int i) {
        Eigen::LLT<EigenMat> llt(eigen_mats[i & 63]);
        EigenMat x = llt.solve(eigen_rhs_mat[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_vec_solve<N>("solve_spd_vec", "eigen_ldlt", eigen_mats, eigen_rhs_vec,
        [](const EigenMat& a, const EigenVec& b, EigenVec& x) {
            Eigen::LDLT<EigenMat> ldlt(a);
            if (ldlt.info() != Eigen::Success)
                return false;
            x = ldlt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_vec", N, "eigen_ldlt", iters, [&](int i) {
        Eigen::LDLT<EigenMat> ldlt(eigen_mats[i & 63]);
        EigenVec x = ldlt.solve(eigen_rhs_vec[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_mat_solve<N>("solve_spd_mat", "eigen_ldlt", eigen_mats, eigen_rhs_mat,
        [](const EigenMat& a, const EigenMat& b, EigenMat& x) {
            Eigen::LDLT<EigenMat> ldlt(a);
            if (ldlt.info() != Eigen::Success)
                return false;
            x = ldlt.solve(b);
            return true;
        });
    time_ns_per_iter("solve_spd_mat", N, "eigen_ldlt", iters, [&](int i) {
        Eigen::LDLT<EigenMat> ldlt(eigen_mats[i & 63]);
        EigenMat x = ldlt.solve(eigen_rhs_mat[i & 63]);
        sink += checksum_eigen(x);
    });
}

template <int N> void bench_solve_general_all(int iters)
{
    typedef Eigen::Matrix<double, N, N> EigenMat;
    typedef Eigen::Matrix<double, N, 1> EigenVec;
    std::array<MiniMatrix<double, N, N>, 64> mini_mats;
    std::array<MiniMatrix<double, N, 1>, 64> mini_rhs_vec;
    std::array<MiniMatrix<double, N, N>, 64> mini_rhs_mat;
    std::array<EigenMat, 64> eigen_mats;
    std::array<EigenVec, 64> eigen_rhs_vec;
    std::array<EigenMat, 64> eigen_rhs_mat;
    fill_minimatrix_general<N>(mini_mats);
    fill_minimatrix_rhs_vec<N>(mini_rhs_vec);
    fill_minimatrix_rhs_mat<N>(mini_rhs_mat);
    fill_eigen_general<N>(eigen_mats);
    fill_eigen_rhs_vec<N>(eigen_rhs_vec);
    fill_eigen_rhs_mat<N>(eigen_rhs_mat);

    bench_solve_general_lu_vec_variant<false, false, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_general_lu_vec_variant<true, false, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_general_lu_vec_variant<false, true, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_general_lu_vec_variant<false, false, true, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_general_lu_vec_variant<true, true, false, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_general_lu_vec_variant<true, false, true, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_general_lu_vec_variant<false, true, true, N>(mini_mats, mini_rhs_vec, iters);
    bench_solve_general_lu_vec_variant<true, true, true, N>(mini_mats, mini_rhs_vec, iters);

    bench_solve_general_lu_mat_variant<false, false, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_general_lu_mat_variant<true, false, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_general_lu_mat_variant<false, true, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_general_lu_mat_variant<false, false, true, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_general_lu_mat_variant<true, true, false, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_general_lu_mat_variant<true, false, true, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_general_lu_mat_variant<false, true, true, N>(mini_mats, mini_rhs_mat, iters);
    bench_solve_general_lu_mat_variant<true, true, true, N>(mini_mats, mini_rhs_mat, iters);

    validate_minimatrix_vec_solve<N>("solve_general_vec", "minimatrix_qr_mgs", mini_mats,
        mini_rhs_vec,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, 1>& b,
            MiniMatrix<double, N, 1>& x) { return minimatrix_qr_mgs_solve_vec<N>(a, b, x); });
    time_ns_per_iter("solve_general_vec", N, "minimatrix_qr_mgs", iters, [&](int i) {
        MiniMatrix<double, N, 1> x;
        minimatrix_qr_mgs_solve_vec<N>(mini_mats[i & 63], mini_rhs_vec[i & 63], x);
        sink += checksum_minimatrix(x);
    });

    validate_minimatrix_mat_solve<N>("solve_general_mat", "minimatrix_qr_mgs", mini_mats,
        mini_rhs_mat,
        [](const MiniMatrix<double, N, N>& a, const MiniMatrix<double, N, N>& b,
            MiniMatrix<double, N, N>& x) { return minimatrix_qr_mgs_solve_mat<N>(a, b, x); });
    time_ns_per_iter("solve_general_mat", N, "minimatrix_qr_mgs", iters, [&](int i) {
        MiniMatrix<double, N, N> x;
        minimatrix_qr_mgs_solve_mat<N>(mini_mats[i & 63], mini_rhs_mat[i & 63], x);
        sink += checksum_minimatrix(x);
    });

    validate_eigen_vec_solve<N>("solve_general_vec", "eigen_partial_piv_lu", eigen_mats,
        eigen_rhs_vec, [](const EigenMat& a, const EigenVec& b, EigenVec& x) {
            Eigen::PartialPivLU<EigenMat> lu(a);
            x = lu.solve(b);
            return true;
        });
    time_ns_per_iter("solve_general_vec", N, "eigen_partial_piv_lu", iters, [&](int i) {
        Eigen::PartialPivLU<EigenMat> lu(eigen_mats[i & 63]);
        EigenVec x = lu.solve(eigen_rhs_vec[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_mat_solve<N>("solve_general_mat", "eigen_partial_piv_lu", eigen_mats,
        eigen_rhs_mat, [](const EigenMat& a, const EigenMat& b, EigenMat& x) {
            Eigen::PartialPivLU<EigenMat> lu(a);
            x = lu.solve(b);
            return true;
        });
    time_ns_per_iter("solve_general_mat", N, "eigen_partial_piv_lu", iters, [&](int i) {
        Eigen::PartialPivLU<EigenMat> lu(eigen_mats[i & 63]);
        EigenMat x = lu.solve(eigen_rhs_mat[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_vec_solve<N>("solve_general_vec", "eigen_householder_qr", eigen_mats,
        eigen_rhs_vec, [](const EigenMat& a, const EigenVec& b, EigenVec& x) {
            Eigen::HouseholderQR<EigenMat> qr(a);
            x = qr.solve(b);
            return true;
        });
    time_ns_per_iter("solve_general_vec", N, "eigen_householder_qr", iters, [&](int i) {
        Eigen::HouseholderQR<EigenMat> qr(eigen_mats[i & 63]);
        EigenVec x = qr.solve(eigen_rhs_vec[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_mat_solve<N>("solve_general_mat", "eigen_householder_qr", eigen_mats,
        eigen_rhs_mat, [](const EigenMat& a, const EigenMat& b, EigenMat& x) {
            Eigen::HouseholderQR<EigenMat> qr(a);
            x = qr.solve(b);
            return true;
        });
    time_ns_per_iter("solve_general_mat", N, "eigen_householder_qr", iters, [&](int i) {
        Eigen::HouseholderQR<EigenMat> qr(eigen_mats[i & 63]);
        EigenMat x = qr.solve(eigen_rhs_mat[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_vec_solve<N>("solve_general_vec", "eigen_col_piv_householder_qr", eigen_mats,
        eigen_rhs_vec, [](const EigenMat& a, const EigenVec& b, EigenVec& x) {
            Eigen::ColPivHouseholderQR<EigenMat> qr(a);
            x = qr.solve(b);
            return true;
        });
    time_ns_per_iter("solve_general_vec", N, "eigen_col_piv_householder_qr", iters, [&](int i) {
        Eigen::ColPivHouseholderQR<EigenMat> qr(eigen_mats[i & 63]);
        EigenVec x = qr.solve(eigen_rhs_vec[i & 63]);
        sink += checksum_eigen(x);
    });

    validate_eigen_mat_solve<N>("solve_general_mat", "eigen_col_piv_householder_qr", eigen_mats,
        eigen_rhs_mat, [](const EigenMat& a, const EigenMat& b, EigenMat& x) {
            Eigen::ColPivHouseholderQR<EigenMat> qr(a);
            x = qr.solve(b);
            return true;
        });
    time_ns_per_iter("solve_general_mat", N, "eigen_col_piv_householder_qr", iters, [&](int i) {
        Eigen::ColPivHouseholderQR<EigenMat> qr(eigen_mats[i & 63]);
        EigenMat x = qr.solve(eigen_rhs_mat[i & 63]);
        sink += checksum_eigen(x);
    });
}

int iters_for_size(int n)
{
    if (n <= 4)
        return 1000000;
    if (n <= 8)
        return 500000;
    if (n <= 12)
        return 150000;
    return 50000;
}

int decomp_iters_for_size(int n)
{
    if (n <= 4)
        return 300000;
    if (n <= 8)
        return 100000;
    if (n <= 12)
        return 30000;
    return 10000;
}

int solve_iters_for_size(int n)
{
    if (n <= 4)
        return 200000;
    if (n <= 8)
        return 60000;
    if (n <= 12)
        return 20000;
    return 8000;
}

template <int N> void bench_matmul_all_variants(int iters)
{
    bench_matmul_current<N>(iters);
    bench_matmul_variant<false, false, false, N>("loop", iters);
    bench_matmul_variant<true, false, false, N>("unroll_i", iters);
    bench_matmul_variant<false, true, false, N>("unroll_j", iters);
    bench_matmul_variant<false, false, true, N>("unroll_k", iters);
    bench_matmul_variant<true, true, false, N>("unroll_ij", iters);
    bench_matmul_variant<true, false, true, N>("unroll_ik", iters);
    bench_matmul_variant<false, true, true, N>("unroll_jk", iters);
    bench_matmul_variant<true, true, true, N>("unroll_ijk", iters);
    bench_matmul_eigen<N>(iters);
}

template <int N> void bench_add_at_mul_b_all_variants(int iters)
{
    bench_add_at_mul_b_current<N>(iters);
    bench_add_at_mul_b_variant<false, false, false, N>("loop", iters);
    bench_add_at_mul_b_variant<true, false, false, N>("unroll_i", iters);
    bench_add_at_mul_b_variant<false, true, false, N>("unroll_j", iters);
    bench_add_at_mul_b_variant<false, false, true, N>("unroll_k", iters);
    bench_add_at_mul_b_variant<true, true, false, N>("unroll_ij", iters);
    bench_add_at_mul_b_variant<true, false, true, N>("unroll_ik", iters);
    bench_add_at_mul_b_variant<false, true, true, N>("unroll_jk", iters);
    bench_add_at_mul_b_variant<true, true, true, N>("unroll_ijk", iters);
    bench_add_at_mul_b_eigen<N>(iters);
}

template <int N> void bench_size()
{
    const int iters = iters_for_size(N);
    bench_matmul_all_variants<N>(iters);
    bench_add_at_mul_b_all_variants<N>(iters);

    const int decomp_iters = decomp_iters_for_size(N);
    bench_decomp_spd_all<N>(decomp_iters);
    bench_decomp_general_all<N>(decomp_iters);

    const int solve_iters = solve_iters_for_size(N);
    bench_solve_spd_all<N>(solve_iters);
    bench_solve_general_all<N>(solve_iters);
}

void bench_sweep_1_to_16()
{
    bench_size<1>();
    bench_size<2>();
    bench_size<3>();
    bench_size<4>();
    bench_size<5>();
    bench_size<6>();
    bench_size<7>();
    bench_size<8>();
    bench_size<9>();
    bench_size<10>();
    bench_size<11>();
    bench_size<12>();
    bench_size<13>();
    bench_size<14>();
    bench_size<15>();
    bench_size<16>();
}

}

int main()
{
    std::cout << "kernel,n,variant,ns_per_iter\n";
    bench_sweep_1_to_16();
    std::cerr << "sink=" << sink << "\n";
    return 0;
}
