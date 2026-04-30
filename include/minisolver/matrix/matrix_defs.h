#pragma once

// Configuration Macro
// CMake usually defines USE_EIGEN via add_definitions(-DUSE_EIGEN)
// Ensure Custom Matrix takes precedence if both are defined
#ifdef USE_CUSTOM_MATRIX
#ifdef USE_EIGEN
#undef USE_EIGEN
#endif
#endif

#if !defined(USE_CUSTOM_MATRIX) && !defined(USE_EIGEN)
#define USE_EIGEN
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace minisolver {

// Type Aliases
template <typename T, int R, int C> using MSMat = Eigen::Matrix<T, R, C>;

template <typename T, int N> using MSVec = Eigen::Matrix<T, N, 1>;

template <typename T, int N> using MSDiag = Eigen::DiagonalMatrix<T, N>;

template <typename MatrixType> using MSLLT = Eigen::LLT<MatrixType>;

template <typename MatrixType> using MSLDLT = Eigen::LDLT<MatrixType>;

template <typename MatrixType> using MSSPDSolver = Eigen::LLT<MatrixType>;

// Operations Abstraction
struct MatOps {
    template <typename Derived> inline static void setZero(Eigen::MatrixBase<Derived>& m)
    {
        m.setZero();
    }

    template <typename Derived> inline static void setIdentity(Eigen::MatrixBase<Derived>& m)
    {
        m.setIdentity();
    }

    template <typename Derived> inline static auto transpose(const Eigen::MatrixBase<Derived>& m)
    {
        return m.transpose();
    }

    // Linear Solve: x = A^-1 * b using Cholesky (LLT)
    // Returns true on success, false on failure (not PD)
    template <typename Mat, typename Vec, typename ResVec>
    inline static bool cholesky_solve(const Mat& A, const Vec& b, ResVec& x)
    {
        Eigen::LLT<Mat> llt(A);
        if (llt.info() == Eigen::NumericalIssue)
            return false;
        x = llt.solve(b);
        return true;
    }

    // Solve with return (for convenience, not error checked)
    template <typename Mat, typename Vec>
    inline static auto cholesky_solve_ret(const Mat& A, const Vec& b)
    {
        Eigen::LLT<Mat> llt(A);
        return llt.solve(b);
    }

    // Linear Solve: x = A^-1 * b using LU (PartialPivLU).
    // Use for non-symmetric matrices (e.g. Newton Jacobian, I - dt*Jx).
    // cholesky_solve requires SPD; lu_solve works for any invertible square matrix.
    template <typename Mat, typename Vec, typename ResVec>
    inline static bool lu_solve(const Mat& A, const Vec& b, ResVec& x)
    {
        Eigen::PartialPivLU<Mat> lu(A);
        if (lu.matrixLU().diagonal().array().abs().minCoeff() < 1e-30)
            return false; // near-singular
        x = lu.solve(b);
        return true;
    }

    // Check PD
    template <typename Mat> inline static bool is_pos_def(const Mat& A)
    {
        Eigen::LLT<Mat> llt(A);
        return llt.info() == Eigen::Success;
    }

    // Infinity Norm
    template <typename Derived> inline static double norm_inf(const Eigen::MatrixBase<Derived>& m)
    {
        return m.template lpNorm<Eigen::Infinity>();
    }

    // Dot Product
    template <typename V1, typename V2> inline static double dot(const V1& a, const V2& b)
    {
        return a.dot(b);
    }

    // Element-wise Max with scalar
    template <typename V> inline static V cwiseMax(const V& a, double val)
    {
        return a.cwiseMax(val);
    }

    // Accumulate Mult: C += A * B
    template <typename DerivedC, typename DerivedA, typename DerivedB>
    inline static void mult_add(Eigen::MatrixBase<DerivedC>& C,
        const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& B)
    {
        C.noalias() += A * B;
    }

    // Accumulate Transpose Mult: D += A^T * B
    template <typename DerivedD, typename DerivedA, typename DerivedB>
    inline static void mult_add_transA(Eigen::MatrixBase<DerivedD>& D,
        const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& B)
    {
        D.noalias() += A.transpose() * B;
    }

    // Accumulate Transpose Mult Vector: d += A^T * b
    template <typename DerivedD, typename DerivedA, typename DerivedB>
    inline static void mult_add_transA_v(Eigen::MatrixBase<DerivedD>& d,
        const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b)
    {
        d.noalias() += A.transpose() * b;
    }

    // Symmetrize: m = 0.5 * (m + m^T)
    template <typename Derived> inline static void symmetrize(Eigen::MatrixBase<Derived>& m)
    {
        m = 0.5 * (m + m.transpose());
    }

    // In-place Cholesky Solve wrapper
    template <typename LLTType, typename Vec>
    inline static void solve_llt_inplace(const LLTType& llt, Vec& b)
    {
        b = llt.solve(b);
    }

    template <typename SolverType, typename Rhs>
    inline static void solve_spd_inplace(const SolverType& solver, Rhs& b)
    {
        b = solver.solve(b);
    }

    // Check LLT success
    template <typename LLTType> inline static bool is_llt_success(const LLTType& llt)
    {
        return llt.info() == Eigen::Success;
    }

    template <typename SolverType> inline static bool is_spd_solver_success(const SolverType& solver)
    {
        return solver.info() == Eigen::Success;
    }

    // Bit-level IEEE 754 inspection — works even with -ffast-math.
    // std::isnan() / Eigen::allFinite() are constant-folded to false/true under
    // -ffast-math (-ffinite-math-only). These functions read the raw bit pattern
    // instead, similar to HPIPM's approach but without relying on x!=x (which
    // -ffinite-math-only can also optimize away).
    //
    // Performance: ~0.09 ns/call (single mov instruction when inlined).

    // Reinterpret double bits as uint64. Uses __builtin_bit_cast (GCC 10+,
    // Clang 13+) when available; falls back to std::memcpy which compiles
    // to the same single register move.
    inline static uint64_t double_to_bits(double val)
    {
#if defined(__GNUC__) && __GNUC__ >= 10
        return __builtin_bit_cast(uint64_t, val);
#else
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        return bits;
#endif
    }

    inline static bool is_nan_scalar(double val)
    {
        uint64_t bits = double_to_bits(val);
        // NaN: exponent = 0x7FF, mantissa != 0
        return ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL)
            && ((bits & 0x000FFFFFFFFFFFFFULL) != 0);
    }

    inline static bool is_inf_scalar(double val)
    {
        uint64_t bits = double_to_bits(val);
        return (bits & 0x7FFFFFFFFFFFFFFFULL) == 0x7FF0000000000000ULL;
    }

    inline static bool is_finite_scalar(double val)
    {
        uint64_t bits = double_to_bits(val);
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    }

    template <typename Derived> inline static bool has_nan(const Eigen::MatrixBase<Derived>& m)
    {
        // Iterate element-wise. For fixed-size matrices this is unrolled.
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                if (is_nan_scalar(m(i, j)))
                    return true;
            }
        }
        return false;
    }
};

}

#else
// Custom Matrix Library
#include "minisolver/matrix/mini_matrix.h"

namespace minisolver {
// Type Aliases
template <typename T, int R, int C> using MSMat = MiniMatrix<T, R, C>;

template <typename T, int N> using MSVec = MiniMatrix<T, N, 1>;

template <typename T, int N> using MSDiag = MiniDiagonal<T, N>;

template <typename MatrixType>
using MSLLT = MiniLLT<typename MatrixType::Scalar, MatrixType::RowsAtCompileTime>;

template <typename MatrixType>
using MSLDLT = MiniLDLT<typename MatrixType::Scalar, MatrixType::RowsAtCompileTime>;

template <typename MatrixType>
using MSSPDSolver = MiniLDLT<typename MatrixType::Scalar, MatrixType::RowsAtCompileTime>;

// Operations Abstraction
struct MatOps {
    template <typename Derived> inline static void setZero(Derived& m) { m.setZero(); }

    template <typename Derived> inline static void setIdentity(Derived& m) { m.setIdentity(); }

    // Return by value for MiniMatrix
    template <typename Derived> inline static auto transpose(const Derived& m)
    {
        return m.transpose();
    }

    // Linear Solve: x = A^-1 * b using Cholesky (LLT)
    template <typename Mat, typename Vec, typename ResVec>
    inline static bool cholesky_solve(const Mat& A, const Vec& b, ResVec& x)
    {
        MiniLLT<double, Mat::Rows> llt(A);
        if (llt.info() != 0)
            return false;

        if ((void*)&b == (void*)&x) {
            llt.solve_in_place(x);
        } else {
            x = llt.solve(b);
        }
        return true;
    }

    template <typename Mat, typename Vec>
    inline static auto cholesky_solve_ret(const Mat& A, const Vec& b)
    {
        MiniLLT<double, Mat::Rows> llt(A);
        return llt.solve(b);
    }

    // LU solve with partial pivoting — works for any invertible square matrix.
    template <typename Mat, typename Vec, typename ResVec>
    inline static bool lu_solve(const Mat& A, const Vec& b, ResVec& x)
    {
        constexpr int N = Mat::Rows;
        // Copy A into LU workspace
        double lu[N * N];
        for (int i = 0; i < N * N; ++i)
            lu[i] = A.data[i];

        // Permutation
        int perm[N];
        for (int i = 0; i < N; ++i)
            perm[i] = i;

        // Forward elimination with partial pivoting
        for (int k = 0; k < N; ++k) {
            // Find pivot
            int max_row = k;
            double max_val = std::abs(lu[k * N + k]);
            for (int i = k + 1; i < N; ++i) {
                double v = std::abs(lu[i * N + k]);
                if (v > max_val) {
                    max_val = v;
                    max_row = i;
                }
            }
            if (max_val < 1e-30)
                return false; // singular

            // Swap rows
            if (max_row != k) {
                std::swap(perm[k], perm[max_row]);
                for (int j = 0; j < N; ++j)
                    std::swap(lu[k * N + j], lu[max_row * N + j]);
            }

            // Eliminate
            for (int i = k + 1; i < N; ++i) {
                double factor = lu[i * N + k] / lu[k * N + k];
                for (int j = k + 1; j < N; ++j)
                    lu[i * N + j] -= factor * lu[k * N + j];
                lu[i * N + k] = factor;
            }
        }

        // Forward substitution (L)
        double y[N];
        for (int i = 0; i < N; ++i) {
            y[i] = b(perm[i]);
            for (int j = 0; j < i; ++j)
                y[i] -= lu[i * N + j] * y[j];
        }

        // Backward substitution (U)
        for (int i = N - 1; i >= 0; --i) {
            x(i) = y[i];
            for (int j = i + 1; j < N; ++j)
                x(i) -= lu[i * N + j] * x(j);
            x(i) /= lu[i * N + i];
        }
        return true;
    }

    template <typename Mat> inline static bool is_pos_def(const Mat& A)
    {
        MiniLLT<double, Mat::Rows> llt(A);
        return llt.info() == 0;
    }

    template <typename Derived> inline static double norm_inf(const Derived& m)
    {
        return m.lpNormInfinity();
    }

    template <typename V1, typename V2> inline static double dot(const V1& a, const V2& b)
    {
        return a.dot(b);
    }

    template <typename V> inline static V cwiseMax(const V& a, double val)
    {
        return a.cwiseMax(val);
    }

    // Accumulate Mult: C += A * B
    template <typename DerivedC, typename DerivedA, typename DerivedB>
    inline static void mult_add(DerivedC& C, const DerivedA& A, const DerivedB& B)
    {
        C.mult_add(A, B);
    }

    // Accumulate Transpose Mult: D += A^T * B
    template <typename DerivedD, typename DerivedA, typename DerivedB>
    inline static void mult_add_transA(DerivedD& D, const DerivedA& A, const DerivedB& B)
    {
        D.add_At_mul_B(A, B);
    }

    // Accumulate Transpose Mult Vector: d += A^T * b
    template <typename DerivedD, typename DerivedA, typename DerivedB>
    inline static void mult_add_transA_v(DerivedD& d, const DerivedA& A, const DerivedB& b)
    {
        d.add_At_mul_v(A, b);
    }

    // Symmetrize: m = 0.5 * (m + m^T)
    template <typename Derived> inline static void symmetrize(Derived& m) { m.symmetrize(); }

    // In-place Cholesky Solve wrapper
    template <typename LLTType, typename Vec>
    inline static void solve_llt_inplace(LLTType& llt, Vec& b)
    {
        llt.solve_in_place(b);
    }

    template <typename SolverType, typename Rhs>
    inline static void solve_spd_inplace(SolverType& solver, Rhs& b)
    {
        solver.solve_in_place(b);
    }

    // Check LLT success
    template <typename LLTType> inline static bool is_llt_success(const LLTType& llt)
    {
        return llt.info() == 0;
    }

    template <typename SolverType> inline static bool is_spd_solver_success(const SolverType& solver)
    {
        return solver.info() == 0;
    }

    // Bit-level IEEE 754 inspection — works even with -ffast-math.
    inline static uint64_t double_to_bits(double val)
    {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        return bits;
    }

    inline static bool is_nan_scalar(double val)
    {
        uint64_t bits = double_to_bits(val);
        return ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL)
            && ((bits & 0x000FFFFFFFFFFFFFULL) != 0);
    }

    inline static bool is_inf_scalar(double val)
    {
        uint64_t bits = double_to_bits(val);
        return (bits & 0x7FFFFFFFFFFFFFFFULL) == 0x7FF0000000000000ULL;
    }

    inline static bool is_finite_scalar(double val)
    {
        uint64_t bits = double_to_bits(val);
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    }

    template <typename Derived> inline static bool has_nan(const Derived& m)
    {
        for (int i = 0; i < Derived::Rows; ++i) {
            for (int j = 0; j < Derived::Cols; ++j) {
                if (is_nan_scalar(m(i, j)))
                    return true;
            }
        }
        return false;
    }
};
}
#endif
