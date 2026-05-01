#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>

#include "minisolver/matrix/kernels.h"
#include "minisolver/matrix/policies.h"

namespace minisolver {

template <typename T, int R, int C> class MiniMatrix;

template <typename T, int ParentR, int ParentC, int R, int C> class MiniBlockConstRef {
public:
    using Scalar = T;
    static constexpr int Rows = R;
    static constexpr int Cols = C;
    static constexpr int RowsAtCompileTime = R;

    MiniBlockConstRef(const MiniMatrix<T, ParentR, ParentC>& parent, int row0, int col0)
        : parent_(parent)
        , row0_(row0)
        , col0_(col0)
    {
        assert(row0_ >= 0 && col0_ >= 0);
        assert(row0_ + R <= ParentR && col0_ + C <= ParentC);
    }

    const T& operator()(int r, int c) const { return parent_(row0_ + r, col0_ + c); }
    const T& operator()(int i) const
    {
        static_assert(C == 1, "Linear accessor is only valid for vectors");
        return parent_(row0_ + i, col0_);
    }

    operator MiniMatrix<T, R, C>() const
    {
        MiniMatrix<T, R, C> out;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                out(r, c) = (*this)(r, c);
            }
        }
        return out;
    }

    MiniMatrix<T, R, C> operator*(T scalar) const
    {
        return static_cast<MiniMatrix<T, R, C>>(*this) * scalar;
    }

    friend MiniMatrix<T, R, C> operator*(T scalar, const MiniBlockConstRef& block)
    {
        return static_cast<MiniMatrix<T, R, C>>(block) * scalar;
    }

    template <typename Other> MiniMatrix<T, R, C> operator+(const Other& other) const
    {
        return static_cast<MiniMatrix<T, R, C>>(*this) + static_cast<MiniMatrix<T, R, C>>(other);
    }

    template <typename Other> MiniMatrix<T, R, C> operator-(const Other& other) const
    {
        return static_cast<MiniMatrix<T, R, C>>(*this) - static_cast<MiniMatrix<T, R, C>>(other);
    }

private:
    const MiniMatrix<T, ParentR, ParentC>& parent_;
    int row0_;
    int col0_;
};

template <typename T, int ParentR, int ParentC, int R, int C> class MiniBlockRef {
public:
    using Scalar = T;
    static constexpr int Rows = R;
    static constexpr int Cols = C;
    static constexpr int RowsAtCompileTime = R;

    MiniBlockRef(MiniMatrix<T, ParentR, ParentC>& parent, int row0, int col0)
        : parent_(parent)
        , row0_(row0)
        , col0_(col0)
    {
        assert(row0_ >= 0 && col0_ >= 0);
        assert(row0_ + R <= ParentR && col0_ + C <= ParentC);
    }

    T& operator()(int r, int c) { return parent_(row0_ + r, col0_ + c); }
    const T& operator()(int r, int c) const { return parent_(row0_ + r, col0_ + c); }

    T& operator()(int i)
    {
        static_assert(C == 1, "Linear accessor is only valid for vectors");
        return parent_(row0_ + i, col0_);
    }
    const T& operator()(int i) const
    {
        static_assert(C == 1, "Linear accessor is only valid for vectors");
        return parent_(row0_ + i, col0_);
    }

    MiniBlockRef& operator=(const MiniBlockRef& other)
    {
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                (*this)(r, c) = other(r, c);
            }
        }
        return *this;
    }

    template <typename Other> MiniBlockRef& operator=(const Other& other)
    {
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                (*this)(r, c) = other(r, c);
            }
        }
        return *this;
    }

    operator MiniMatrix<T, R, C>() const
    {
        MiniMatrix<T, R, C> out;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                out(r, c) = (*this)(r, c);
            }
        }
        return out;
    }

    MiniMatrix<T, R, C> operator*(T scalar) const
    {
        return static_cast<MiniMatrix<T, R, C>>(*this) * scalar;
    }

    friend MiniMatrix<T, R, C> operator*(T scalar, const MiniBlockRef& block)
    {
        return static_cast<MiniMatrix<T, R, C>>(block) * scalar;
    }

    template <typename Other> MiniMatrix<T, R, C> operator+(const Other& other) const
    {
        return static_cast<MiniMatrix<T, R, C>>(*this) + static_cast<MiniMatrix<T, R, C>>(other);
    }

    template <typename Other> MiniMatrix<T, R, C> operator-(const Other& other) const
    {
        return static_cast<MiniMatrix<T, R, C>>(*this) - static_cast<MiniMatrix<T, R, C>>(other);
    }

private:
    MiniMatrix<T, ParentR, ParentC>& parent_;
    int row0_;
    int col0_;
};

// Simple fixed-size matrix class for embedded use (no malloc)
template <typename T, int R, int C> class MiniMatrix {
public:
    using Scalar = T;
    static constexpr int Rows = R;
    static constexpr int Cols = C;

    std::array<T, R * C> data;

    MiniMatrix() { matrix::fill(*this, T(0)); }

    // Accessors
    T& operator()(int r, int c) { return data[r * C + c]; }
    const T& operator()(int r, int c) const { return data[r * C + c]; }

    T& operator()(int i) { return data[i]; }
    const T& operator()(int i) const { return data[i]; }

    // Fill
    void setZero() { matrix::fill(*this, T(0)); }
    void setOnes() { matrix::fill(*this, T(1)); }
    void fill(T val) { matrix::fill(*this, val); }

    T minCoeff() const
    {
        if (R * C == 0) {
            return T(0);
        }
        T min_val = data[0];
        for (int i = 1; i < R * C; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
            }
        }
        return min_val;
    }

    static MiniMatrix Identity()
    {
        MiniMatrix res;
        res.setIdentity();
        return res;
    }

    // Metadata for Templates
    static constexpr int RowsAtCompileTime = R;

    void setIdentity()
    {
        setZero();
        for (int i = 0; i < std::min(R, C); ++i) {
            (*this)(i, i) = T(1);
        }
    }

    // Assignment
    MiniMatrix& operator=(const MiniMatrix& other) = default;

    // Scalar Ops
    MiniMatrix operator-() const
    {
        MiniMatrix res;
        matrix::scale(res, *this, T(-1));
        return res;
    }

    MiniMatrix operator*(T scalar) const
    {
        MiniMatrix res;
        matrix::scale(res, *this, scalar);
        return res;
    }

    friend MiniMatrix operator*(T scalar, const MiniMatrix& m)
    {
        MiniMatrix res;
        matrix::scale(res, m, scalar);
        return res;
    }

    // Matrix Ops
    template <int C2> MiniMatrix<T, R, C2> operator*(const MiniMatrix<T, C, C2>& other) const
    {
        MiniMatrix<T, R, C2> res;
        matrix::matmul(res, *this, other);
        return res;
    }

    MiniMatrix operator+(const MiniMatrix& other) const
    {
        MiniMatrix res;
        matrix::add(res, *this, other);
        return res;
    }

    MiniMatrix operator-(const MiniMatrix& other) const
    {
        MiniMatrix res;
        matrix::sub(res, *this, other);
        return res;
    }

    MiniMatrix& operator+=(const MiniMatrix& other)
    {
        matrix::add_scaled(*this, other, T(1));
        return *this;
    }

    MiniMatrix& operator-=(const MiniMatrix& other)
    {
        matrix::add_scaled(*this, other, T(-1));
        return *this;
    }

    MiniMatrix operator/(T scalar) const
    {
        MiniMatrix res;
        matrix::scale(res, *this, T(1) / scalar);
        return res;
    }

    // Transpose
    MiniMatrix<T, C, R> transpose() const
    {
        MiniMatrix<T, C, R> res;
        matrix::transpose(res, *this);
        return res;
    }

    // Eigen-like API
    MiniMatrix& noalias() { return *this; }

    template <int N> MiniBlockRef<T, R, C, N, 1> head()
    {
        static_assert(C == 1, "head() is only valid for vectors");
        static_assert(N <= R, "head size exceeds vector size");
        return MiniBlockRef<T, R, C, N, 1>(*this, 0, 0);
    }

    template <int N> MiniBlockConstRef<T, R, C, N, 1> head() const
    {
        static_assert(C == 1, "head() is only valid for vectors");
        static_assert(N <= R, "head size exceeds vector size");
        return MiniBlockConstRef<T, R, C, N, 1>(*this, 0, 0);
    }

    template <int N> MiniBlockRef<T, R, C, N, 1> tail()
    {
        static_assert(C == 1, "tail() is only valid for vectors");
        static_assert(N <= R, "tail size exceeds vector size");
        return MiniBlockRef<T, R, C, N, 1>(*this, R - N, 0);
    }

    template <int N> MiniBlockConstRef<T, R, C, N, 1> tail() const
    {
        static_assert(C == 1, "tail() is only valid for vectors");
        static_assert(N <= R, "tail size exceeds vector size");
        return MiniBlockConstRef<T, R, C, N, 1>(*this, R - N, 0);
    }

    template <int BR, int BC> MiniBlockRef<T, R, C, BR, BC> block(int row0, int col0)
    {
        return MiniBlockRef<T, R, C, BR, BC>(*this, row0, col0);
    }

    template <int BR, int BC> MiniBlockConstRef<T, R, C, BR, BC> block(int row0, int col0) const
    {
        return MiniBlockConstRef<T, R, C, BR, BC>(*this, row0, col0);
    }

    MiniBlockRef<T, R, C, R, 1> col(int col) { return MiniBlockRef<T, R, C, R, 1>(*this, 0, col); }

    MiniBlockConstRef<T, R, C, R, 1> col(int col) const
    {
        return MiniBlockConstRef<T, R, C, R, 1>(*this, 0, col);
    }

    template <int N> MiniBlockRef<T, R, C, N, C> topRows()
    {
        static_assert(N <= R, "topRows size exceeds row count");
        return MiniBlockRef<T, R, C, N, C>(*this, 0, 0);
    }

    template <int N> MiniBlockConstRef<T, R, C, N, C> topRows() const
    {
        static_assert(N <= R, "topRows size exceeds row count");
        return MiniBlockConstRef<T, R, C, N, C>(*this, 0, 0);
    }

    template <int N> MiniBlockRef<T, R, C, N, C> bottomRows()
    {
        static_assert(N <= R, "bottomRows size exceeds row count");
        return MiniBlockRef<T, R, C, N, C>(*this, R - N, 0);
    }

    template <int N> MiniBlockConstRef<T, R, C, N, C> bottomRows() const
    {
        static_assert(N <= R, "bottomRows size exceeds row count");
        return MiniBlockConstRef<T, R, C, N, C>(*this, R - N, 0);
    }

    // In-place add scaled: this += other * scale
    void add_scaled(const MiniMatrix& other, T scale) { matrix::add_scaled(*this, other, scale); }

    double dot(const MiniMatrix& other) const { return matrix::dot(*this, other); }

    MiniMatrix cwiseMax(T val) const
    {
        MiniMatrix res;
        for (int i = 0; i < R * C; ++i) {
            res.data[i] = std::max(data[i], val);
        }
        return res;
    }

    double lpNormInfinity() const { return matrix::norm_inf(*this); }

    bool allFinite() const { return matrix::all_finite(*this); }

    // [OPTIMIZATION] In-place symmetrization (for square matrices)
    void symmetrize()
    {
        static_assert(R == C, "Must be square");
        matrix::symmetrize(*this);
    }

    // [OPTIMIZATION] In-place multiply-add: this += A * B
    template <int K> void mult_add(const MiniMatrix<T, R, K>& A, const MiniMatrix<T, K, C>& B)
    {
        matrix::matmul_add(*this, A, B);
    }

    // [OPTIMIZATION] Accumulate transposed matrix multiplication: this += A^T * B
    // Avoids creating temporary A^T and product matrix
    // this: (R x C), A: (R_A x R), B: (R_A x C) -> A^T * B: (R x C)
    template <int R_A>
    void add_At_mul_B(const MiniMatrix<T, R_A, R>& A, const MiniMatrix<T, R_A, C>& B)
    {
        matrix::add_At_mul_B(*this, A, B);
    }

    // [OPTIMIZATION] Accumulate transposed matrix-vector multiplication: this_vec += A^T * x
    // this: (R x 1), A: (R_A x R), x: (R_A x 1)
    template <int R_A>
    void add_At_mul_v(const MiniMatrix<T, R_A, R>& A, const MiniMatrix<T, R_A, 1>& x)
    {
        static_assert(C == 1, "Destination must be a vector");
        matrix::add_At_mul_v(*this, A, x);
    }
};

// Diagonal Wrapper
template <typename T, int N> class MiniDiagonal {
public:
    MiniMatrix<T, N, 1> diag;
    MiniDiagonal() = default;
    MiniDiagonal(const MiniMatrix<T, N, 1>& d)
        : diag(d)
    {
    }

    // Diag * Matrix
    template <int C> MiniMatrix<T, N, C> operator*(const MiniMatrix<T, N, C>& m) const
    {
        MiniMatrix<T, N, C> res;
        for (int i = 0; i < N; ++i) {
            T s = diag(i);
            for (int j = 0; j < C; ++j) {
                res(i, j) = s * m(i, j);
            }
        }
        return res;
    }
};

// Cholesky Decomposition (LLT)
template <typename T, int N> class MiniLLT {
    MiniMatrix<T, N, N> L;
    std::array<T, N> inv_diag;
    bool success;

public:
    MiniLLT()
        : success(false)
    {
    }
    MiniLLT(const MiniMatrix<T, N, N>& A) { compute(A); }

    void compute(const MiniMatrix<T, N, N>& A)
    {
        L.setZero();
        success = true;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                T sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += L(i, k) * L(j, k);
                }

                if (i == j) {
                    T val = A(i, i) - sum;
                    if (val <= 0) {
                        success = false;
                        return;
                    }
                    L(i, j) = std::sqrt(val);
                    inv_diag[i] = T(1) / L(i, j);
                } else {
                    L(i, j) = (A(i, j) - sum) * inv_diag[j];
                }
            }
        }
    }

    int info() const { return success ? 0 : 1; }

    MiniMatrix<T, N, 1> solve(const MiniMatrix<T, N, 1>& b)
    {
        MiniMatrix<T, N, 1> x = b;
        solve_in_place(x);
        return x;
    }

    // In-place solve
    void solve_in_place(MiniMatrix<T, N, 1>& b)
    {
        if (!success) {
            b.setZero();
            return;
        }

        // Forward sub Ly = b (overwrite b with y)
        for (int i = 0; i < N; ++i) {
            T sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += L(i, k) * b(k);
            }
            b(i) = (b(i) - sum) * inv_diag[i];
        }

        // Backward sub L^T x = y (overwrite b with x)
        for (int i = N - 1; i >= 0; --i) {
            T sum = 0;
            for (int k = i + 1; k < N; ++k) {
                sum += L(k, i) * b(k);
            }
            b(i) = (b(i) - sum) * inv_diag[i];
        }
    }

    // Overload for Matrix RHS
    template <int C> MiniMatrix<T, N, C> solve(const MiniMatrix<T, N, C>& B)
    {
        MiniMatrix<T, N, C> X = B;
        solve_in_place(X);
        return X;
    }

    // In-place Matrix RHS
    template <int C> void solve_in_place(MiniMatrix<T, N, C>& B)
    {
        if (!success) {
            B.setZero();
            return;
        }

        // Forward substitution over all RHS columns: L * Y = B.
        for (int i = 0; i < N; ++i) {
            for (int c = 0; c < C; ++c) {
                T sum = B(i, c);
                for (int k = 0; k < i; ++k) {
                    sum -= L(i, k) * B(k, c);
                }
                B(i, c) = sum * inv_diag[i];
            }
        }

        // Backward substitution over all RHS columns: L^T * X = Y.
        for (int i = N - 1; i >= 0; --i) {
            for (int c = 0; c < C; ++c) {
                T sum = B(i, c);
                for (int k = i + 1; k < N; ++k) {
                    sum -= L(k, i) * B(k, c);
                }
                B(i, c) = sum * inv_diag[i];
            }
        }
    }
};

// Square-root-free Cholesky decomposition for SPD matrices: A = L * D * L^T.
template <typename T, int N> class MiniLDLT {
    MiniMatrix<T, N, N> L;
    std::array<T, N> D;
    std::array<T, N> inv_D;
    bool success;

public:
    MiniLDLT()
        : success(false)
    {
    }
    MiniLDLT(const MiniMatrix<T, N, N>& A) { compute(A); }

    void compute(const MiniMatrix<T, N, N>& A)
    {
        compute_impl<matrix::MatrixPolicy::LDLTFactor<N>::outer,
            matrix::MatrixPolicy::LDLTFactor<N>::row, matrix::MatrixPolicy::LDLTFactor<N>::inner>(
            A);
    }

private:
    template <bool UnrollOuter, bool UnrollRow, bool UnrollInner>
    void compute_impl(const MiniMatrix<T, N, N>& A)
    {
        L.setZero();
        D.fill(T(0));
        inv_D.fill(T(0));
        success = true;

        struct OuterBody {
            MiniLDLT& self;
            const MiniMatrix<T, N, N>& A;
            inline void operator()(int j)
            {
                self.template compute_column<UnrollRow, UnrollInner>(A, j);
            }
        } body = { *this, A };
        matrix::ForRange<UnrollOuter, N>::run(body);
    }

    template <bool UnrollRow, bool UnrollInner>
    void compute_column(const MiniMatrix<T, N, N>& A, int j)
    {
        if (!success) {
            return;
        }

        T diag_sum = T(0);
        struct DiagBody {
            MiniLDLT& self;
            int j;
            T& diag_sum;
            inline void operator()(int k) { diag_sum += self.L(j, k) * self.L(j, k) * self.D[k]; }
        } diag_body = { *this, j, diag_sum };
        matrix::PrefixRange<UnrollInner, N>::run(j, diag_body);

        D[j] = A(j, j) - diag_sum;
        if (D[j] <= T(0)) {
            success = false;
            return;
        }
        inv_D[j] = T(1) / D[j];
        L(j, j) = T(1);

        struct RowBody {
            MiniLDLT& self;
            const MiniMatrix<T, N, N>& A;
            int j;
            inline void operator()(int i) { self.template compute_row<UnrollInner>(A, j, i); }
        } row_body = { *this, A, j };
        matrix::SuffixRange<UnrollRow, N>::run(j + 1, row_body);
    }

    template <bool UnrollInner> void compute_row(const MiniMatrix<T, N, N>& A, int j, int i)
    {
        T sum = T(0);
        struct InnerBody {
            MiniLDLT& self;
            int i;
            int j;
            T& sum;
            inline void operator()(int k) { sum += self.L(i, k) * self.L(j, k) * self.D[k]; }
        } inner_body = { *this, i, j, sum };
        matrix::PrefixRange<UnrollInner, N>::run(j, inner_body);
        L(i, j) = (A(i, j) - sum) * inv_D[j];
    }

public:
    int info() const { return success ? 0 : 1; }

    MiniMatrix<T, N, 1> solve(const MiniMatrix<T, N, 1>& b)
    {
        MiniMatrix<T, N, 1> x = b;
        solve_in_place(x);
        return x;
    }

    void solve_in_place(MiniMatrix<T, N, 1>& b)
    {
        if (!success) {
            b.setZero();
            return;
        }

        // L * y = b.
        for (int i = 0; i < N; ++i) {
            T sum = b(i);
            for (int k = 0; k < i; ++k) {
                sum -= L(i, k) * b(k);
            }
            b(i) = sum;
        }

        // D * z = y.
        for (int i = 0; i < N; ++i) {
            b(i) *= inv_D[i];
        }

        // L^T * x = z.
        for (int i = N - 1; i >= 0; --i) {
            T sum = b(i);
            for (int k = i + 1; k < N; ++k) {
                sum -= L(k, i) * b(k);
            }
            b(i) = sum;
        }
    }

    template <int C> MiniMatrix<T, N, C> solve(const MiniMatrix<T, N, C>& B)
    {
        MiniMatrix<T, N, C> X = B;
        solve_in_place(X);
        return X;
    }

    template <int C> void solve_in_place(MiniMatrix<T, N, C>& B)
    {
        if (!success) {
            B.setZero();
            return;
        }

        // L * Y = B. Process all RHS columns together to keep the Riccati K solve fast.
        for (int i = 0; i < N; ++i) {
            for (int c = 0; c < C; ++c) {
                T sum = B(i, c);
                for (int k = 0; k < i; ++k) {
                    sum -= L(i, k) * B(k, c);
                }
                B(i, c) = sum;
            }
        }

        // D * Z = Y.
        for (int i = 0; i < N; ++i) {
            for (int c = 0; c < C; ++c) {
                B(i, c) *= inv_D[i];
            }
        }

        // L^T * X = Z.
        for (int i = N - 1; i >= 0; --i) {
            for (int c = 0; c < C; ++c) {
                T sum = B(i, c);
                for (int k = i + 1; k < N; ++k) {
                    sum -= L(k, i) * B(k, c);
                }
                B(i, c) = sum;
            }
        }
    }
};

}
