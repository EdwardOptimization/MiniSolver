#pragma once
#include <array>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <initializer_list>
#include <iostream>

namespace minisolver {

// Simple fixed-size matrix class for embedded use (no malloc)
template<typename T, int R, int C>
class MiniMatrix {
public:
    static constexpr int Rows = R;
    static constexpr int Cols = C;
    
    std::array<T, R*C> data;

    MiniMatrix() { data.fill(T(0)); }

    // Accessors
    T& operator()(int r, int c) { return data[r * C + c]; }
    const T& operator()(int r, int c) const { return data[r * C + c]; }
    
    T& operator()(int i) { return data[i]; }
    const T& operator()(int i) const { return data[i]; }

    // Fill
    void setZero() { data.fill(T(0)); }
    void setOnes() { data.fill(T(1)); } // [FIX] Added setOnes
    void fill(T val) { data.fill(val); } // [FIX] Added fill
    
    T minCoeff() const { // [FIX] Added minCoeff
        if (R*C == 0) return T(0);
        T min_val = data[0];
        for(int i=1; i<R*C; ++i) if(data[i] < min_val) min_val = data[i];
        return min_val;
    }

    static MiniMatrix Identity() { // [FIX] Added Identity
        MiniMatrix res;
        res.setIdentity();
        return res;
    }
    
    // Metadata for Templates
    static constexpr int RowsAtCompileTime = R; // [FIX] Added RowsAtCompileTime

    void setIdentity() {
        setZero();
        for(int i=0; i<std::min(R,C); ++i) (*this)(i,i) = T(1);
    }
    
    // Assignment
    MiniMatrix& operator=(const MiniMatrix& other) = default;

    // Scalar Ops
    MiniMatrix operator-() const {
        MiniMatrix res;
        for(int i=0; i<R*C; ++i) res.data[i] = -data[i];
        return res;
    }

    // Scalar Multiplication (Matrix * Scalar) [FIX] Added
    MiniMatrix operator*(T scalar) const {
        MiniMatrix res;
        for(int i=0; i<R*C; ++i) res.data[i] = data[i] * scalar;
        return res;
    }

    // Scalar Multiplication (Scalar * Matrix) [FIX] Added as friend
    friend MiniMatrix operator*(T scalar, const MiniMatrix& m) {
        MiniMatrix res;
        for(int i=0; i<R*C; ++i) res.data[i] = scalar * m.data[i];
        return res;
    }

    // Matrix Ops
    template<int C2>
    MiniMatrix<T, R, C2> operator*(const MiniMatrix<T, C, C2>& other) const {
        MiniMatrix<T, R, C2> res;
        for(int i=0; i<R; ++i) {
            for(int j=0; j<C2; ++j) {
                T sum = 0;
                for(int k=0; k<C; ++k) {
                    sum += (*this)(i,k) * other(k,j);
                }
                res(i,j) = sum;
            }
        }
        return res;
    }
    
    MiniMatrix operator+(const MiniMatrix& other) const {
        MiniMatrix res;
        for(int i=0; i<R*C; ++i) res.data[i] = data[i] + other.data[i];
        return res;
    }
    
    MiniMatrix operator-(const MiniMatrix& other) const {
        MiniMatrix res;
        for(int i=0; i<R*C; ++i) res.data[i] = data[i] - other.data[i];
        return res;
    }

    MiniMatrix& operator+=(const MiniMatrix& other) {
        for(int i=0; i<R*C; ++i) data[i] += other.data[i];
        return *this;
    }
    
    // Transpose
    MiniMatrix<T, C, R> transpose() const {
        MiniMatrix<T, C, R> res;
        for(int i=0; i<R; ++i) {
            for(int j=0; j<C; ++j) {
                res(j,i) = (*this)(i,j);
            }
        }
        return res;
    }
    
    // Eigen-like API
    MiniMatrix& noalias() { return *this; }
    
    // In-place add scaled: this += other * scale
    void add_scaled(const MiniMatrix& other, T scale) {
        for(int i=0; i<R*C; ++i) data[i] += other.data[i] * scale;
    }

    double dot(const MiniMatrix& other) const {
        double sum = 0;
        for(int i=0; i<R*C; ++i) sum += data[i] * other.data[i];
        return sum;
    }
    
    MiniMatrix cwiseMax(T val) const {
        MiniMatrix res;
        for(int i=0; i<R*C; ++i) res.data[i] = std::max(data[i], val);
        return res;
    }
    
    double lpNormInfinity() const {
        double max_val = 0;
        for(int i=0; i<R*C; ++i) max_val = std::max(max_val, std::abs(data[i]));
        return max_val;
    }
    
    bool allFinite() const {
        for(int i=0; i<R*C; ++i) if(!std::isfinite(data[i])) return false;
        return true;
    }

    // [OPTIMIZATION] In-place symmetrization (for square matrices)
    void symmetrize() {
        static_assert(R == C, "Must be square");
        for(int i=0; i<R-1; ++i) {
            for(int j=i+1; j<C; ++j) { // Iterate upper triangle
                T val = (data[i*C+j] + data[j*C+i]) * 0.5;
                data[i*C+j] = val;
                data[j*C+i] = val;
            }
        }
    }

    // [OPTIMIZATION] Accumulate transposed matrix multiplication: this += A^T * B
    // Avoids creating temporary A^T and product matrix
    // this: (R x C), A: (R_A x R), B: (R_A x C) -> A^T * B: (R x C)
    template<int R_A>
    void add_At_mul_B(const MiniMatrix<T, R_A, R>& A, const MiniMatrix<T, R_A, C>& B) {
        for(int i=0; i<R; ++i) { // this rows (A cols)
            for(int j=0; j<C; ++j) { // this cols (B cols)
                T sum = 0;
                for(int k=0; k<R_A; ++k) { // common dim (A rows, B rows)
                    // A^T(i, k) = A(k, i)
                    sum += A(k, i) * B(k, j);
                }
                (*this)(i,j) += sum;
            }
        }
    }

    // [OPTIMIZATION] Accumulate transposed matrix-vector multiplication: this_vec += A^T * x
    // this: (R x 1), A: (R_A x R), x: (R_A x 1)
    template<int R_A>
    void add_At_mul_v(const MiniMatrix<T, R_A, R>& A, const MiniMatrix<T, R_A, 1>& x) {
        static_assert(C == 1, "Destination must be a vector");
        for(int i=0; i<R; ++i) { // this rows (A cols)
            T sum = 0;
            for(int k=0; k<R_A; ++k) { // A rows
                sum += A(k, i) * x(k);
            }
            data[i] += sum;
        }
    }
};

// Diagonal Wrapper
template<typename T, int N>
class MiniDiagonal {
public:
    MiniMatrix<T, N, 1> diag;
    MiniDiagonal(const MiniMatrix<T, N, 1>& d) : diag(d) {}
    
    // Diag * Matrix
    template<int C>
    MiniMatrix<T, N, C> operator*(const MiniMatrix<T, N, C>& m) const {
        MiniMatrix<T, N, C> res;
        for(int i=0; i<N; ++i) {
            T s = diag(i);
            for(int j=0; j<C; ++j) {
                res(i,j) = s * m(i,j);
            }
        }
        return res;
    }
};

// Cholesky Decomposition (LLT)
template<typename T, int N>
class MiniLLT {
    MiniMatrix<T, N, N> L;
    bool success;
public:
    MiniLLT(const MiniMatrix<T, N, N>& A) {
        compute(A);
    }
    
    void compute(const MiniMatrix<T, N, N>& A) {
        L.setZero();
        success = true;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                T sum = 0;
                for (int k = 0; k < j; k++)
                    sum += L(i, k) * L(j, k);

                if (i == j) {
                    T val = A(i, i) - sum;
                    if (val <= 0) {
                        success = false;
                        return;
                    }
                    L(i, j) = std::sqrt(val);
                } else {
                    L(i, j) = (1.0 / L(j, j) * (A(i, j) - sum));
                }
            }
        }
    }
    
    int info() const { return success ? 0 : 1; }
    
    MiniMatrix<T, N, 1> solve(const MiniMatrix<T, N, 1>& b) {
        // Forward sub Ly = b
        MiniMatrix<T, N, 1> y;
        for(int i=0; i<N; ++i) {
            T sum = 0;
            for(int k=0; k<i; ++k) sum += L(i,k) * y(k);
            y(i) = (b(i) - sum) / L(i,i);
        }
        
        // Backward sub L^T x = y
        MiniMatrix<T, N, 1> x;
        for(int i=N-1; i>=0; --i) {
            T sum = 0;
            for(int k=i+1; k<N; ++k) sum += L(k,i) * x(k); // L(k,i) is L^T(i,k)
            x(i) = (y(i) - sum) / L(i,i);
        }
        return x;
    }

    // In-place solve
    void solve_in_place(MiniMatrix<T, N, 1>& b) {
        // Forward sub Ly = b (overwrite b with y)
        for(int i=0; i<N; ++i) {
            T sum = 0;
            for(int k=0; k<i; ++k) sum += L(i,k) * b(k);
            b(i) = (b(i) - sum) / L(i,i);
        }
        
        // Backward sub L^T x = y (overwrite b with x)
        for(int i=N-1; i>=0; --i) {
            T sum = 0;
            for(int k=i+1; k<N; ++k) sum += L(k,i) * b(k);
            b(i) = (b(i) - sum) / L(i,i);
        }
    }
    
    // Overload for Matrix RHS
    template<int C>
    MiniMatrix<T, N, C> solve(const MiniMatrix<T, N, C>& B) {
        MiniMatrix<T, N, C> X;
        for(int c=0; c<C; ++c) {
             // Extract col
             MiniMatrix<T, N, 1> b_col;
             for(int i=0; i<N; ++i) b_col(i) = B(i,c);
             auto x_col = solve(b_col);
             for(int i=0; i<N; ++i) X(i,c) = x_col(i);
        }
        return X;
    }

    // In-place Matrix RHS
    template<int C>
    void solve_in_place(MiniMatrix<T, N, C>& B) {
        for(int c=0; c<C; ++c) {
             // Forward
             for(int i=0; i<N; ++i) {
                 T sum = 0;
                 for(int k=0; k<i; ++k) sum += L(i,k) * B(k,c);
                 B(i,c) = (B(i,c) - sum) / L(i,i);
             }
             // Backward
             for(int i=N-1; i>=0; --i) {
                 T sum = 0;
                 for(int k=i+1; k<N; ++k) sum += L(k,i) * B(k,c);
                 B(i,c) = (B(i,c) - sum) / L(i,i);
             }
        }
    }
};

}
