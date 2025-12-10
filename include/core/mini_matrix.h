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
};

}
