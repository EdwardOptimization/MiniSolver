#pragma once

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace minisolver {

// Forward Declaration
template<typename T, int N> struct MiniDiagonal;

// --- MiniMatrix Implementation ---
template<typename T, int R, int C>
struct alignas(16) MiniMatrix {
    T data[R * C];

    MiniMatrix() { setZero(); }
    
    T& operator()(int r, int c) { return data[r * C + c]; }
    const T& operator()(int r, int c) const { return data[r * C + c]; }
    
    T& operator()(int i) { return data[i]; }
    const T& operator()(int i) const { return data[i]; }

    static constexpr int Rows = R;
    static constexpr int Cols = C;
    static constexpr int RowsAtCompileTime = R;
    static constexpr int ColsAtCompileTime = C;
    static constexpr int Size = R * C;

    void fill(T val) {
        for(int i=0; i<Size; ++i) data[i] = val;
    }

    void setZero() { fill(T(0)); }
    void setOnes() { fill(T(1)); }
    
    void setIdentity() {
        setZero();
        constexpr int min_dim = (R < C) ? R : C;
        for(int i=0; i<min_dim; ++i) (*this)(i,i) = T(1);
    }
    
    bool allFinite() const {
        for(int i=0; i<Size; ++i) {
            if(!std::isfinite(data[i])) return false;
        }
        return true;
    }

    // --- Member Operators ---
    
    MiniMatrix<T, R, C> operator+(const MiniMatrix<T, R, C>& other) const {
        MiniMatrix<T, R, C> res;
        for(int i=0; i<Size; ++i) res.data[i] = data[i] + other.data[i];
        return res;
    }
    
    MiniMatrix<T, R, C>& operator+=(const MiniMatrix<T, R, C>& other) {
        for(int i=0; i<Size; ++i) data[i] += other.data[i];
        return *this;
    }
    
    MiniMatrix<T, R, C> operator-(const MiniMatrix<T, R, C>& other) const {
        MiniMatrix<T, R, C> res;
        for(int i=0; i<Size; ++i) res.data[i] = data[i] - other.data[i];
        return res;
    }
    
    MiniMatrix<T, R, C> operator-() const {
        MiniMatrix<T, R, C> res;
        for(int i=0; i<Size; ++i) res.data[i] = -data[i];
        return res;
    }

    // Scalar Mul
    MiniMatrix<T, R, C> operator*(T s) const {
        MiniMatrix<T, R, C> res;
        for(int i=0; i<Size; ++i) res.data[i] = data[i] * s;
        return res;
    }

    // Matrix Mul
    template<int K>
    MiniMatrix<T, R, K> operator*(const MiniMatrix<T, C, K>& other) const {
        MiniMatrix<T, R, K> res;
        for(int i=0; i<R; ++i) {
            for(int j=0; j<K; ++j) {
                T sum = 0;
                // Manual unrolling for small inner dimension
                if constexpr (C <= 4) {
                    #pragma GCC unroll 4
                    for(int k=0; k<C; ++k) {
                        sum += (*this)(i,k) * other(k,j);
                    }
                } else {
                    for(int k=0; k<C; ++k) {
                        sum += (*this)(i,k) * other(k,j);
                    }
                }
                res(i,j) = sum;
            }
        }
        return res;
    }
    
    MiniMatrix<T, C, R> transpose() const {
        MiniMatrix<T, C, R> res;
        for(int i=0; i<R; ++i) {
            for(int j=0; j<C; ++j) {
                res(j,i) = (*this)(i,j);
            }
        }
        return res;
    }
    
    T dot(const MiniMatrix<T, R, C>& other) const {
        T sum = 0;
        for(int i=0; i<Size; ++i) sum += data[i] * other.data[i];
        return sum;
    }
    
    T lpNormInfinity() const {
        T max_val = 0;
        for(int i=0; i<Size; ++i) {
            T abs_val = std::abs(data[i]);
            if(abs_val > max_val) max_val = abs_val;
        }
        return max_val;
    }

    MiniMatrix<T, R, C> cwiseMax(T val) const {
        MiniMatrix<T, R, C> res;
        for(int i=0; i<Size; ++i) res.data[i] = std::max(data[i], val);
        return res;
    }

    MiniMatrix<T, R, C>& noalias() { return *this; }
};

// --- Scalar * Matrix (Global) ---
template<typename T, int R, int C>
MiniMatrix<T, R, C> operator*(T s, const MiniMatrix<T, R, C>& m) {
    return m * s;
}

// --- Diagonal Matrix Wrapper ---
template<typename T, int N>
struct MiniDiagonal {
    MiniMatrix<T, N, 1> diag;
    
    MiniDiagonal(const MiniMatrix<T, N, 1>& v) : diag(v) {}
    
    T operator()(int i) const { return diag(i); }
};

// --- Diagonal Operations (Global) ---

// Operator: Diagonal * Matrix (Scaling rows)
template<typename TD, int ND, int CM>
MiniMatrix<TD, ND, CM> operator*(const MiniDiagonal<TD, ND>& D, const MiniMatrix<TD, ND, CM>& M) {
    MiniMatrix<TD, ND, CM> res;
    for(int i=0; i<ND; ++i) {
        TD s = D(i);
        for(int j=0; j<CM; ++j) {
            res(i,j) = s * M(i,j);
        }
    }
    return res;
}

// Operator: Matrix * Diagonal (Scaling cols)
template<typename TM, int RM, int ND>
MiniMatrix<TM, RM, ND> operator*(const MiniMatrix<TM, RM, ND>& M, const MiniDiagonal<TM, ND>& D) {
    MiniMatrix<TM, RM, ND> res;
    for(int j=0; j<ND; ++j) {
        TM s = D(j);
        for(int i=0; i<RM; ++i) {
            res(i,j) = M(i,j) * s;
        }
    }
    return res;
}

// --- Cholesky Decomposition ---
template<typename T, int N>
class MiniLLT {
public:
    MiniMatrix<T, N, N> L;
    bool success;

    MiniLLT(const MiniMatrix<T, N, N>& A) {
        compute(A);
    }
    
    void compute(const MiniMatrix<T, N, N>& A) {
        success = true;
        L.setZero();
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
        MiniMatrix<T, N, 1> y;
        for(int i=0; i<N; ++i) {
            T sum = 0;
            for(int j=0; j<i; ++j) sum += L(i,j) * y(j);
            y(i) = (b(i) - sum) / L(i,i);
        }
        
        MiniMatrix<T, N, 1> x;
        for(int i=N-1; i>=0; --i) {
            T sum = 0;
            for(int j=i+1; j<N; ++j) sum += L(j,i) * x(j); 
            x(i) = (y(i) - sum) / L(i,i);
        }
        return x;
    }
    
    template<int K>
    MiniMatrix<T, N, K> solve(const MiniMatrix<T, N, K>& B) {
        MiniMatrix<T, N, K> X;
        for(int k=0; k<K; ++k) {
            MiniMatrix<T, N, 1> b_col;
            for(int i=0; i<N; ++i) b_col(i) = B(i,k);
            
            auto x_col = solve(b_col);
            
            for(int i=0; i<N; ++i) X(i,k) = x_col(i);
        }
        return X;
    }
};

} // namespace minisolver
