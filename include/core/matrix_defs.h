#pragma once

// Configuration Macro
// CMake usually defines USE_EIGEN via add_definitions(-DUSE_EIGEN)
// Here we provide a default fallback if nothing is defined.

#if !defined(USE_CUSTOM_MATRIX) && !defined(USE_EIGEN)
#define USE_EIGEN
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

namespace minisolver {

// Type Aliases
template<typename T, int R, int C>
using MSMat = Eigen::Matrix<T, R, C>;

template<typename T, int N>
using MSVec = Eigen::Matrix<T, N, 1>;

// Operations Abstraction
struct MatOps {
    template<typename Derived>
    static void setZero(Eigen::MatrixBase<Derived>& m) {
        m.setZero();
    }

    template<typename Derived>
    static void setIdentity(Eigen::MatrixBase<Derived>& m) {
        m.setIdentity();
    }

    template<typename Derived>
    static auto transpose(const Eigen::MatrixBase<Derived>& m) {
        return m.transpose();
    }

    // Linear Solve: x = A^-1 * b using Cholesky (LLT)
    // Returns true on success, false on failure (not PD)
    template<typename Mat, typename Vec, typename ResVec>
    static bool cholesky_solve(const Mat& A, const Vec& b, ResVec& x) {
        Eigen::LLT<Mat> llt(A);
        if (llt.info() == Eigen::NumericalIssue) return false;
        x = llt.solve(b);
        return true;
    }
    
    // Solve with return (for convenience, not error checked)
    template<typename Mat, typename Vec>
    static auto cholesky_solve_ret(const Mat& A, const Vec& b) {
        Eigen::LLT<Mat> llt(A);
        return llt.solve(b);
    }
    
    // Check PD
    template<typename Mat>
    static bool is_pos_def(const Mat& A) {
        Eigen::LLT<Mat> llt(A);
        return llt.info() == Eigen::Success;
    }

    // Infinity Norm
    template<typename Derived>
    static double norm_inf(const Eigen::MatrixBase<Derived>& m) {
        return m.template lpNorm<Eigen::Infinity>();
    }
    
    // Dot Product
    template<typename V1, typename V2>
    static double dot(const V1& a, const V2& b) {
        return a.dot(b);
    }
    
    // Element-wise Max with scalar
    template<typename V>
    static V cwiseMax(const V& a, double val) {
        return a.cwiseMax(val);
    }
};

}

#else
// Placeholder for Custom Matrix Library
// #include "core/tiny_matrix.h"
// namespace minisolver {
//     template<typename T, int R, int C> using MSMat = TinyMatrix<T, R, C>;
//     ...
// }
#error "Custom Matrix Library not implemented yet. Define USE_EIGEN."
#endif

