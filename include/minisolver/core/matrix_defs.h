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
// Custom Matrix Library
#include "minisolver/core/mini_matrix.h"

namespace minisolver {
    // Type Aliases
    template<typename T, int R, int C> 
    using MSMat = MiniMatrix<T, R, C>;
    
    template<typename T, int N>
    using MSVec = MiniMatrix<T, N, 1>;
    
    template<typename T, int N>
    using MSDiag = MiniDiagonal<T, N>;

    // Operations Abstraction
    struct MatOps {
        template<typename Derived>
        static void setZero(Derived& m) {
            m.setZero();
        }

        template<typename Derived>
        static void setIdentity(Derived& m) {
            m.setIdentity();
        }
        
        // Return by value for MiniMatrix
        template<typename Derived>
        static auto transpose(const Derived& m) {
            return m.transpose();
        }

        // Linear Solve: x = A^-1 * b using Cholesky (LLT)
        template<typename Mat, typename Vec, typename ResVec>
        static bool cholesky_solve(const Mat& A, const Vec& b, ResVec& x) {
            MiniLLT<double, Mat::Rows> llt(A);
            if (llt.info() != 0) return false;
            x = llt.solve(b);
            return true;
        }
        
        template<typename Mat, typename Vec>
        static auto cholesky_solve_ret(const Mat& A, const Vec& b) {
             MiniLLT<double, Mat::Rows> llt(A);
             return llt.solve(b);
        }
        
        template<typename Mat>
        static bool is_pos_def(const Mat& A) {
            MiniLLT<double, Mat::Rows> llt(A);
            return llt.info() == 0;
        }

        template<typename Derived>
        static double norm_inf(const Derived& m) {
            return m.lpNormInfinity();
        }
        
        template<typename V1, typename V2>
        static double dot(const V1& a, const V2& b) {
            return a.dot(b);
        }
        
        template<typename V>
        static V cwiseMax(const V& a, double val) {
            return a.cwiseMax(val);
        }
    };
}
#endif

