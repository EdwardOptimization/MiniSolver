#include <gtest/gtest.h>
#include "minisolver/core/mini_matrix.h"

using namespace minisolver;

// 1. Core / Matrix: MiniMatrix_Cholesky_EdgeCases
TEST(MiniMatrixTest, Cholesky_EdgeCases) {
    // 1. Semi-Positive Definite (Semi-Singular)
    // [ 1  1 ]
    // [ 1  1 ] -> Det = 0
    MiniMatrix<double, 2, 2> A_semi;
    A_semi.setZero();
    A_semi(0,0) = 1.0; A_semi(0,1) = 1.0;
    A_semi(1,0) = 1.0; A_semi(1,1) = 1.0;
    
    MiniLLT<double, 2> llt_semi(A_semi);
    EXPECT_NE(llt_semi.info(), 0) << "Cholesky should fail for semi-positive definite matrix";

    // 2. Indefinite (Negative Determinant)
    // [ 1  2 ]
    // [ 2  1 ] -> Det = 1 - 4 = -3
    MiniMatrix<double, 2, 2> A_indef;
    A_indef.setZero();
    A_indef(0,0) = 1.0; A_indef(0,1) = 2.0;
    A_indef(1,0) = 2.0; A_indef(1,1) = 1.0;
    
    MiniLLT<double, 2> llt_indef(A_indef);
    EXPECT_NE(llt_indef.info(), 0) << "Cholesky should fail for indefinite matrix";
    
    // 3. Positive Definite (Valid)
    // [ 2  1 ]
    // [ 1  2 ] -> Det = 3
    MiniMatrix<double, 2, 2> A_pd;
    A_pd.setZero();
    A_pd(0,0) = 2.0; A_pd(0,1) = 1.0;
    A_pd(1,0) = 1.0; A_pd(1,1) = 2.0;
    
    MiniLLT<double, 2> llt_pd(A_pd);
    EXPECT_EQ(llt_pd.info(), 0) << "Cholesky should succeed for PD matrix";
    
    // Check solve accuracy
    MiniMatrix<double, 2, 1> b;
    b(0) = 3.0; b(1) = 3.0; // Solution should be x=[1, 1]
    
    auto x = llt_pd.solve(b);
    EXPECT_NEAR(x(0), 1.0, 1e-9);
    EXPECT_NEAR(x(1), 1.0, 1e-9);
}

// Optimization Correctness: add_At_mul_B
TEST(MiniMatrixTest, Optimization_AddAtMulB) {
    // C += A^T * B
    // A: 2x3
    MiniMatrix<double, 2, 3> A;
    for(int i=0; i<6; ++i) A.data[i] = i; // 0,1,2; 3,4,5
    
    // B: 2x3
    MiniMatrix<double, 2, 3> B;
    for(int i=0; i<6; ++i) B.data[i] = 1.0; // All 1s
    
    // Ref C: 3x3
    MiniMatrix<double, 3, 3> C_ref; 
    C_ref.setZero();
    
    // Compute Ref: A^T * B
    // A = [[0,1,2], [3,4,5]]
    // A^T = [[0,3], [1,4], [2,5]]
    // B = [[1,1,1], [1,1,1]]
    // A^T * B = 
    // [[0+3, 0+3, 0+3], 
    //  [1+4, 1+4, 1+4], 
    //  [2+5, 2+5, 2+5]] 
    // = [[3,3,3], [5,5,5], [7,7,7]]
    
    auto At = A.transpose();
    C_ref = At * B;
    
    // Opt C
    MiniMatrix<double, 3, 3> C_opt; 
    C_opt.setZero();
    C_opt.add_At_mul_B(A, B);
    
    for(int i=0; i<9; ++i) {
        EXPECT_NEAR(C_ref.data[i], C_opt.data[i], 1e-9);
        // Additional hardcoded check based on manual calculation
        if (i < 3) EXPECT_NEAR(C_opt.data[i], 3.0, 1e-9);
        else if (i < 6) EXPECT_NEAR(C_opt.data[i], 5.0, 1e-9);
        else EXPECT_NEAR(C_opt.data[i], 7.0, 1e-9);
    }
}

