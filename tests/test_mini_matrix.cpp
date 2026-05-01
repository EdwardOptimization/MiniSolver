#include "minisolver/matrix/matrix_defs.h"
#include "minisolver/matrix/mini_matrix.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace minisolver;

template <int N, int C>
double residual_inf(const MiniMatrix<double, N, N>& A, const MiniMatrix<double, N, C>& X,
    const MiniMatrix<double, N, C>& B)
{
    double max_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < C; ++c) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                sum += A(i, j) * X(j, c);
            }
            max_abs = std::max(max_abs, std::abs(sum - B(i, c)));
        }
    }
    return max_abs;
}

template <int N, int C, typename MatA, typename MatX, typename MatB>
double residual_inf_static(const MatA& A, const MatX& X, const MatB& B)
{
    double max_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < C; ++c) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                sum += A(i, j) * X(j, c);
            }
            max_abs = std::max(max_abs, std::abs(sum - B(i, c)));
        }
    }
    return max_abs;
}

// 1. Core / Matrix: MiniMatrix_Cholesky_EdgeCases
TEST(MiniMatrixTest, Cholesky_EdgeCases)
{
    // 1. Semi-Positive Definite (Semi-Singular)
    // [ 1  1 ]
    // [ 1  1 ] -> Det = 0
    MiniMatrix<double, 2, 2> A_semi;
    A_semi.setZero();
    A_semi(0, 0) = 1.0;
    A_semi(0, 1) = 1.0;
    A_semi(1, 0) = 1.0;
    A_semi(1, 1) = 1.0;

    MiniLLT<double, 2> llt_semi(A_semi);
    EXPECT_NE(llt_semi.info(), 0) << "Cholesky should fail for semi-positive definite matrix";

    // 2. Indefinite (Negative Determinant)
    // [ 1  2 ]
    // [ 2  1 ] -> Det = 1 - 4 = -3
    MiniMatrix<double, 2, 2> A_indef;
    A_indef.setZero();
    A_indef(0, 0) = 1.0;
    A_indef(0, 1) = 2.0;
    A_indef(1, 0) = 2.0;
    A_indef(1, 1) = 1.0;

    MiniLLT<double, 2> llt_indef(A_indef);
    EXPECT_NE(llt_indef.info(), 0) << "Cholesky should fail for indefinite matrix";

    MiniMatrix<double, 2, 1> bad_b;
    bad_b(0) = 1.0;
    bad_b(1) = -2.0;
    auto bad_x = llt_indef.solve(bad_b);
    EXPECT_DOUBLE_EQ(bad_x(0), 0.0);
    EXPECT_DOUBLE_EQ(bad_x(1), 0.0);

    MiniMatrix<double, 2, 2> bad_B;
    bad_B.setIdentity();
    auto bad_X = llt_indef.solve(bad_B);
    for (double v : bad_X.data) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }

    // 3. Positive Definite (Valid)
    // [ 2  1 ]
    // [ 1  2 ] -> Det = 3
    MiniMatrix<double, 2, 2> A_pd;
    A_pd.setZero();
    A_pd(0, 0) = 2.0;
    A_pd(0, 1) = 1.0;
    A_pd(1, 0) = 1.0;
    A_pd(1, 1) = 2.0;

    MiniLLT<double, 2> llt_pd(A_pd);
    EXPECT_EQ(llt_pd.info(), 0) << "Cholesky should succeed for PD matrix";

    // Check solve accuracy
    MiniMatrix<double, 2, 1> b;
    b(0) = 3.0;
    b(1) = 3.0; // Solution should be x=[1, 1]

    auto x = llt_pd.solve(b);
    EXPECT_NEAR(x(0), 1.0, 1e-9);
    EXPECT_NEAR(x(1), 1.0, 1e-9);
}

TEST(MiniMatrixTest, Cholesky_MatrixRhsSolve)
{
    MiniMatrix<double, 2, 2> A;
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;

    MiniMatrix<double, 2, 2> B;
    B.setIdentity();

    MiniLLT<double, 2> llt(A);
    ASSERT_EQ(llt.info(), 0);
    auto X = llt.solve(B);

    EXPECT_NEAR(X(0, 0), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(X(0, 1), -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(X(1, 0), -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(X(1, 1), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(residual_inf(A, X, B), 0.0, 1e-12);
}

TEST(MiniMatrixTest, MatOps_LuMatrixRhsSolve)
{
    MSMat<double, 3, 3> A;
    A(0, 0) = 3.0;
    A(0, 1) = 1.0;
    A(0, 2) = -1.0;
    A(1, 0) = 2.0;
    A(1, 1) = 4.0;
    A(1, 2) = 1.0;
    A(2, 0) = -1.0;
    A(2, 1) = 2.0;
    A(2, 2) = 5.0;

    MSMat<double, 3, 2> B;
    B(0, 0) = 1.0;
    B(1, 0) = 2.0;
    B(2, 0) = 3.0;
    B(0, 1) = -2.0;
    B(1, 1) = 0.5;
    B(2, 1) = 4.0;

    MSMat<double, 3, 2> X;
    ASSERT_TRUE(MatOps::lu_solve_matrix(A, B, X));
    const double residual = residual_inf_static<3, 2>(A, X, B);
    EXPECT_NEAR(residual, 0.0, 1e-12);
}

TEST(MiniMatrixTest, LDLT_EdgeCasesAndSolves)
{
    MiniMatrix<double, 2, 2> A_semi;
    A_semi.setZero();
    A_semi(0, 0) = 1.0;
    A_semi(0, 1) = 1.0;
    A_semi(1, 0) = 1.0;
    A_semi(1, 1) = 1.0;
    MiniLDLT<double, 2> ldlt_semi(A_semi);
    EXPECT_NE(ldlt_semi.info(), 0);

    MiniMatrix<double, 2, 2> A_indef;
    A_indef.setZero();
    A_indef(0, 0) = 1.0;
    A_indef(0, 1) = 2.0;
    A_indef(1, 0) = 2.0;
    A_indef(1, 1) = 1.0;
    MiniLDLT<double, 2> ldlt_indef(A_indef);
    EXPECT_NE(ldlt_indef.info(), 0);

    MiniMatrix<double, 2, 1> bad_b;
    bad_b(0) = 1.0;
    bad_b(1) = -2.0;
    auto bad_x = ldlt_indef.solve(bad_b);
    EXPECT_DOUBLE_EQ(bad_x(0), 0.0);
    EXPECT_DOUBLE_EQ(bad_x(1), 0.0);

    MiniMatrix<double, 2, 2> bad_B;
    bad_B.setIdentity();
    auto bad_X = ldlt_indef.solve(bad_B);
    for (double v : bad_X.data) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }

    MiniMatrix<double, 2, 2> A;
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    MiniLDLT<double, 2> ldlt(A);
    ASSERT_EQ(ldlt.info(), 0);

    MiniMatrix<double, 2, 1> b;
    b(0) = 3.0;
    b(1) = 3.0;
    auto x = ldlt.solve(b);
    EXPECT_NEAR(x(0), 1.0, 1e-12);
    EXPECT_NEAR(x(1), 1.0, 1e-12);
    EXPECT_NEAR(residual_inf(A, x, b), 0.0, 1e-12);

    MiniMatrix<double, 2, 2> B;
    B.setIdentity();
    auto X = ldlt.solve(B);
    EXPECT_NEAR(X(0, 0), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(X(0, 1), -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(X(1, 0), -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(X(1, 1), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(residual_inf(A, X, B), 0.0, 1e-12);
}

// Optimization Correctness: add_At_mul_B
TEST(MiniMatrixTest, Optimization_AddAtMulB)
{
    // C += A^T * B
    // A: 2x3
    MiniMatrix<double, 2, 3> A;
    for (int i = 0; i < 6; ++i) {
        A.data[i] = i; // 0,1,2; 3,4,5
    }

    // B: 2x3
    MiniMatrix<double, 2, 3> B;
    for (int i = 0; i < 6; ++i) {
        B.data[i] = 1.0; // All 1s
    }

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

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(C_ref.data[i], C_opt.data[i], 1e-9);
        // Additional hardcoded check based on manual calculation
        if (i < 3) {
            EXPECT_NEAR(C_opt.data[i], 3.0, 1e-9);
        } else if (i < 6) {
            EXPECT_NEAR(C_opt.data[i], 5.0, 1e-9);
        } else {
            EXPECT_NEAR(C_opt.data[i], 7.0, 1e-9);
        }
    }
}

TEST(MiniMatrixTest, Kernel_WeightedAddAtMulB)
{
    MiniMatrix<double, 3, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = -2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A(2, 0) = -5.0;
    A(2, 1) = 6.0;

    MiniMatrix<double, 3, 2> B;
    B(0, 0) = 0.5;
    B(0, 1) = 1.5;
    B(1, 0) = -2.0;
    B(1, 1) = 2.5;
    B(2, 0) = 3.0;
    B(2, 1) = -3.5;

    MiniMatrix<double, 3, 1> weights;
    weights(0) = 2.0;
    weights(1) = 0.25;
    weights(2) = 1.5;

    MiniMatrix<double, 2, 2> ref;
    ref.setZero();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                ref(i, j) += A(k, i) * weights(k) * B(k, j);
            }
        }
    }

    MiniMatrix<double, 2, 2> opt;
    opt.setZero();
    matrix::weighted_add_At_mul_B(opt, A, weights, B);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ref.data[i], opt.data[i], 1e-12);
    }
}

TEST(MiniMatrixTest, Kernel_MatMulAndMultAdd)
{
    MiniMatrix<double, 2, 3> A;
    MiniMatrix<double, 3, 2> B;
    for (int i = 0; i < 6; ++i) {
        A.data[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < 6; ++i) {
        B.data[i] = static_cast<double>(10 + i);
    }

    auto C = A * B;
    EXPECT_NEAR(C(0, 0), 76.0, 1e-12);
    EXPECT_NEAR(C(0, 1), 82.0, 1e-12);
    EXPECT_NEAR(C(1, 0), 184.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 199.0, 1e-12);

    MiniMatrix<double, 2, 2> Accum;
    Accum.setOnes();
    Accum.mult_add(A, B);
    EXPECT_NEAR(Accum(0, 0), 77.0, 1e-12);
    EXPECT_NEAR(Accum(0, 1), 83.0, 1e-12);
    EXPECT_NEAR(Accum(1, 0), 185.0, 1e-12);
    EXPECT_NEAR(Accum(1, 1), 200.0, 1e-12);
}

TEST(MiniMatrixTest, Kernel_AddAtMulV)
{
    MiniMatrix<double, 3, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A(2, 0) = 5.0;
    A(2, 1) = 6.0;

    MiniMatrix<double, 3, 1> x;
    x(0) = 10.0;
    x(1) = 20.0;
    x(2) = 30.0;

    MiniMatrix<double, 2, 1> y;
    y.setOnes();
    y.add_At_mul_v(A, x);

    EXPECT_NEAR(y(0), 221.0, 1e-12);
    EXPECT_NEAR(y(1), 281.0, 1e-12);
}

TEST(MiniMatrixTest, DotPromotesFloatOperandsBeforeMultiply)
{
    MiniMatrix<float, 1, 1> a;
    MiniMatrix<float, 1, 1> b;
    a(0) = 1e20f;
    b(0) = 1e20f;

    const double result = a.dot(b);

    EXPECT_TRUE(MatOps::is_finite_scalar(result));
    EXPECT_GT(result, 1e39);
}

TEST(MiniMatrixTest, Kernel_SymmetrizeAndFiniteChecks)
{
    MiniMatrix<double, 3, 3> A;
    for (int i = 0; i < 9; ++i) {
        A.data[i] = static_cast<double>(i);
    }

    A.symmetrize();
    EXPECT_NEAR(A(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(A(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(A(0, 2), 4.0, 1e-12);
    EXPECT_NEAR(A(2, 0), 4.0, 1e-12);
    EXPECT_TRUE(A.allFinite());

    A(2, 2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(A.allFinite());
    EXPECT_TRUE(matrix::has_nan(A));
}

TEST(MiniMatrixTest, EigenLikeBlocksForIntegratorCompatibility)
{
    MiniMatrix<double, 4, 1> v;
    MiniMatrix<double, 2, 1> head;
    head(0) = 1.0;
    head(1) = 2.0;
    MiniMatrix<double, 2, 1> tail;
    tail(0) = 3.0;
    tail(1) = 4.0;

    v.template head<2>() = head;
    v.template tail<2>() = tail;

    auto h = v.template head<2>();
    auto t = v.template tail<2>();
    auto sum = h + t;
    EXPECT_NEAR(sum(0), 4.0, 1e-12);
    EXPECT_NEAR(sum(1), 6.0, 1e-12);

    MiniMatrix<double, 4, 4> M;
    M.setZero();
    MiniMatrix<double, 2, 2> I = MiniMatrix<double, 2, 2>::Identity();
    M.template block<2, 2>(0, 0) = I;
    M.template block<2, 2>(2, 2) = I * 2.0;

    EXPECT_NEAR(M(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(M(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(M(2, 2), 2.0, 1e-12);
    EXPECT_NEAR(M(3, 3), 2.0, 1e-12);

    MiniMatrix<double, 4, 1> col;
    col(0) = 5.0;
    col(1) = 6.0;
    col(2) = 7.0;
    col(3) = 8.0;
    M.col(1) = col;
    EXPECT_NEAR(M(0, 1), 5.0, 1e-12);
    EXPECT_NEAR(M(3, 1), 8.0, 1e-12);

    auto rows = M.template topRows<2>() + M.template bottomRows<2>();
    EXPECT_NEAR(rows(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(rows(0, 1), 12.0, 1e-12);
    EXPECT_NEAR(rows(1, 1), 14.0, 1e-12);

    MiniMatrix<double, 2, 1> delta;
    delta(0) = 0.5;
    delta(1) = 1.5;
    head -= delta;
    EXPECT_NEAR(head(0), 0.5, 1e-12);
    EXPECT_NEAR(head(1), 0.5, 1e-12);
}

#ifndef NDEBUG
TEST(MiniMatrixDeathTest, BlockOutOfBoundsAsserts)
{
    MiniMatrix<double, 2, 2> M;
    EXPECT_DEATH({ (void)M.template block<2, 2>(1, 1); }, "");
}
#endif
