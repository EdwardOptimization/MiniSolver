#include <gtest/gtest.h>
#include "minisolver/core/matrix_defs.h"
#include <cmath>

using namespace minisolver;

// Only test Eigen backend as it's the default
#ifdef USE_EIGEN
TEST(MatrixTest, BasicOperations) {
    MSVec<double, 3> v1;
    v1 << 1.0, 2.0, 3.0;
    
    MSVec<double, 3> v2;
    v2 << 4.0, 5.0, 6.0;
    
    // Test Addition
    MSVec<double, 3> v3 = v1 + v2;
    EXPECT_DOUBLE_EQ(v3(0), 5.0);
    EXPECT_DOUBLE_EQ(v3(1), 7.0);
    EXPECT_DOUBLE_EQ(v3(2), 9.0);
    
    // Test Dot Product
    double dot = v1.dot(v2);
    EXPECT_DOUBLE_EQ(dot, 1*4 + 2*5 + 3*6); // 4+10+18 = 32
}

TEST(MatrixTest, CholeskySolve) {
    // Solve A x = b where A is SPD
    // A = [[4, 1], [1, 3]]
    // b = [1, 2]
    // Solution:
    // 4x + y = 1
    // x + 3y = 2 => x = 2 - 3y
    // 4(2-3y) + y = 1 => 8 - 12y + y = 1 => -11y = -7 => y = 7/11
    // x = 2 - 21/11 = 1/11
    
    MSMat<double, 2, 2> A;
    A << 4.0, 1.0, 1.0, 3.0;
    
    MSVec<double, 2> b;
    b << 1.0, 2.0;
    
    MSVec<double, 2> x;
    bool success = MatOps::cholesky_solve(A, b, x);
    
    EXPECT_TRUE(success);
    EXPECT_NEAR(x(0), 1.0/11.0, 1e-9);
    EXPECT_NEAR(x(1), 7.0/11.0, 1e-9);
}

TEST(MatrixTest, CholeskySolveFail) {
    // Solve A x = b where A is NOT SPD (Indefinite)
    MSMat<double, 2, 2> A;
    A << 1.0, 2.0, 2.0, 1.0; // Det = 1-4 = -3
    
    MSVec<double, 2> b;
    b << 1.0, 1.0;
    
    MSVec<double, 2> x;
    bool success = MatOps::cholesky_solve(A, b, x);
    
    EXPECT_FALSE(success);
}
#endif

