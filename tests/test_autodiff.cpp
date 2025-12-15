#include <gtest/gtest.h>
#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/core/types.h"
#include <cmath>

using namespace minisolver;

// This test verifies that the generated C++ code for CarModel 
// produces values consistent with expected kinematics.
// This indirectly validates MiniModel.py's code generation.

TEST(AutoDiffTest, CarModelDynamics) {
    // x = [x, y, theta, v]
    // u = [acc, steer]
    // x_dot = [v cos(theta), v sin(theta), v/L tan(steer), acc]
    
    MSVec<double, 4> x;
    x << 0.0, 0.0, 0.0, 10.0;
    
    MSVec<double, 2> u;
    u << 2.0, 0.1; // steer approx 0.1 rad
    
    MSVec<double, 13> p;
    p.setZero();
    p(6) = 2.5; // L
    
    // Test Continuous Dynamics
    MSVec<double, 4> xdot = CarModel::dynamics_continuous(x, u, p);
    
    EXPECT_DOUBLE_EQ(xdot(0), 10.0 * 1.0); // v cos(0)
    EXPECT_DOUBLE_EQ(xdot(1), 0.0);        // v sin(0)
    EXPECT_DOUBLE_EQ(xdot(3), 2.0);        // acc
    
    double expected_theta_dot = (10.0 / 2.5) * std::tan(0.1);
    EXPECT_NEAR(xdot(2), expected_theta_dot, 1e-9);
}

TEST(AutoDiffTest, CarModelIntegratorEuler) {
    MSVec<double, 4> x;
    x << 0.0, 0.0, 0.0, 10.0;
    
    MSVec<double, 2> u;
    u << 0.0, 0.0; // Straight line
    
    MSVec<double, 13> p;
    p.setZero();
    p(6) = 2.5;
    
    double dt = 0.1;
    MSVec<double, 4> x_next = CarModel::integrate(x, u, p, dt, IntegratorType::EULER_EXPLICIT);
    
    // x += v * dt = 10 * 0.1 = 1.0
    EXPECT_DOUBLE_EQ(x_next(0), 1.0);
    EXPECT_DOUBLE_EQ(x_next(1), 0.0);
}

TEST(AutoDiffTest, CostDerivatives) {
    // Check if gradients are non-zero where expected
    KnotPointV2<double, 4, 2, 5, 13> kp;
    kp.set_zero();
    
    // Split into state and model
    StateNode<double, 4, 2, 5, 13> state;
    ModelData<double, 4, 2, 5> model;
    
    // Copy data from kp to state
    state.x = kp.x;
    state.u = kp.u;
    state.p = kp.p;
    
    state.x(0) = 5.0; // x
    state.p(1) = 10.0; // x_ref
    state.p(8) = 1.0;  // w_pos
    
    // Cost = w_pos * (x - x_ref)^2 = 1.0 * (5 - 10)^2 = 25.0
    // Grad x = 2 * w_pos * (x - x_ref) = 2 * 1 * (-5) = -10
    
    CarModel::compute_cost_exact(state, model); // Computes Cost, q, r, Q, R, H
    
    EXPECT_DOUBLE_EQ(state.cost, 25.0);
    EXPECT_DOUBLE_EQ(model.q(0), -10.0);
    EXPECT_DOUBLE_EQ(model.q(1), 0.0); // y error is 0 (0-0)
}

