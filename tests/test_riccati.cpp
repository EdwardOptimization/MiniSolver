#include <gtest/gtest.h>
#include "minisolver/algorithms/riccati_solver.h"
#include "minisolver/core/trajectory.h"
#include "../examples/01_car_tutorial/generated/car_model.h"

using namespace minisolver;

// Mock Trajectory with N=2
// We test if Riccati solves a trivial Linear Quadratic problem correctly
TEST(RiccatiTest, TrivialLQR) {
    using Knot = KnotPointV2<double, 4, 2, 5, 13>;
    using TrajType = Trajectory<Knot, 10>;
    
    TrajType trajectory(2);
    int N = 2;
    
    // Problem: min sum(x'x + u'u)
    // s.t. x_{k+1} = x_k + u_k
    // x0 = [1,0,0,0]
    
    auto* state = trajectory.get_active_state();
    auto* model = trajectory.get_model_data();
    auto* workspace = trajectory.get_workspace();
    
    for(int k=0; k<=N; ++k) {
        state[k].x.setZero();
        state[k].u.setZero();
        state[k].s.setZero();
        state[k].lam.setZero();
        state[k].soft_s.setZero();
        state[k].p.setZero();
        state[k].g_val.setZero();
        state[k].cost = 0.0;
        
        model[k].Q.setIdentity();
        model[k].R.setIdentity();
        model[k].q.setZero();
        model[k].r.setZero();
        model[k].H.setZero();
        
        // Simple Dynamics A=I, B=I (for first 2 dim)
        model[k].A.setIdentity();
        model[k].B.setZero();
        model[k].B(0,0) = 1.0;
        model[k].B(1,1) = 1.0;
        model[k].f_resid.setZero();
        model[k].C.setZero();
        model[k].D.setZero();
    }
    
    // Set initial state via "current state" (in Riccati, dx0 is constrained to 0 usually, 
    // but here we solve for dx relative to current. Current x=0. 
    // We want to simulate x0=1. 
    // So we set state[0].x = [0...], but actually Riccati linearizes around current traj.
    // If current traj is x=0, then dx is the state.
    // We want dx0 = 0? No, typically MPC solves for dx from x_current.
    // Here we test the KKT solver directly.
    // The KKT solver solves:
    // [H   A^T] [z]   [q]
    // [A    0 ] [lam] [b]
    // Riccati specifically solves the LQR subproblem.
    // It propagates gradients q back.
    // Let's set a gradient at k=N to see if it propagates.
    
    model[N].q(0) = 1.0; // Gradient at terminal cost
    
    SolverConfig config;
    config.reg_min = 1e-9;
    
    RiccatiSolver<TrajType, CarModel> solver;
    bool success = solver.solve(trajectory, N, 0.01, 1e-9, InertiaStrategy::REGULARIZATION, config);
    
    EXPECT_TRUE(success);
    
    // Check Feedback Gains
    // For scalar LQR with Q=1, R=1, A=1, B=1:
    // P = Q + A'PA - A'PB(R+B'PB)^-1 B'PA
    // P_N = 1
    // K_{N-1} = -(1 + 1*1*1)^-1 * 1*1*1 = -0.5
    // P_{N-1} = 1 + 1 - 1*0.5 = 1.5
    // Our B is only identity for first 2 dims.
    // So K(0,0) should be -0.5?
    // Let's check k=1 (N-1)
    
    // K is 2x4.
    // K(0,0) connects u0 to x0.
    EXPECT_NEAR(workspace[1].K(0,0), -0.5, 1e-5);
    EXPECT_NEAR(workspace[1].K(1,1), -0.5, 1e-5);
    
    // x2 and x3 are uncontrollable (B=0), so K should be 0 there?
    // P propagates A=I. So P_xx grows.
    EXPECT_NEAR(workspace[1].K(0,2), 0.0, 1e-5);
}

