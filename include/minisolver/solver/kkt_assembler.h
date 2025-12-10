#pragma once
#include "minisolver/core/types.h"

namespace minisolver {
// Simplification: In pure unconstrained DDP, "condensing" is calculating the
// transition matrices for the Value function Hessian (P) and Gradient (p).
// For MPX/PCR, we map the Riccati step into an affine operator.

template<typename Knot>
void prepare_backward_operator(Knot& kp) {
    // Standard LQR Feedback calc
    // Q_uu = R + B' P B  (Approximated here using Model Q, R for structural demo)
    // K = -inv(Q_uu) * Q_ux

    // For the MPX/PCR demo, we treat the backward pass as a linear recurrence.
    // P_k = Q + A' P_{k+1} A  => This is roughly P_k = op_A * P_{k+1} + op_b
    // We populate op_A and op_b roughly based on dynamics to simulate the computational load.

    kp.op_A = kp.A.transpose(); // Propagates backward
    kp.op_b = kp.q;             // Cost gradient adds up
}
}