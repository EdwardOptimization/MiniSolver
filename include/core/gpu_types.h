#pragma once
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

namespace minisolver {

// Represents the affine operator for the backward pass:
// P_k = A_op * P_{k+1} + b_op
// In DDP, this corresponds to the relationship between Value function Hessians/Gradients.
template<int NX>
struct GpuLinearOp {
    float A[NX * NX];
    float b[NX];

    HOST_DEVICE static GpuLinearOp identity() {
        GpuLinearOp op;
        for(int i=0; i<NX*NX; ++i) op.A[i] = 0.0f;
        for(int i=0; i<NX; ++i) {
            op.A[i*NX + i] = 1.0f;
            op.b[i] = 0.0f;
        }
        return op;
    }
};

// The Associative Operator: Combine (Next, Prev)
// New_Op = Next_Op * Prev_Op
template<int N>
HOST_DEVICE inline GpuLinearOp<N> combine_ops(const GpuLinearOp<N>& next, const GpuLinearOp<N>& prev) {
    GpuLinearOp<N> res;
    // A_new = Next.A * Prev.A
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            float sum = 0.0f;
            for(int k=0; k<N; ++k) sum += next.A[i*N + k] * prev.A[k*N + j];
            res.A[i*N + j] = sum;
        }
    }
    // b_new = Next.A * Prev.b + Next.b
    for(int i=0; i<N; ++i) {
        float sum = 0.0f;
        for(int k=0; k<N; ++k) sum += next.A[i*N + k] * prev.b[k];
        res.b[i] = sum + next.b[i];
    }
    return res;
}
}