#include "minisolver/solver/backend_interface.h"
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace minisolver {

// [FIX] Define combine_ops
template<int N>
__host__ __device__
GpuLinearOp<N> combine_ops(const GpuLinearOp<N>& b, const GpuLinearOp<N>& a) {
    // Placeholder implementation for parallel scan associative operator
    // In real implementation, this computes the composite state transition.
    // Order matters: op(b, a) usually means applying 'b' then 'a' (or similar, depending on convention).
    GpuLinearOp<N> res;
    res.dummy_val = a.dummy_val + b.dummy_val; 
    return res;
}

// MPX Functor
template<int N>
struct MPXOp {
    __host__ __device__
    GpuLinearOp<N> operator()(const GpuLinearOp<N>& a, const GpuLinearOp<N>& b) const {
        return combine_ops(b, a); // inclusive: op(acc, curr)
    }
};

// PCR Kernel
template<int N>
__global__ void pcr_kernel(GpuLinearOp<N>* in, GpuLinearOp<N>* out, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (idx >= stride) {
            out[idx] = combine_ops(in[idx], in[idx - stride]);
        } else {
            out[idx] = in[idx];
        }
    }
}

template<int NX>
void gpu_dispatch_solve(std::vector<GpuLinearOp<NX>>& h_ops, Backend mode) {
    int N = h_ops.size();
    thrust::device_vector<GpuLinearOp<NX>> d_ops = h_ops;

    if (mode == Backend::GPU_MPX) {
        thrust::inclusive_scan(d_ops.begin(), d_ops.end(), d_ops.begin(), MPXOp<NX>());
    }
    else if (mode == Backend::GPU_PCR) {
        thrust::device_vector<GpuLinearOp<NX>> d_temp = d_ops;
        GpuLinearOp<NX>* in_ptr = thrust::raw_pointer_cast(d_ops.data());
        GpuLinearOp<NX>* out_ptr = thrust::raw_pointer_cast(d_temp.data());

        int threads = 128;
        int blocks = (N + threads - 1) / threads;

        for(int stride=1; stride < N; stride *= 2) {
            pcr_kernel<NX><<<blocks, threads>>>(in_ptr, out_ptr, N, stride);
            cudaDeviceSynchronize();
            std::swap(in_ptr, out_ptr);
        }
        if (in_ptr != thrust::raw_pointer_cast(d_ops.data())) d_ops = d_temp;
    }

    thrust::copy(d_ops.begin(), d_ops.end(), h_ops.begin());
}

// Explicit Instantiation for Common Model Sizes
// Using macro to cover a range of dimensions (1 to 12) to support various models
#define INSTANTIATE_GPU_SOLVE(N) \
    template void gpu_dispatch_solve<N>(std::vector<GpuLinearOp<N>>&, Backend);

INSTANTIATE_GPU_SOLVE(1)
INSTANTIATE_GPU_SOLVE(2)
INSTANTIATE_GPU_SOLVE(3)
INSTANTIATE_GPU_SOLVE(4)  // CarModel
INSTANTIATE_GPU_SOLVE(5)
INSTANTIATE_GPU_SOLVE(6)  // BicycleExtModel
INSTANTIATE_GPU_SOLVE(7)
INSTANTIATE_GPU_SOLVE(8)
INSTANTIATE_GPU_SOLVE(9)
INSTANTIATE_GPU_SOLVE(10)
INSTANTIATE_GPU_SOLVE(11)
INSTANTIATE_GPU_SOLVE(12)

#undef INSTANTIATE_GPU_SOLVE

}