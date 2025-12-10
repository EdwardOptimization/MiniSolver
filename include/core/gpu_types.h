#pragma once

// Placeholder for GPU backend types and configuration.
// Future implementation will support CUDA/OpenCL backends for parallel Riccati solve (Scan).

namespace minisolver {

enum class GpuBackendType {
    CUDA,
    OPENCL,
    NONE
};

struct GpuConfig {
    int block_size = 256;
    int grid_size = 1;
    bool use_shared_memory = true;
};

// Data structure for parallel scan operations
template<int NX>
struct GpuLinearOp {
    // Matrices for the associative operator (A_k, B_k in scan terms)
    // Placeholder logic
    double dummy_val;
};

}
