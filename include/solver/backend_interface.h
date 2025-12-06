#pragma once
#include <vector>
#include "config.h"
#include "core/gpu_types.h"

namespace minisolver {
    // The Bridge function. Implemented in .cu file.
    template<int NX>
    void gpu_dispatch_solve(std::vector<GpuLinearOp<NX>>& ops, Backend mode);
}