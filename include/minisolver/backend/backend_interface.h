#pragma once
#include "minisolver/core/gpu_types.h"
#include "minisolver/core/solver_options.h" // Backend enum
#include <vector>

namespace minisolver {
// The Bridge function. Implemented in .cu file.
template <int NX> void gpu_dispatch_solve(std::vector<GpuLinearOp<NX>>& ops, Backend mode);
}