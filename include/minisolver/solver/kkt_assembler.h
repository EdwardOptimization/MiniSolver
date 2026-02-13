#pragma once
#include "minisolver/core/types.h"

namespace minisolver {

// =============================================================================
// KKT Assembler — GPU Backend Placeholder
//
// This file is reserved for future GPU (MPX/PCR) backend support.
// The GPU backend would map the Riccati backward pass into affine operators
// suitable for parallel associative scan algorithms.
//
// Currently unused. The CPU serial backend performs the Riccati recursion
// directly in riccati.h / riccati_solver.h.
// =============================================================================

// TODO: Implement GPU operator packing when GPU backend is completed.
// The packing logic should convert per-stage KKT data (A, B, Q, R, etc.)
// into GpuLinearOp<NX> operators for parallel Riccati solve.

}
