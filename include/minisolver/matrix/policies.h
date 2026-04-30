#pragma once

namespace minisolver {
namespace matrix {

#ifndef MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK
#define MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK 256
#endif

#ifndef MINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N
#define MINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N 4
#endif

#ifndef MINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N
#define MINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N 4
#endif

#ifndef MINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N
#define MINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N 8
#endif

    // Conservative default policy, not a benchmark-winner table.
    //
    // Tiny factorizations get fully static control flow; medium sizes only
    // unroll short reductions; larger sizes stay as ordinary loops to limit
    // register pressure, code size, and compile time. Platform-specific tuning
    // should be validated with matrix_kernel_bench using target compiler flags.
    struct MatrixPolicy {
        template <int Work> struct StaticUnroll {
            enum { value = (Work <= MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK) };
        };

        template <int N> struct LDLTFactor {
            enum {
                outer = (N <= MINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N),
                row = (N <= MINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N),
                inner = (N <= MINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N)
            };
        };
    };

}
}
