#pragma once
namespace minisolver {
    enum class Backend {
        CPU_SERIAL,
        GPU_MPX, // Associative Scan
        GPU_PCR  // Cyclic Reduction
    };
}