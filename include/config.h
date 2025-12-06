#pragma once
namespace roboopt {
    enum class Backend {
        CPU_SERIAL,
        GPU_MPX, // Associative Scan
        GPU_PCR  // Cyclic Reduction
    };
}