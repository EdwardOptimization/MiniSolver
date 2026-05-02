#pragma once

namespace minisolver {

struct LinearSolveResult {
    bool ok = false;
    bool degraded_step = false;
    int degraded_riccati_freeze_count = 0;

    constexpr LinearSolveResult() = default;

    constexpr LinearSolveResult(
        bool ok_value, bool degraded_value = false, int degraded_freeze_count = 0)
        : ok(ok_value)
        , degraded_step(degraded_value)
        , degraded_riccati_freeze_count(degraded_freeze_count)
    {
    }

    constexpr operator bool() const { return ok; }
};

} // namespace minisolver
