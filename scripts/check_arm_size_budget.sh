#!/usr/bin/env bash
# Verify that the embedded ARM Cortex-M4 build of MiniSolver stays within the
# project size budget. The budget is intentionally conservative; if a future
# refactor needs to grow it, update this script and the testing matrix entry
# for "Embedded ARM cross-build" in the same commit so the contract change is
# tracked.
#
# Usage: scripts/check_arm_size_budget.sh <build_dir>

set -euo pipefail

BUILD_DIR="${1:-build_arm}"
SMOKE_OBJECT="${BUILD_DIR}/CMakeFiles/minisolver_embedded_smoke.dir/cmake/embedded/smoke.cpp.o"

# Budget is in bytes. We measure the embedded smoke object: it instantiates
# the MiniSolver template at a representative small problem size and is
# therefore a stable proxy for "did we accidentally bloat the embedded path?".
#
# Override via environment if the budget needs to change for a specific
# experiment, but never silently bump it: see ROADMAP/testing-matrix for the
# accepted process.
BUDGET_BYTES="${MINISOLVER_ARM_SMOKE_BUDGET_BYTES:-262144}"  # 256 KiB

if [[ ! -f "${SMOKE_OBJECT}" ]]; then
    echo "[size-budget] expected smoke object ${SMOKE_OBJECT} not found"
    echo "[size-budget] hint: configure with -DMINISOLVER_EMBEDDED_PROFILE=ON,"
    echo "[size-budget]       then build the minisolver_embedded_smoke target"
    exit 1
fi

ACTUAL=$(stat -c%s "${SMOKE_OBJECT}")

echo "[size-budget] ${SMOKE_OBJECT}: ${ACTUAL} bytes (budget ${BUDGET_BYTES})"

if (( ACTUAL > BUDGET_BYTES )); then
    echo "[size-budget] FAIL: smoke object exceeds budget by $((ACTUAL - BUDGET_BYTES)) bytes"
    if command -v arm-none-eabi-size >/dev/null 2>&1; then
        echo "[size-budget] arm-none-eabi-size breakdown for diagnostics:"
        arm-none-eabi-size "${SMOKE_OBJECT}" || true
    fi
    exit 1
fi

if command -v arm-none-eabi-size >/dev/null 2>&1; then
    echo "[size-budget] arm-none-eabi-size summary:"
    arm-none-eabi-size "${SMOKE_OBJECT}" || true
fi

echo "[size-budget] OK"
