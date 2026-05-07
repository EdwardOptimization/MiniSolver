#!/usr/bin/env bash
# Verify that the embedded ARM Cortex-M4 build of MiniSolver stays within the
# project size budget. The budget is intentionally conservative; if a future
# refactor needs to grow it, update this script and the testing matrix entry
# for "Embedded ARM cross-build" in the same commit so the contract change is
# tracked.
#
# Usage: scripts/check_arm_size_budget.sh <build_dir>
#
# The metric is `text + data + bss` reported by `arm-none-eabi-size` on the
# embedded smoke object, NOT the on-disk file size. Rationale:
#   * `text + data + bss` matches what actually lands on the target (flash
#     for text/data, RAM for data/bss), which is the embedded contract the
#     budget guards.
#   * On-disk size is dominated by DWARF debug info, ELF metadata, symbol
#     tables, and relocations - none of which are flashed and which can swing
#     wildly with `-g0`/`-g1`/`-g3`, optimisation level, and cross-toolchain
#     version. Tying the budget to that number produces noisy regressions
#     and false stability.
#   * `arm-none-eabi-size` is part of the ARM cross-binutils package that the
#     toolchain already requires, so we can mandate it.

set -euo pipefail

BUILD_DIR="${1:-build_arm}"
SMOKE_OBJECT="${BUILD_DIR}/CMakeFiles/minisolver_embedded_smoke.dir/cmake/embedded/smoke.cpp.o"

# Budget is in bytes for `text + data + bss` (i.e. the on-target footprint).
# Override via environment if the budget needs to change for a specific
# experiment, but never silently bump it: see ROADMAP/testing-matrix for the
# accepted process.
BUDGET_BYTES="${MINISOLVER_ARM_SMOKE_BUDGET_BYTES:-65536}"  # 64 KiB

if [[ ! -f "${SMOKE_OBJECT}" ]]; then
    echo "[size-budget] expected smoke object ${SMOKE_OBJECT} not found"
    echo "[size-budget] hint: configure with -DMINISOLVER_EMBEDDED_PROFILE=ON,"
    echo "[size-budget]       then build the minisolver_embedded_smoke target"
    exit 1
fi

if ! command -v arm-none-eabi-size >/dev/null 2>&1; then
    echo "[size-budget] FAIL: arm-none-eabi-size is required to measure"
    echo "[size-budget]       text+data+bss but is not on PATH."
    echo "[size-budget]       hint: apt install binutils-arm-none-eabi"
    exit 1
fi

# Parse `arm-none-eabi-size` Berkeley-format output:
#
#    text    data     bss     dec     hex filename
#   12345     678     900   13923    3663 smoke.cpp.o
#
# The second line's first three columns are text/data/bss in bytes. The
# fourth column (`dec`) is their sum already, but we recompute it ourselves
# so the breakdown is explicit in the failure message.
SIZE_OUTPUT=$(arm-none-eabi-size --format=berkeley "${SMOKE_OBJECT}")
read -r SIZE_TEXT SIZE_DATA SIZE_BSS _ _ _ < <(echo "${SIZE_OUTPUT}" | tail -n 1)

ACTUAL=$((SIZE_TEXT + SIZE_DATA + SIZE_BSS))

echo "[size-budget] ${SMOKE_OBJECT}"
echo "[size-budget]   text=${SIZE_TEXT} data=${SIZE_DATA} bss=${SIZE_BSS}"
echo "[size-budget]   text+data+bss=${ACTUAL} (budget ${BUDGET_BYTES})"

if (( ACTUAL > BUDGET_BYTES )); then
    echo "[size-budget] FAIL: text+data+bss exceeds budget by $((ACTUAL - BUDGET_BYTES)) bytes"
    echo "[size-budget] arm-none-eabi-size full breakdown:"
    echo "${SIZE_OUTPUT}"
    exit 1
fi

echo "[size-budget] OK"
