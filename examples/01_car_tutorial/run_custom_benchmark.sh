#!/bin/bash
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

mkdir -p "$SCRIPT_DIR/build"

CXX=${CXX:-g++}
# Benchmark-only flags: these favor local speed measurement over portable binaries.
CXXFLAGS=(-O3 -std=c++17 -march=native -ffast-math)

# Compile Benchmark with Custom Matrix Backend
echo ">> Compiling Benchmark (MiniMatrix)..."
# We define USE_CUSTOM_MATRIX to trigger the #else branch in matrix_defs.h
"$CXX" "${CXXFLAGS[@]}" \
    -I"$PROJECT_ROOT/include" -I"$SCRIPT_DIR/generated" \
    -DUSE_CUSTOM_MATRIX \
    "$PROJECT_ROOT/tools/benchmark_suite/benchmark_suite.cpp" \
    -o "$SCRIPT_DIR/build/benchmark_mini"

# Compile Benchmark with Eigen Backend (Reference)
echo ">> Compiling Benchmark (Eigen)..."
# Find Eigen include path (common locations). Override with EIGEN_PATH=/path if needed.
EIGEN_PATH=${EIGEN_PATH:-/usr/include/eigen3}
if [ ! -d "$EIGEN_PATH" ] && [ -d "/usr/local/include/eigen3" ]; then
    EIGEN_PATH="/usr/local/include/eigen3"
fi
if [ ! -d "$EIGEN_PATH" ]; then
    echo "Error: Eigen include directory not found." >&2
    echo "Install Eigen3 or run with EIGEN_PATH=/path/to/eigen3." >&2
    exit 1
fi
"$CXX" "${CXXFLAGS[@]}" \
    -I"$PROJECT_ROOT/include" -I"$SCRIPT_DIR/generated" -I"$EIGEN_PATH" \
    -DUSE_EIGEN \
    "$PROJECT_ROOT/tools/benchmark_suite/benchmark_suite.cpp" \
    -o "$SCRIPT_DIR/build/benchmark_eigen"

echo ">> Running Comparison..."
echo ""
echo "=== 1. MiniMatrix Backend (Custom) ==="
"$SCRIPT_DIR/build/benchmark_mini"
echo ""
echo "=== 2. Eigen Backend ==="
"$SCRIPT_DIR/build/benchmark_eigen"
echo ""
echo "=== 3. Eigen Backend (repeat) ==="
"$SCRIPT_DIR/build/benchmark_eigen"
echo ""
echo "=== 4. MiniMatrix Backend (Custom, repeat) ==="
"$SCRIPT_DIR/build/benchmark_mini"
