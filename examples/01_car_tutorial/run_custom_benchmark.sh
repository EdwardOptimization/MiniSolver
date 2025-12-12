#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

mkdir -p "$SCRIPT_DIR/build"

# Compile Benchmark with Custom Matrix Backend
echo ">> Compiling Benchmark (MiniMatrix)..."
# We define USE_CUSTOM_MATRIX to trigger the #else branch in matrix_defs.h
g++ -O3 -std=c++17 -march=native -ffast-math \
    -I"$PROJECT_ROOT/include" -I"$SCRIPT_DIR/generated" \
    -DUSE_CUSTOM_MATRIX \
    "$PROJECT_ROOT/tools/benchmark_suite/benchmark_suite.cpp" \
    -o "$SCRIPT_DIR/build/benchmark_mini"

# Compile Benchmark with Eigen Backend (Reference)
echo ">> Compiling Benchmark (Eigen)..."
# Find Eigen include path (common locations)
EIGEN_PATH="/usr/include/eigen3"
if [ ! -d "$EIGEN_PATH" ]; then
    EIGEN_PATH="/usr/local/include/eigen3"
fi
g++ -O3 -std=c++17 -march=native -ffast-math \
    -I"$PROJECT_ROOT/include" -I"$SCRIPT_DIR/generated" -I$EIGEN_PATH \
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
echo "=== 3. Eigen Backend ==="
"$SCRIPT_DIR/build/benchmark_eigen"
echo ""
echo "=== 4. MiniMatrix Backend (Custom) ==="
"$SCRIPT_DIR/build/benchmark_mini"