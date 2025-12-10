#!/bin/bash
set -e

mkdir -p build

# Compile Benchmark with Custom Matrix Backend
echo ">> Compiling Benchmark (MiniMatrix)..."
# We define USE_CUSTOM_MATRIX to trigger the #else branch in matrix_defs.h
# Adjust paths for new location
g++ -O3 -std=c++17 -march=native -ffast-math -I../../include -Igenerated -DUSE_CUSTOM_MATRIX ../../tools/benchmark_suite/benchmark_suite.cpp -o build/benchmark_mini

# Compile Benchmark with Eigen Backend (Reference)
echo ">> Compiling Benchmark (Eigen)..."
# Find Eigen include path (common locations)
EIGEN_PATH="/usr/include/eigen3"
if [ ! -d "$EIGEN_PATH" ]; then
    EIGEN_PATH="/usr/local/include/eigen3"
fi
g++ -O3 -std=c++17 -march=native -ffast-math -I../../include -Igenerated -I$EIGEN_PATH -DUSE_EIGEN ../../tools/benchmark_suite/benchmark_suite.cpp -o build/benchmark_eigen

echo ">> Running Comparison..."
echo ""
echo "=== 1. MiniMatrix Backend (Custom) ==="
./build/benchmark_mini
echo ""
echo "=== 2. Eigen Backend ==="
./build/benchmark_eigen
echo ""
echo "=== 3. Eigen Backend ==="
./build/benchmark_eigen
echo ""
echo "=== 4. MiniMatrix Backend (Custom) ==="
./build/benchmark_mini