#!/bin/bash
set -e

mkdir -p build

echo ">> Compiling Benchmark (Extended Bicycle Model)..."

# Find Eigen if needed (still need headers even if using MiniMatrix fallback logic in riccati.h)
EIGEN_PATH="/usr/include/eigen3"
if [ ! -d "$EIGEN_PATH" ]; then
    EIGEN_PATH="/usr/local/include/eigen3"
fi

# Compile with MiniMatrix backend
g++ -O3 -std=c++17 -march=native -ffast-math \
    -Iinclude -I$EIGEN_PATH \
    -DUSE_CUSTOM_MATRIX \
    tools/demo_bicycle_ext/benchmark_ext.cpp -o build/benchmark_ext

echo ">> Running Benchmark..."
./build/benchmark_ext

