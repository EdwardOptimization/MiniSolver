#!/bin/bash
set -e

mkdir -p build

echo ">> Compiling Debug Demo (Extended Bicycle)..."

EIGEN_PATH="/usr/include/eigen3"
if [ ! -d "$EIGEN_PATH" ]; then
    EIGEN_PATH="/usr/local/include/eigen3"
fi

# Compile with MiniMatrix backend and Debug info
g++ -O3 -std=c++17 -march=native -ffast-math \
    -Iinclude -I$EIGEN_PATH \
    -DUSE_CUSTOM_MATRIX \
    -DMINISOLVER_LOG_LEVEL=4 \
    tools/demo_bicycle_ext/debug_ext.cpp -o build/debug_ext

echo ">> Running Debug Demo..."
./build/debug_ext

