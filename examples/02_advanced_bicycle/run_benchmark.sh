#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

mkdir -p "$SCRIPT_DIR/build"

echo ">> Compiling Advanced Benchmark (Fused vs Standard)..."

# 1. Compile Fused Riccati (Default generated model has it enabled)
# Using MiniMatrix as backend for embedded relevance
g++ -O3 -std=c++17 -march=native -ffast-math \
    -I"$PROJECT_ROOT/include" -I"$SCRIPT_DIR/generated" \
    -DUSE_CUSTOM_MATRIX \
    "$SCRIPT_DIR/advanced_benchmark.cpp" \
    -o "$SCRIPT_DIR/build/benchmark_fused"

echo ">> Running Fused Riccati Benchmark..."
"$SCRIPT_DIR/build/benchmark_fused"

echo ""
echo ">> Regenerating Model with Fused Riccati DISABLED..."
cd "$SCRIPT_DIR"
python3 generate_advanced_model.py --no-fused

echo ">> Compiling Standard Benchmark..."
g++ -O3 -std=c++17 -march=native -ffast-math \
    -I"$PROJECT_ROOT/include" -I"$SCRIPT_DIR/generated" \
    -DUSE_CUSTOM_MATRIX \
    "$SCRIPT_DIR/advanced_benchmark.cpp" \
    -o "$SCRIPT_DIR/build/benchmark_standard"

echo ">> Running Standard Benchmark..."
"$SCRIPT_DIR/build/benchmark_standard"

echo ""
echo ">> Restoring Default Model (Fused Riccati ENABLED)..."
python3 generate_advanced_model.py


