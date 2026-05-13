#!/bin/bash
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

mkdir -p "$SCRIPT_DIR/build"

CXX=${CXX:-g++}
PYTHON=${PYTHON:-python3}
# Benchmark-only flags: these favor local speed measurement over portable binaries.
CXXFLAGS=(-O3 -std=c++17 -march=native -ffast-math)

restore_default_model() {
    echo ""
    echo ">> Restoring Default Model (Fused Riccati ENABLED)..."
    "$PYTHON" "$SCRIPT_DIR/generate_advanced_model.py"
}

trap restore_default_model EXIT

generate_and_compile() {
    local obstacle_mode="$1"
    local riccati_mode="$2"
    local output="$SCRIPT_DIR/build/benchmark_${obstacle_mode}_${riccati_mode}"
    local generate_args=()

    if [[ "$obstacle_mode" == "quad" ]]; then
        generate_args+=(--with-obstacle-quad)
    fi
    if [[ "$riccati_mode" == "standard" ]]; then
        generate_args+=(--no-fused)
    fi

    echo ""
    echo ">> Generating model: obstacle=${obstacle_mode}, riccati=${riccati_mode}"
    "$PYTHON" "$SCRIPT_DIR/generate_advanced_model.py" "${generate_args[@]}"

    echo ">> Compiling benchmark: obstacle=${obstacle_mode}, riccati=${riccati_mode}"
    "$CXX" "${CXXFLAGS[@]}" \
        -I"$PROJECT_ROOT/include" -I"$SCRIPT_DIR/generated" \
        -DUSE_CUSTOM_MATRIX \
        "$SCRIPT_DIR/advanced_benchmark.cpp" \
        -o "$output"
}

run_case() {
    local obstacle_mode="$1"
    local riccati_mode="$2"
    echo ""
    echo ">> Running benchmark: obstacle=${obstacle_mode}, riccati=${riccati_mode}"
    "$SCRIPT_DIR/build/benchmark_${obstacle_mode}_${riccati_mode}"
}

echo ">> Compiling Advanced Benchmark matrix (no_quad/quad x fused/standard)..."

for obstacle_mode in no_quad quad; do
    for riccati_mode in fused standard; do
        generate_and_compile "$obstacle_mode" "$riccati_mode"
    done
done

for obstacle_mode in no_quad quad; do
    for riccati_mode in fused standard; do
        run_case "$obstacle_mode" "$riccati_mode"
    done
done
