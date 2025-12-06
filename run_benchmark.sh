#!/bin/bash
set -e

echo ">> Building Benchmark..."
mkdir -p build
cd build
cmake ..
# We need to compile benchmark_solver separately or add it to CMake
# For simplicity, let's compile it manually if CMakeLists is not updated
# But updating CMakeLists is cleaner. Let's assume user wants me to do it.
# Wait, I can't edit CMakeLists here easily without seeing it.
# I'll just use g++ directly for the benchmark tool since it's simple.
# Include paths: ../include, ../include/eigen3 (system default usually)

# But wait, I need to know where Eigen is.
# CMakeLists said: find_package(Eigen3 REQUIRED)
# I'll try to update CMakeLists.txt instead.
cd ..

# Append benchmark target to CMakeLists if not present
if ! grep -q "benchmark_solver" CMakeLists.txt; then
    echo "Adding benchmark_solver to CMakeLists.txt..."
    cat <<EOF >> CMakeLists.txt

# Benchmark Tool
add_executable(benchmark_solver tools/benchmark_solver.cpp)
# No GPU link needed for this CPU benchmark
EOF
fi

cd build
cmake ..
make benchmark_solver -j4

echo ">> Running Benchmark..."
./benchmark_solver

