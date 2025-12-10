#!/bin/bash
set -e

# 1. Generate Models using Python
echo ">> [1/5] Generating Models..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/python

echo "   - Generating Car Model..."
python3 examples/01_car_tutorial/generate_model.py

echo "   - Generating Bicycle Model..."
python3 examples/02_advanced_bicycle/generate.py

# 2. Configure CMake
echo ">> [2/5] Configuring CMake..."
rm -rf build
mkdir -p build
cd build
cmake ..

# 3. Build Project
echo ">> [3/5] Building Project..."
make -j4

# 4. Run Benchmark Suite
echo ">> [4/5] Running Benchmark Suite..."
if [ -f "tools/benchmark_suite" ]; then
    ./tools/benchmark_suite
elif [ -f "benchmark_suite" ]; then
    ./benchmark_suite
else
    # Fallback search
    find . -name benchmark_suite -exec {} \;
fi

# 5. Run Tests
echo ">> [5/5] Running Unit Tests..."
ctest --output-on-failure

echo ">> BUILD & TEST COMPLETE."

