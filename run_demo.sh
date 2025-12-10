#!/bin/bash
set -e

echo ">> 1. Building C++ Project (and generating models)..."
# Clean build to ensure fresh config
rm -rf build
mkdir -p build
cd build

# Configure and Build
cmake ..
make car_demo -j4

cd ..

echo ">> 2. Running Solver..."
# Path depends on CMake version/layout, but usually mirrors source
BIN_PATH="./build/examples/01_car_tutorial/car_demo"

if [ -f "$BIN_PATH" ]; then
    $BIN_PATH
else
    echo "Error: car_demo executable not found at $BIN_PATH"
    # Try to find it
    find ./build -name car_demo
    exit 1
fi

echo ">> 3. Plotting Results..."
if [ -f "trajectory.csv" ]; then
    python3 tools/plot_trajectory.py trajectory.csv
    echo ">> Done. Check trajectory_plot.png"
else
    echo "Error: trajectory.csv not generated!"
    exit 1
fi
