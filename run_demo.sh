#!/bin/bash
set -e

echo ">> 0. Generating Model..."
python3 tools/car_model_gen.py

echo ">> 1. Building C++ Project..."
# Clean build to ensure fresh config
rm -rf build
mkdir -p build
cd build

# Configure and Build
cmake ..
make MiniSolverApp -j4

cd ..

echo ">> 2. Running Solver..."
if [ -f "./build/MiniSolverApp" ]; then
    ./build/MiniSolverApp
else
    echo "Error: MiniSolverApp executable not found!"
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
