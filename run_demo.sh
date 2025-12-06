#!/bin/bash
set -e

echo ">> 1. Building C++ Project..."
rm -rf build
mkdir -p build
cd build
cmake ..
make -j4
cd ..

echo ">> 2. Running Solver..."
./build/MiniSolverApp

echo ">> 3. Plotting Results..."
python3 tools/plot_trajectory.py trajectory.csv

echo ">> Done. Check trajectory_plot.png"

