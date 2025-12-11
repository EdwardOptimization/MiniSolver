#!/bin/bash
set -e # Stop immediately on error

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}>> [MiniSolver] One-click build started...${NC}"

# ==========================================
# 1. Simple Environment Check
# ==========================================
echo -e "${GREEN}>> [1/5] Checking system tools...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake not found.${NC}"
    exit 1
fi
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}Error: g++ not found.${NC}"
    exit 1
fi

# ==========================================
# 2. Python Dependencies (SymPy)
# ==========================================
echo -e "${GREEN}>> [2/5] Checking Python dependencies (SymPy)...${NC}"

if ! python3 -c "import sympy" &> /dev/null; then
    echo -e "${GREEN}>> SymPy not found, installing...${NC}"
    pip3 install --user sympy || { echo -e "${RED}Installation failed, please run: pip3 install sympy${NC}"; exit 1; }
fi

# ==========================================
# 3. Generate Code
# ==========================================
echo -e "${GREEN}>> [3/5] Generating model code...${NC}"
export PYTHONPATH=$PYTHONPATH:$(pwd)/python
python3 examples/01_car_tutorial/generate_model.py
python3 examples/02_advanced_bicycle/generate_advanced_model.py

# ==========================================
# 4. Configure & Build
# ==========================================
echo -e "${GREEN}>> [4/5] CMake configuration and compilation...${NC}"
rm -rf build && mkdir build && cd build
cmake ..
# Use nproc (Linux) or sysctl (Mac) to determine core count, default to 4
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ==========================================
# 5. Run Tests
# ==========================================
echo -e "${GREEN}>> [5/5] Running tests...${NC}"
ctest --output-on-failure

echo -e "${GREEN}>> Build successful!${NC}"