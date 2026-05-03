#!/bin/bash
set -e # Stop immediately on error

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}>> [MiniSolver] One-click build started...${NC}"

BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_EXTRA_ARGS=()

if [[ "${ASAN:-0}" == "1" || "${UBSAN:-0}" == "1" || "${SANITIZE:-0}" == "1" ]]; then
    SANITIZER_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
    echo -e "${BLUE}>> Sanitizer mode enabled: ${SANITIZER_FLAGS}${NC}"
    CMAKE_EXTRA_ARGS+=(
        "-DCMAKE_BUILD_TYPE=Debug"
        "-DCMAKE_CXX_FLAGS=${SANITIZER_FLAGS}"
        "-DCMAKE_C_FLAGS=${SANITIZER_FLAGS}"
        "-DCMAKE_EXE_LINKER_FLAGS=${SANITIZER_FLAGS}"
    )
fi

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
rm -rf "${BUILD_DIR}" && mkdir "${BUILD_DIR}" && cd "${BUILD_DIR}"
cmake .. "${CMAKE_EXTRA_ARGS[@]}"
# Use nproc (Linux) or sysctl (Mac) to determine core count, default to 4
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ==========================================
# 5. Run Tests
# ==========================================
echo -e "${GREEN}>> [5/5] Running tests...${NC}"
ctest --output-on-failure

echo -e "${GREEN}>> Build successful!${NC}"
