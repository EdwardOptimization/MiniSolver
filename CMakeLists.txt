cmake_minimum_required(VERSION 3.10)
project(RoboOpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(Eigen3 REQUIRED)
find_package(CUDA QUIET)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

# --- Sources ---
set(MAIN_SOURCES src/main.cpp)

# --- GPU Configuration ---
if(CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS ">> CUDA Found. Enabling MPX and PCR Backends.")
    add_definitions(-DUSE_CUDA)

    # Compile CUDA kernels into a static library
    add_library(gpu_backend STATIC src/cuda/gpu_ops.cu)
    set_target_properties(gpu_backend PROPERTIES CUDA_STANDARD 14)

    # Main Executable
    add_executable(RoboOptApp ${MAIN_SOURCES})
    target_link_libraries(RoboOptApp PRIVATE gpu_backend)
else()
    message(WARNING ">> CUDA NOT Found. Compiling CPU-Only Version.")
    # Main Executable (No GPU link)
    add_executable(RoboOptApp ${MAIN_SOURCES})
endif()
# Benchmark Tool
add_executable(benchmark_solver tools/benchmark_solver.cpp)
# No GPU link needed for this CPU benchmark
