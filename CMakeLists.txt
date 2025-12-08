cmake_minimum_required(VERSION 3.10)
project(MiniSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags (CRITICAL for Eigen performance)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# Matrix Backend Option
option(USE_EIGEN "Use Eigen3 backend" ON)

if(USE_EIGEN)
    add_definitions(-DUSE_EIGEN)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    # Add logic for custom matrix lib here
    message(STATUS "Using Custom Matrix Backend")
endif()

# --- Dependencies ---
# find_package(Eigen3 REQUIRED) -> Moved up
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
    add_executable(MiniSolverApp ${MAIN_SOURCES})
    target_link_libraries(MiniSolverApp PRIVATE gpu_backend)
else()
    message(WARNING ">> CUDA NOT Found. Compiling CPU-Only Version.")
    # Main Executable (No GPU link)
    add_executable(MiniSolverApp ${MAIN_SOURCES})
endif()
# Benchmark Tool
add_executable(benchmark_solver tools/benchmark_solver.cpp)

# Auto Tuner Tool
add_executable(auto_tuner tools/auto_tuner.cpp)

# Replay Tool
add_executable(replay_solver tools/replay_solver.cpp)

# --- Unit Tests (GoogleTest) ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Helper to add test
function(add_minisolver_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} GTest::gtest_main)
    target_include_directories(${test_name} PRIVATE include)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Register Tests (Files will be created in next steps)
add_minisolver_test(test_matrix tests/test_matrix.cpp)
add_minisolver_test(test_autodiff tests/test_autodiff.cpp)
add_minisolver_test(test_riccati tests/test_riccati.cpp)
add_minisolver_test(test_linesearch tests/test_line_search.cpp)
add_minisolver_test(test_solver tests/test_solver.cpp)
add_minisolver_test(test_advanced tests/test_advanced.cpp)
add_minisolver_test(test_soft_constraints tests/test_soft_constraints.cpp)
add_minisolver_test(test_soc tests/test_soc.cpp)

# No GPU link needed for this CPU benchmark
