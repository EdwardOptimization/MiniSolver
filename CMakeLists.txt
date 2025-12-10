cmake_minimum_required(VERSION 3.10)
project(MiniSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# Dependencies
find_package(Eigen3 REQUIRED)

# Include Directories (Core Library)
include_directories(include) 
include_directories(${EIGEN3_INCLUDE_DIR})

# CUDA Support
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    add_library(gpu_backend STATIC src/cuda/gpu_ops.cu)
    set_target_properties(gpu_backend PROPERTIES CUDA_STANDARD 14)
endif()

# Define Interface Library for MiniSolver
add_library(minisolver_lib INTERFACE)
target_include_directories(minisolver_lib INTERFACE include ${EIGEN3_INCLUDE_DIR})
if(TARGET gpu_backend)
    target_link_libraries(minisolver_lib INTERFACE gpu_backend)
endif()

# --- Examples ---
add_subdirectory(examples/01_car_tutorial)
add_subdirectory(examples/02_advanced_bicycle)

# --- Tools ---
# These tools depend on the CarModel from Example 1, so they are slightly coupled.
add_executable(benchmark_suite tools/benchmark_suite/benchmark_suite.cpp)
target_link_libraries(benchmark_suite PRIVATE minisolver_lib)

add_executable(auto_tuner tools/auto_tuner.cpp)
target_link_libraries(auto_tuner PRIVATE minisolver_lib)

add_executable(replay_solver tools/replay_solver.cpp)
target_link_libraries(replay_solver PRIVATE minisolver_lib)

# --- Unit Tests ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

function(add_minisolver_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE minisolver_lib GTest::gtest_main)
    # Tests rely on the CarModel generated in Example 1
    target_include_directories(${test_name} PRIVATE examples/01_car_tutorial/generated)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Register Tests
add_minisolver_test(test_matrix tests/test_matrix.cpp)
add_minisolver_test(test_autodiff tests/test_autodiff.cpp)
add_minisolver_test(test_riccati tests/test_riccati.cpp)
add_minisolver_test(test_linesearch tests/test_line_search.cpp)
add_minisolver_test(test_solver tests/test_solver.cpp)
add_minisolver_test(test_advanced tests/test_advanced.cpp)
add_minisolver_test(test_soft_constraints tests/test_soft_constraints.cpp)
add_minisolver_test(test_soc tests/test_soc.cpp)
add_minisolver_test(test_ir tests/test_ir.cpp)
