cmake_minimum_required(VERSION 3.5) # Compatible with older systems
project(MiniSolver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

# =============================================================
# Intelligent Dependency Management: Adapts to CMake version
# =============================================================

# Define manual download function (fallback for older CMake)
function(download_fallback NAME URL DEST_DIR)
    set(ARCHIVE_FILE "${DEST_DIR}/${NAME}.zip")
    set(SOURCE_DIR "${DEST_DIR}/${NAME}-src")
    
    if(EXISTS "${SOURCE_DIR}/CMakeLists.txt" OR EXISTS "${SOURCE_DIR}/COPYING")
        return() # Already exists, skipping
    endif()

    message(STATUS "[Fallback] Downloading ${NAME} for older CMake: ${URL}")
    file(MAKE_DIRECTORY "${DEST_DIR}")
    
    # 1. Download
    file(DOWNLOAD "${URL}" "${ARCHIVE_FILE}" SHOW_PROGRESS STATUS status)
    list(GET status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Download failed: ${status}")
    endif()

    # 2. Extract (requires unzip)
    find_program(UNZIP_EXE unzip)
    if(NOT UNZIP_EXE)
        message(FATAL_ERROR "unzip tool not found. Older CMake versions require unzip to extract dependencies.")
    endif()

    file(MAKE_DIRECTORY "${SOURCE_DIR}")
    execute_process(
        COMMAND ${UNZIP_EXE} "${ARCHIVE_FILE}" -d "${DEST_DIR}"
        RESULT_VARIABLE res OUTPUT_QUIET
    )
    
    # Simplification: usually extracted as a folder with a version number.
    # Here we assume there's only one new folder in the parent dir after extraction.
    file(GLOB CONTENTS "${DEST_DIR}/*")
    foreach(ITEM ${CONTENTS})
        if(IS_DIRECTORY "${ITEM}" AND NOT "${ITEM}" STREQUAL "${SOURCE_DIR}")
            set(EXTRACTED_DIR "${ITEM}")
            break()
        endif()
    endforeach()
    
    if(EXTRACTED_DIR)
        # Move content to standard path for reference
        file(RENAME "${EXTRACTED_DIR}" "${SOURCE_DIR}_tmp")
        file(REMOVE_RECURSE "${SOURCE_DIR}")
        file(RENAME "${SOURCE_DIR}_tmp" "${SOURCE_DIR}")
    endif()
    
    file(REMOVE "${ARCHIVE_FILE}")
endfunction()


# --- 1. Eigen3 Dependency Handling ---
find_package(Eigen3 3.3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found System Eigen3: ${EIGEN3_INCLUDE_DIR}")
    # Create alias to unify interface
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
    endif()
else()
    set(EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip")
    
    # >>> Core Version Detection Logic <<<
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.11")
        message(STATUS "CMake ${CMAKE_VERSION} >= 3.11, using FetchContent to download Eigen3...")
        include(FetchContent)
        FetchContent_Declare(Eigen3 URL ${EIGEN_URL})
        FetchContent_MakeAvailable(Eigen3)
        # Fix potential issue where FetchContent doesn't expose the Target
        if(NOT TARGET Eigen3::Eigen)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            target_include_directories(Eigen3::Eigen INTERFACE ${eigen3_SOURCE_DIR})
        endif()
    else()
        message(STATUS "CMake ${CMAKE_VERSION} < 3.11, using manual fallback mode to download Eigen3...")
        set(DEPS_DIR "${CMAKE_BINARY_DIR}/deps")
        download_fallback("Eigen3" ${EIGEN_URL} "${DEPS_DIR}")
        
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE "${DEPS_DIR}/Eigen3-src")
    endif()
endif()

# --- 2. GoogleTest Dependency Handling ---
set(GTEST_URL "https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# >>> Core Version Detection Logic <<<
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.11")
    message(STATUS "CMake >= 3.11, using FetchContent to download GoogleTest...")
    include(FetchContent)
    FetchContent_Declare(googletest URL ${GTEST_URL})
    FetchContent_MakeAvailable(googletest)
else()
    message(STATUS "CMake < 3.11, using manual fallback mode to download GoogleTest...")
    set(DEPS_DIR "${CMAKE_BINARY_DIR}/deps")
    download_fallback("googletest" ${GTEST_URL} "${DEPS_DIR}")
    # Manually add subdirectory
    add_subdirectory("${DEPS_DIR}/googletest-src" "${CMAKE_BINARY_DIR}/googletest-build" EXCLUDE_FROM_ALL)
endif()


# =============================================================
# Project Build (Unchanged)
# =============================================================

# CUDA Support
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    add_library(gpu_backend STATIC src/cuda/gpu_ops.cu)
endif()

# Define Interface Library for MiniSolver
add_library(minisolver_lib INTERFACE)
target_include_directories(minisolver_lib INTERFACE include)
target_link_libraries(minisolver_lib INTERFACE Eigen3::Eigen)

if(TARGET gpu_backend)
    target_link_libraries(minisolver_lib INTERFACE gpu_backend)
endif()

# --- Examples ---
add_subdirectory(examples/01_car_tutorial)
add_subdirectory(examples/02_advanced_bicycle)

# --- Tools ---
add_executable(benchmark_suite tools/benchmark_suite/benchmark_suite.cpp)
target_link_libraries(benchmark_suite minisolver_lib)

add_executable(auto_tuner tools/auto_tuner.cpp)
target_link_libraries(auto_tuner minisolver_lib)

add_executable(replay_solver tools/replay_solver.cpp)
target_link_libraries(replay_solver minisolver_lib)

# --- Unit Tests ---
enable_testing()

function(add_minisolver_test test_name source_file)
    add_executable(${test_name} ${source_file})
    # Link GTest main
    target_link_libraries(${test_name} minisolver_lib gtest_main)
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
add_minisolver_test(test_different_soft_constraints tests/test_different_soft_constraints.cpp)
add_minisolver_test(test_serializer tests/test_serializer.cpp)
