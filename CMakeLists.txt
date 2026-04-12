cmake_minimum_required(VERSION 3.5) # Compatible with older systems
project(MiniSolver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================
# Linear Algebra Backend Selection
# =============================================================
# These options control preprocessor macros used in include/minisolver/core/matrix_defs.h.
# Default is Eigen. If both are enabled, CustomMatrix takes precedence (matches matrix_defs.h).
option(USE_EIGEN "Use Eigen backend for linear algebra" ON)
option(USE_CUSTOM_MATRIX "Use custom MiniMatrix backend for linear algebra" OFF)

if(USE_CUSTOM_MATRIX AND USE_EIGEN)
    message(STATUS "USE_CUSTOM_MATRIX=ON overrides USE_EIGEN=ON")
    set(USE_EIGEN OFF CACHE BOOL "Use Eigen backend for linear algebra" FORCE)
endif()

if(NOT USE_EIGEN AND NOT USE_CUSTOM_MATRIX)
    message(STATUS "Neither USE_EIGEN nor USE_CUSTOM_MATRIX selected; defaulting to USE_EIGEN")
    set(USE_EIGEN ON CACHE BOOL "Use Eigen backend for linear algebra" FORCE)
endif()

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
        message(STATUS "CMake ${CMAKE_VERSION} >= 3.11, using FetchContent for Eigen3 (Header Only Mode)...")
        include(FetchContent)
        FetchContent_Declare(Eigen3 URL ${EIGEN_URL} DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        
        # Only download and extract, do not configure Eigen's CMakeLists.txt
        FetchContent_GetProperties(Eigen3)
        if(NOT eigen3_POPULATED)
            FetchContent_Populate(Eigen3)
            
            # Manually define Target to bypass all Eigen test logic
            if(NOT TARGET Eigen3::Eigen)
                add_library(Eigen3::Eigen INTERFACE IMPORTED)
                target_include_directories(Eigen3::Eigen INTERFACE "${eigen3_SOURCE_DIR}")
            endif()
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

# GTest is not a pure header-only library, we need to compile it, so keep MakeAvailable (or add_subdirectory)
# But we need to explicitly disable its install and test options
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.11")
    message(STATUS "CMake >= 3.11, using FetchContent to download GoogleTest...")
    include(FetchContent)
    FetchContent_Declare(googletest URL ${GTEST_URL} DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
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

# CUDA Support (experimental, disabled by default)
# The GPU backend under src/cuda/gpu_ops.cu is incomplete and emits #error
# at compile time, so CUDA is gated behind an opt-in option. Enable with
# -DMINISOLVER_USE_CUDA=ON once a working GPU implementation lands.
option(MINISOLVER_USE_CUDA "Enable CUDA GPU backend (incomplete, experimental)" OFF)
if(MINISOLVER_USE_CUDA)
    if(POLICY CMP0146)
        cmake_policy(SET CMP0146 OLD)
    endif()
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        add_definitions(-DUSE_CUDA)
        add_library(gpu_backend STATIC src/cuda/gpu_ops.cu)
    else()
        message(WARNING "MINISOLVER_USE_CUDA=ON but CUDA toolkit not found; GPU backend disabled.")
    endif()
endif()

# Define Interface Library for MiniSolver
add_library(minisolver_lib INTERFACE)
target_include_directories(minisolver_lib INTERFACE include)

if(USE_CUSTOM_MATRIX)
    target_compile_definitions(minisolver_lib INTERFACE USE_CUSTOM_MATRIX)
else()
    target_compile_definitions(minisolver_lib INTERFACE USE_EIGEN)
    target_link_libraries(minisolver_lib INTERFACE Eigen3::Eigen)
endif()

if(TARGET gpu_backend)
    target_link_libraries(minisolver_lib INTERFACE gpu_backend)
endif()

# --- Examples ---
add_subdirectory(examples/01_car_tutorial)
add_subdirectory(examples/02_advanced_bicycle)

# --- Tools ---
add_executable(benchmark_suite tools/benchmark_suite/benchmark_suite.cpp)
target_link_libraries(benchmark_suite minisolver_lib)

add_executable(fast_inverse_bench tools/fast_inverse_bench/fast_inverse_bench.cpp)
target_link_libraries(fast_inverse_bench minisolver_lib)

add_executable(auto_tuner tools/auto_tuner.cpp)
target_link_libraries(auto_tuner minisolver_lib)

add_executable(replay_solver tools/replay_solver.cpp)
target_link_libraries(replay_solver minisolver_lib)

# --- Unit Tests ---
enable_testing()
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(MINISOLVER_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
else()
    if(POLICY CMP0148)
        cmake_policy(SET CMP0148 OLD)
    endif()
    find_package(PythonInterp REQUIRED)
    set(MINISOLVER_PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

set(TEST_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated_tests")
set(KINEMATIC_BICYCLE_TEST_MODEL "${TEST_GENERATED_DIR}/kinematicbicycleregressionmodel.h")
set(DOUBLE_INTEGRATOR_3D_TEST_MODEL "${TEST_GENERATED_DIR}/doubleintegrator3dregressionmodel.h")
set(ASSET_REGRESSION_REFERENCE_HEADER "${TEST_GENERATED_DIR}/asset_regression_reference_data.h")

add_custom_command(
    OUTPUT ${KINEMATIC_BICYCLE_TEST_MODEL}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_GENERATED_DIR}
    COMMAND ${MINISOLVER_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_kinematic_bicycle_regression_model.py ${TEST_GENERATED_DIR}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_kinematic_bicycle_regression_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/MiniModel.py
    COMMENT "Generating kinematic bicycle regression test model"
)

add_custom_command(
    OUTPUT ${DOUBLE_INTEGRATOR_3D_TEST_MODEL}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_GENERATED_DIR}
    COMMAND ${MINISOLVER_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_double_integrator_3d_regression_model.py ${TEST_GENERATED_DIR}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_double_integrator_3d_regression_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/MiniModel.py
    COMMENT "Generating double integrator 3D regression test model"
)

add_custom_command(
    OUTPUT ${ASSET_REGRESSION_REFERENCE_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_GENERATED_DIR}
    COMMAND ${MINISOLVER_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/generate_asset_regression_reference_data.py ${ASSET_REGRESSION_REFERENCE_HEADER}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/generate_asset_regression_reference_data.py
    COMMENT "Generating asset regression reference data"
)

add_custom_target(minisolver_test_models
    DEPENDS
        ${KINEMATIC_BICYCLE_TEST_MODEL}
        ${DOUBLE_INTEGRATOR_3D_TEST_MODEL}
        ${ASSET_REGRESSION_REFERENCE_HEADER}
)

function(add_minisolver_test test_name source_file)
    add_executable(${test_name} ${source_file})
    # Link GTest main
    target_link_libraries(${test_name} minisolver_lib gtest_main)
    target_include_directories(${test_name} PRIVATE examples/01_car_tutorial/generated)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Register Tests
# --- Core ---
add_minisolver_test(test_matrix tests/test_matrix.cpp)
add_minisolver_test(test_mini_matrix tests/test_mini_matrix.cpp)
add_minisolver_test(test_memory tests/test_memory.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_memory PRIVATE -Wno-mismatched-new-delete)
endif()
add_minisolver_test(test_integrator tests/test_integrator.cpp)

# --- Solver Components ---
add_minisolver_test(test_riccati tests/test_riccati.cpp)
add_minisolver_test(test_line_search tests/test_line_search.cpp)
add_minisolver_test(test_autodiff tests/test_autodiff.cpp)
add_minisolver_test(test_status tests/test_status.cpp)
add_minisolver_test(test_robustness tests/test_robustness.cpp)

# --- Solver Integration ---
add_minisolver_test(test_solver tests/test_solver.cpp)
add_minisolver_test(test_features tests/test_features.cpp)
add_minisolver_test(test_advanced tests/test_advanced.cpp)
add_minisolver_test(test_soft_constraints tests/test_soft_constraints.cpp)
add_minisolver_test(test_serializer tests/test_serializer.cpp)

# --- Quality & Regression ---
add_minisolver_test(test_bugfixes tests/test_bugfixes.cpp)
add_minisolver_test(test_solver_quality tests/test_solver_quality.cpp)
add_executable(test_asset_regressions
    tests/test_asset_regressions.cpp
    ${KINEMATIC_BICYCLE_TEST_MODEL}
    ${DOUBLE_INTEGRATOR_3D_TEST_MODEL}
    ${ASSET_REGRESSION_REFERENCE_HEADER}
)
add_dependencies(test_asset_regressions minisolver_test_models)
target_link_libraries(test_asset_regressions minisolver_lib gtest_main)
target_include_directories(test_asset_regressions PRIVATE ${TEST_GENERATED_DIR})
add_test(NAME test_asset_regressions COMMAND test_asset_regressions)
