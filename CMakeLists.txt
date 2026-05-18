cmake_minimum_required(VERSION 3.5) # Compatible with older systems
project(MiniSolver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================
# Linear Algebra Backend Selection
# =============================================================
# These options control preprocessor macros used in include/minisolver/matrix/matrix_defs.h.
# Default is Eigen. If both are enabled, CustomMatrix takes precedence (matches matrix_defs.h).
option(USE_EIGEN "Use Eigen backend for linear algebra" ON)
option(USE_CUSTOM_MATRIX "Use custom MiniMatrix backend for linear algebra" OFF)
set(MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK 256 CACHE STRING
    "Maximum scalar work count for generic MiniMatrix static unrolling")
set(MINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N 4 CACHE STRING
    "Maximum fixed LDLT size for static outer-loop unrolling")
set(MINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N 4 CACHE STRING
    "Maximum fixed LDLT size for static row-loop unrolling")
set(MINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N 8 CACHE STRING
    "Maximum fixed LDLT size for static inner-reduction unrolling")

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

option(MINISOLVER_ENABLE_FAST_MATH "Enable -ffast-math for benchmark builds" OFF)
option(MINISOLVER_ENABLE_NATIVE_ARCH "Enable -march=native for local benchmark builds" OFF)
option(MINISOLVER_BUILD_TESTS "Build MiniSolver tests" ON)
option(MINISOLVER_BUILD_EXAMPLES "Build MiniSolver examples" ON)
option(MINISOLVER_BUILD_TOOLS "Build MiniSolver tools and local benchmarks" ON)
option(MINISOLVER_BUILD_CUDA_BENCHMARKS "Build standalone CUDA exploratory benchmarks" OFF)
option(MINISOLVER_FETCH_DEPS "Fetch missing third-party dependencies during configure" ON)
set(MINISOLVER_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures for standalone CUDA benchmarks, e.g. native, 90, 120")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra)
    if(MINISOLVER_ENABLE_FAST_MATH)
        add_compile_options(-ffast-math)
    endif()
    if(MINISOLVER_ENABLE_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
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


set(MINISOLVER_NEEDS_EIGEN OFF)
if(USE_EIGEN OR MINISOLVER_BUILD_TOOLS)
    set(MINISOLVER_NEEDS_EIGEN ON)
endif()

# --- 1. Eigen3 Dependency Handling ---
if(MINISOLVER_NEEDS_EIGEN)
    find_package(Eigen3 3.3 QUIET)
    if(Eigen3_FOUND)
        message(STATUS "Found System Eigen3: ${EIGEN3_INCLUDE_DIR}")
        # Create alias to unify interface
        if(NOT TARGET Eigen3::Eigen)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
        endif()
    else()
        if(NOT MINISOLVER_FETCH_DEPS)
            message(FATAL_ERROR "Eigen3 not found and MINISOLVER_FETCH_DEPS=OFF")
        endif()
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
endif()

if(MINISOLVER_BUILD_TESTS)
    # --- 2. GoogleTest Dependency Handling ---
    set(GTEST_URL "https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # GTest is not a pure header-only library, we need to compile it, so keep MakeAvailable (or add_subdirectory)
    # But we need to explicitly disable its install and test options
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.11")
        if(NOT MINISOLVER_FETCH_DEPS)
            find_package(GTest REQUIRED)
        else()
            message(STATUS "CMake >= 3.11, using FetchContent to download GoogleTest...")
            include(FetchContent)
            FetchContent_Declare(googletest URL ${GTEST_URL} DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
            FetchContent_MakeAvailable(googletest)
        endif()
    else()
        if(NOT MINISOLVER_FETCH_DEPS)
            message(FATAL_ERROR "GoogleTest requires system package support when MINISOLVER_FETCH_DEPS=OFF")
        endif()
        message(STATUS "CMake < 3.11, using manual fallback mode to download GoogleTest...")
        set(DEPS_DIR "${CMAKE_BINARY_DIR}/deps")
        download_fallback("googletest" ${GTEST_URL} "${DEPS_DIR}")
        # Manually add subdirectory
        add_subdirectory("${DEPS_DIR}/googletest-src" "${CMAKE_BINARY_DIR}/googletest-build" EXCLUDE_FROM_ALL)
    endif()
    if(TARGET gtest_main)
        set(MINISOLVER_GTEST_MAIN_TARGET gtest_main)
    elseif(TARGET GTest::gtest_main)
        set(MINISOLVER_GTEST_MAIN_TARGET GTest::gtest_main)
    else()
        message(FATAL_ERROR "GoogleTest main target not found")
    endif()
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
target_compile_definitions(minisolver_lib INTERFACE
    MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK=${MINISOLVER_MATRIX_STATIC_UNROLL_MAX_WORK}
    MINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N=${MINISOLVER_LDLT_FACTOR_UNROLL_OUTER_MAX_N}
    MINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N=${MINISOLVER_LDLT_FACTOR_UNROLL_ROW_MAX_N}
    MINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N=${MINISOLVER_LDLT_FACTOR_UNROLL_INNER_MAX_N}
)

if(TARGET gpu_backend)
    target_link_libraries(minisolver_lib INTERFACE gpu_backend)
endif()

# --- Examples ---
if(MINISOLVER_BUILD_EXAMPLES)
    add_subdirectory(examples/01_car_tutorial)
    add_subdirectory(examples/02_advanced_bicycle)
    add_subdirectory(examples/03_model_update_callback)
    add_subdirectory(examples/04_dcol_two_cars)
endif()

# --- Tools ---
function(add_minisolver_bench bench_name bench_source)
    add_executable(${bench_name} EXCLUDE_FROM_ALL ${bench_source})
    target_link_libraries(${bench_name} minisolver_lib ${ARGN})
endfunction()

if(MINISOLVER_BUILD_TOOLS)
    add_minisolver_bench(benchmark_suite tools/benchmark_suite/benchmark_suite.cpp)

    add_minisolver_bench(fast_inverse_bench tools/fast_inverse_bench/fast_inverse_bench.cpp)

    add_minisolver_bench(matrix_kernel_bench tools/matrix_kernel_bench/matrix_kernel_bench.cpp Eigen3::Eigen)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(matrix_kernel_bench PRIVATE
            -Wno-array-bounds
            -Wno-aggressive-loop-optimizations
            -Wno-maybe-uninitialized
        )
    endif()

    add_minisolver_bench(barrier_fusion_bench tools/barrier_fusion_bench.cpp)

    add_minisolver_bench(implicit_lu_bench tools/implicit_lu_bench.cpp)

    add_minisolver_bench(block_copy_bench tools/block_copy_bench.cpp)

    add_executable(auto_tuner tools/auto_tuner.cpp)
    target_link_libraries(auto_tuner minisolver_lib)

    add_executable(replay_solver tools/replay_solver.cpp)
    target_link_libraries(replay_solver minisolver_lib)
endif()

if(MINISOLVER_BUILD_CUDA_BENCHMARKS)
    enable_language(CUDA)
    add_executable(parallel_scan_gpu_bench EXCLUDE_FROM_ALL tools/parallel_scan_gpu_bench.cu)
    target_link_libraries(parallel_scan_gpu_bench minisolver_lib)
    set_target_properties(parallel_scan_gpu_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )

    add_executable(cuda_batched_factor_bench EXCLUDE_FROM_ALL tools/cuda_batched_factor_bench.cu)
    target_link_libraries(cuda_batched_factor_bench minisolver_lib)
    target_compile_options(cuda_batched_factor_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012>
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20014>
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015>
    )
    set_target_properties(cuda_batched_factor_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )

    add_executable(cuda_scalar_riccati_scan_bench EXCLUDE_FROM_ALL
        tools/cuda_scalar_riccati_scan_bench.cu
    )
    target_link_libraries(cuda_scalar_riccati_scan_bench minisolver_lib)
    set_target_properties(cuda_scalar_riccati_scan_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )

    add_executable(cuda_batched_scalar_riccati_bench EXCLUDE_FROM_ALL
        tools/cuda_batched_scalar_riccati_bench.cu
    )
    target_link_libraries(cuda_batched_scalar_riccati_bench minisolver_lib)
    set_target_properties(cuda_batched_scalar_riccati_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )

    add_executable(cuda_block_lft_scan_bench EXCLUDE_FROM_ALL
        tools/cuda_block_lft_scan_bench.cu
    )
    target_link_libraries(cuda_block_lft_scan_bench minisolver_lib)
    set_target_properties(cuda_block_lft_scan_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )

    add_executable(cuda_batched_lqr_riccati_bench EXCLUDE_FROM_ALL
        tools/cuda_batched_lqr_riccati_bench.cu
    )
    target_link_libraries(cuda_batched_lqr_riccati_bench minisolver_lib)
    set_target_properties(cuda_batched_lqr_riccati_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )

    add_executable(cuda_full_kkt_factor_bench EXCLUDE_FROM_ALL
        tools/cuda_full_kkt_factor_bench.cu
    )
    target_link_libraries(cuda_full_kkt_factor_bench minisolver_lib)
    set_target_properties(cuda_full_kkt_factor_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )

    add_executable(cuda_generated_packet_upload_bench EXCLUDE_FROM_ALL
        tools/cuda_generated_packet_upload_bench.cu
    )
    target_link_libraries(cuda_generated_packet_upload_bench minisolver_lib)
    target_compile_options(cuda_generated_packet_upload_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012>
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015>
    )
    set_target_properties(cuda_generated_packet_upload_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${MINISOLVER_CUDA_ARCHITECTURES}"
    )
endif()

# --- Unit Tests ---
if(MINISOLVER_BUILD_TESTS)
    enable_testing()
endif()
if(MINISOLVER_BUILD_TESTS OR MINISOLVER_BUILD_TOOLS)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
        if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
            set(Python3_FIND_VIRTUALENV FIRST)
            set(Python3_FIND_STRATEGY LOCATION)
        endif()
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(MINISOLVER_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    else()
        if(POLICY CMP0148)
            cmake_policy(SET CMP0148 OLD)
        endif()
        find_package(PythonInterp REQUIRED)
        set(MINISOLVER_PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
    endif()
endif()

if(MINISOLVER_BUILD_TESTS)
    function(add_minisolver_python_test test_name source_file)
        add_test(
            NAME ${test_name}
            COMMAND ${MINISOLVER_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
        )
    endfunction()

    add_minisolver_python_test(test_minimodel_identifiers tests/minimodel/test_identifiers.py)
    add_minisolver_python_test(test_minimodel_implicit_patterns tests/minimodel/test_implicit_patterns.py)
    add_minisolver_python_test(test_minimodel_constraints tests/minimodel/test_constraints.py)
    add_minisolver_python_test(test_minimodel_dynamics_dsl tests/minimodel/test_dynamics_dsl.py)
    add_minisolver_python_test(test_minimodel_residual_costs tests/minimodel/test_residual_costs.py)
    add_minisolver_python_test(test_minimodel_terminal tests/minimodel/test_terminal.py)
    add_minisolver_python_test(test_minimodel_model_safety tests/minimodel/test_model_safety.py)
endif()

if(MINISOLVER_BUILD_TESTS OR MINISOLVER_BUILD_TOOLS)
    set(TEST_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated_tests")
    set(KINEMATIC_BICYCLE_TEST_MODEL "${TEST_GENERATED_DIR}/kinematicbicycleregressionmodel.h")
    set(DOUBLE_INTEGRATOR_3D_TEST_MODEL "${TEST_GENERATED_DIR}/doubleintegrator3dregressionmodel.h")
    set(ASSET_REGRESSION_REFERENCE_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/asset_regression_reference_data.h")
    set(IMPLICIT_SPARSE_RICCATI_TEST_MODELS
        "${TEST_GENERATED_DIR}/fusedeulerimplicitregressionmodel.h"
        "${TEST_GENERATED_DIR}/genericeulerimplicitregressionmodel.h"
        "${TEST_GENERATED_DIR}/fusedmidpointimplicitregressionmodel.h"
        "${TEST_GENERATED_DIR}/genericmidpointimplicitregressionmodel.h"
        "${TEST_GENERATED_DIR}/fusedgaussimplicitregressionmodel.h"
        "${TEST_GENERATED_DIR}/genericgaussimplicitregressionmodel.h"
    )

    add_custom_command(
        OUTPUT ${KINEMATIC_BICYCLE_TEST_MODEL}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_GENERATED_DIR}
        COMMAND ${MINISOLVER_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_kinematic_bicycle_regression_model.py ${TEST_GENERATED_DIR}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_kinematic_bicycle_regression_model.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/MiniModel.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/templates/model.h.in
        COMMENT "Generating kinematic bicycle regression test model"
    )

    add_custom_command(
        OUTPUT ${DOUBLE_INTEGRATOR_3D_TEST_MODEL}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_GENERATED_DIR}
        COMMAND ${MINISOLVER_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_double_integrator_3d_regression_model.py ${TEST_GENERATED_DIR}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_double_integrator_3d_regression_model.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/MiniModel.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/templates/model.h.in
        COMMENT "Generating double integrator 3D regression test model"
    )

    add_custom_command(
        OUTPUT ${IMPLICIT_SPARSE_RICCATI_TEST_MODELS}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_GENERATED_DIR}
        COMMAND
            ${MINISOLVER_PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_implicit_sparse_riccati_regression_models.py
            ${TEST_GENERATED_DIR}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/generate_implicit_sparse_riccati_regression_models.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/MiniModel.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/minisolver/templates/model.h.in
        COMMENT "Generating implicit sparse Riccati regression test models"
    )

    add_custom_target(minisolver_test_models
        DEPENDS
            ${KINEMATIC_BICYCLE_TEST_MODEL}
            ${DOUBLE_INTEGRATOR_3D_TEST_MODEL}
            ${ASSET_REGRESSION_REFERENCE_HEADER}
            ${IMPLICIT_SPARSE_RICCATI_TEST_MODELS}
    )
endif()

if(MINISOLVER_BUILD_TESTS)
    function(add_minisolver_test test_name source_file)
        add_executable(${test_name} ${source_file})
        # Link GTest main
        target_link_libraries(${test_name} minisolver_lib ${MINISOLVER_GTEST_MAIN_TARGET})
        target_include_directories(${test_name} PRIVATE examples/01_car_tutorial/generated)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

# Register Tests
# --- Core ---
add_minisolver_test(test_matrix tests/test_matrix.cpp)
add_minisolver_test(test_mini_matrix tests/test_mini_matrix.cpp)
add_minisolver_test(test_memory tests/test_memory.cpp)
add_dependencies(test_memory minisolver_test_models)
target_include_directories(test_memory PRIVATE ${TEST_GENERATED_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_memory PRIVATE -Wno-mismatched-new-delete)
endif()
add_minisolver_test(test_integrator tests/test_integrator.cpp)
add_minisolver_test(test_property_regressions tests/test_property_regressions.cpp)

# --- Solver Components ---
add_minisolver_test(test_riccati tests/test_riccati.cpp)
add_minisolver_test(test_line_search tests/test_line_search.cpp)
add_minisolver_test(test_car_model_basic tests/test_car_model_basic.cpp)
add_minisolver_test(test_status tests/test_status.cpp)
add_minisolver_test(test_robustness tests/test_robustness.cpp)

# --- Solver Integration ---
add_minisolver_test(test_solver tests/test_solver.cpp)
add_minisolver_test(test_features tests/test_features.cpp)
add_minisolver_test(test_advanced tests/test_advanced.cpp)
add_minisolver_test(test_soft_constraints tests/test_soft_constraints.cpp)
add_minisolver_test(test_solver_snapshot tests/test_solver_snapshot.cpp)
add_minisolver_test(test_config_regressions tests/test_config_regressions.cpp)
add_minisolver_test(test_barrier_residual_contract tests/test_barrier_residual_contract.cpp)
add_minisolver_test(test_scaling_regressions tests/test_scaling_regressions.cpp)
add_minisolver_test(test_replay_corpus tests/test_replay_corpus.cpp)
add_dependencies(test_replay_corpus minisolver_test_models)
target_include_directories(test_replay_corpus PRIVATE ${TEST_GENERATED_DIR})
add_minisolver_test(test_logger tests/test_logger.cpp)

# --- Quality & Regression ---
add_minisolver_test(test_bugfixes tests/test_bugfixes.cpp)
add_minisolver_test(test_solver_quality tests/test_solver_quality.cpp)
add_executable(test_implicit_sparse_riccati
    tests/test_implicit_sparse_riccati.cpp
    ${IMPLICIT_SPARSE_RICCATI_TEST_MODELS}
)
add_dependencies(test_implicit_sparse_riccati minisolver_test_models)
target_link_libraries(test_implicit_sparse_riccati minisolver_lib ${MINISOLVER_GTEST_MAIN_TARGET})
target_include_directories(test_implicit_sparse_riccati PRIVATE ${TEST_GENERATED_DIR})
add_test(NAME test_implicit_sparse_riccati COMMAND test_implicit_sparse_riccati)
endif()

if(MINISOLVER_BUILD_TOOLS)
    add_minisolver_bench(implicit_sparse_riccati_bench tools/implicit_sparse_riccati_bench.cpp)
    add_dependencies(implicit_sparse_riccati_bench minisolver_test_models)
    target_include_directories(implicit_sparse_riccati_bench PRIVATE ${TEST_GENERATED_DIR})

    add_minisolver_bench(line_search_rollout_bench tools/line_search_rollout_bench.cpp)
    add_dependencies(line_search_rollout_bench minisolver_test_models)
    target_include_directories(line_search_rollout_bench PRIVATE ${TEST_GENERATED_DIR})

    add_minisolver_bench(merit_armijo_bench tools/merit_armijo_bench.cpp)

    add_minisolver_bench(warm_start_bench tools/warm_start_bench.cpp)

    add_minisolver_bench(scaling_bench tools/scaling_bench.cpp)
endif()

if(MINISOLVER_BUILD_TESTS)
add_executable(test_asset_regressions
    tests/test_asset_regressions.cpp
    ${KINEMATIC_BICYCLE_TEST_MODEL}
    ${DOUBLE_INTEGRATOR_3D_TEST_MODEL}
    ${ASSET_REGRESSION_REFERENCE_HEADER}
)
add_dependencies(test_asset_regressions minisolver_test_models)
target_link_libraries(test_asset_regressions minisolver_lib ${MINISOLVER_GTEST_MAIN_TARGET})
target_include_directories(test_asset_regressions PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference
    ${TEST_GENERATED_DIR})
add_test(NAME test_asset_regressions COMMAND test_asset_regressions)
endif()
