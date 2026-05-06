# CMake toolchain file for ARM Cortex-M4 (with FPU).
#
# Usage:
#   cmake -S . -B build_arm \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4.cmake \
#         -DMINISOLVER_EMBEDDED_PROFILE=ON
#
# Requires arm-none-eabi-gcc on PATH (apt install gcc-arm-none-eabi).

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_AR arm-none-eabi-ar)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_SIZE arm-none-eabi-size)

# Cortex-M4F flags. -mthumb selects Thumb-2 ISA, -mcpu picks the core, the
# FPU flags enable hard-float ABI for the on-die single-precision FPU.
set(_minisolver_arm_cortex_m4_flags
    "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -ffunction-sections -fdata-sections")

set(CMAKE_C_FLAGS_INIT "${_minisolver_arm_cortex_m4_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_minisolver_arm_cortex_m4_flags} -fno-rtti -fno-exceptions")
set(CMAKE_ASM_FLAGS_INIT "${_minisolver_arm_cortex_m4_flags}")

# Strip dead code at link time and avoid host libc surprises.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--gc-sections --specs=nosys.specs")

# When cross-compiling we never want the host find_program/find_path to
# return host binaries from the toolchain prefix.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# arm-none-eabi-gcc cannot run the executables it produces, so disable the
# compiler test that tries to do so.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
