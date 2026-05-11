# Toolchain file for SG200X (Sophgo CV181X/CV182X/SG2002) cross-compilation
#
# Usage:
#   1. Set environment variable: export SG200X_SDK_PATH=/path/to/reCamera-OS/output/sg2002_recamera_emmc
#   2. Configure: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -B build
#   3. Build: cmake --build build -j$(nproc)

# Check required environment variable
if(DEFINED ENV{SG200X_SDK_PATH})
    set(SG200X_SDK_PATH $ENV{SG200X_SDK_PATH})
elseif(DEFINED CACHE{SG200X_SDK_PATH})
    set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES SG200X_SDK_PATH)
else()
    message(FATAL_ERROR
        "SG200X_SDK_PATH environment variable not set.\n"
        "Please set it to the SDK path, for example:\n"
        "  export SG200X_SDK_PATH=/path/to/reCamera-OS/output/sg2002_recamera_emmc\n"
    )
endif()

message(STATUS "SG200X SDK Path: ${SG200X_SDK_PATH}")

# =============================================================================
# Target System Configuration
# =============================================================================
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# =============================================================================
# Toolchain Configuration
# =============================================================================
# Assume toolchain is in reCamera-OS/host-tools
get_filename_component(RECAMERA_OS_ROOT "${SG200X_SDK_PATH}/../.." ABSOLUTE)
set(TOOLCHAIN_ROOT "${RECAMERA_OS_ROOT}/host-tools/gcc/riscv64-linux-musl-x86_64")

# Set compilers
set(CMAKE_C_COMPILER   "${TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-musl-gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-musl-g++")

# Verify toolchain exists
if(NOT EXISTS "${CMAKE_C_COMPILER}")
    message(FATAL_ERROR
        "RISC-V toolchain not found at: ${TOOLCHAIN_ROOT}\n"
        "Expected structure: reCamera-OS/host-tools/gcc/riscv64-linux-musl-x86_64/\n"
    )
endif()

message(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")

# =============================================================================
# RISC-V CPU-specific Flags (C906FD core)
# =============================================================================
set(RISCV_FLAGS "-mcpu=c906fdv -march=rv64gcv0p7_zfh_xthead -mabi=lp64d")
set(CMAKE_C_FLAGS_INIT "${RISCV_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${RISCV_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${RISCV_FLAGS}")
set(CMAKE_ASM_FLAGS "${RISCV_FLAGS}" CACHE STRING "RISC-V ASM flags" FORCE)

# Release optimization (required for TPU performance)
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE_INIT "-flto")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE_INIT "-flto")

# Ensure musl dynamic linker path matches SG200X rootfs (/lib/ld-musl-riscv64.so.1)
# libgcc_s is not available in the musl toolchain, so link libgcc statically.
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--dynamic-linker=/lib/ld-musl-riscv64.so.1 -static-libgcc"
    CACHE STRING "SG200X musl dynamic linker" FORCE)

# Default to Release unless explicitly specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

# =============================================================================
# Sophgo TPU SDK Paths
# =============================================================================
set(TPU_SDK_ROOT "${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk")
set(TPU_INCLUDE_DIR "${TPU_SDK_ROOT}/include")
set(TPU_LIB_DIR "${TPU_SDK_ROOT}/lib")

# CVI MPI (Video Processing) paths
set(CVI_MPI_INCLUDE_DIR "${SG200X_SDK_PATH}/cvi_mpi/include")
set(CVI_MPI_LIB_DIR "${SG200X_SDK_PATH}/cvi_mpi/lib")

# CVI RTSP paths
set(CVI_RTSP_INCLUDE_DIR "${SG200X_SDK_PATH}/cvi_rtsp/install/include")
set(CVI_RTSP_LIB_DIR "${SG200X_SDK_PATH}/cvi_rtsp/install/lib")

# ISP include paths (use cv181x for SG2002/CV181X/CV182X)
set(CVI_ISP_INCLUDE_DIR "${SG200X_SDK_PATH}/cvi_mpi/include/isp/cv181x")

message(STATUS "TPU SDK include: ${TPU_INCLUDE_DIR}")
message(STATUS "TPU SDK lib: ${TPU_LIB_DIR}")
message(STATUS "CVI MPI include: ${CVI_MPI_INCLUDE_DIR}")
message(STATUS "CVI MPI lib: ${CVI_MPI_LIB_DIR}")
message(STATUS "CVI RTSP include: ${CVI_RTSP_INCLUDE_DIR}")
message(STATUS "CVI RTSP lib: ${CVI_RTSP_LIB_DIR}")

# =============================================================================
# Search Paths Configuration
# =============================================================================
# Only search in target environment, not host
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Set root paths for library search
set(CMAKE_FIND_ROOT_PATH
    "${SG200X_SDK_PATH}"
    "${TPU_SDK_ROOT}"
)

# =============================================================================
# Package Hints (help CMake find target packages)
# =============================================================================
# OpenCV - Try multiple possible locations
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${TPU_SDK_ROOT}/lib/cmake" "${TPU_SDK_ROOT}/share")
set(OpenCV_DIR "${TPU_SDK_ROOT}/lib/cmake/opencv4" CACHE PATH "OpenCV config path")

# Workaround for OpenCV SDK packaging issue:
# The OpenCVConfig.cmake references a non-existent include/opencv4 directory
# We fix the include paths in CMakeLists.txt, but we can't suppress the warning
# from OpenCV's own config file without modifying it
# Note: This is a known issue with the SG200X SDK's OpenCV packaging

# Disable ONNX Runtime (use CVI Runtime instead)
set(USE_ONNX_RUNTIME OFF CACHE BOOL "Disable ONNX Runtime on RISC-V")

# Enable Sophgo TPU support
add_definitions(-DUSE_CVI_TPU)
