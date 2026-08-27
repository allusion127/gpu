cmake_minimum_required(VERSION 3.25)

# Project
project(RASBERY LANGUAGES C CXX)

option(RASBERY_ENABLE_OPENMP "Enable OpenMP support when available." ON)
option(RASBERY_ENABLE_CUDA "Enable the CUDA BiCGSTAB/CMFD backend." OFF)
set(RASBERY_CUDA_ARCHITECTURES "80;86;89;100;120" CACHE STRING
        "CUDA architectures to build the CMFD backend for (e.g. 61 for GTX 10xx).")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (RASBERY_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif ()

# Dependencies
find_package(HDF5 REQUIRED COMPONENTS C CXX)

if (RASBERY_ENABLE_OPENMP AND APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND NOT DEFINED OpenMP_ROOT)
    foreach (libomp_root IN ITEMS /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if (EXISTS "${libomp_root}")
            set(OpenMP_ROOT "${libomp_root}" CACHE PATH "OpenMP root directory" FORCE)
            break()
        endif ()
    endforeach ()
endif ()

if (RASBERY_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif ()

# Target
file(GLOB RASBERY_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if (RASBERY_ENABLE_CUDA)
    list(REMOVE_ITEM RASBERY_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/CudaBICGBackendStub.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/CudaXsReconBackendStub.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/GpuPhysicsBackendStub.cpp"
    )
    list(APPEND RASBERY_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/CudaBICGBackend.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/CudaXsReconBackend.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/GpuPhysicsArenaCuda.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/GpuPhaseScheduler.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/GpuPhysicsBackendCuda.cu"
    )
    # The xsrecon kernel's contract is bit-identical reproduction of the host
    # loop; nvcc must not contract a*b+c into fma where the host compiler did
    # not.  Scoped to this one file so the CMFD backend keeps its flags.
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/CudaXsReconBackend.cu"
            PROPERTIES COMPILE_OPTIONS "--fmad=false")
endif ()

add_executable(RASBERY ${RASBERY_SOURCES})
target_compile_features(RASBERY PRIVATE cxx_std_20)

target_include_directories(RASBERY PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/chiffon"
)
# src/ is deliberately NOT an include dir on Windows: MSVC searches /I dirs for
# angle includes too, and the case-insensitive filesystem makes plog's CRT
# `#include <io.h>` resolve to src/IO.h, poisoning every downstream header.
# Quoted includes between src files resolve includer-relative, so src/*.cpp
# never needed the dir; non-MSVC keeps it for the .cu separable-compilation
# path parity with the historical build.
if (NOT MSVC)
    target_include_directories(RASBERY PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
endif ()

if (TARGET HDF5::HDF5)
    target_link_libraries(RASBERY PRIVATE HDF5::HDF5)
else ()
    target_include_directories(RASBERY PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(RASBERY PRIVATE ${HDF5_LIBRARIES})
endif ()

if (OpenMP_CXX_FOUND)
    target_link_libraries(RASBERY PRIVATE OpenMP::OpenMP_CXX)
endif ()

if (RASBERY_ENABLE_CUDA)
    target_link_libraries(RASBERY PRIVATE CUDA::cudart CUDA::cublas)
    target_compile_definitions(RASBERY PRIVATE RASBERY_HAS_CUDA=1)
    set_target_properties(RASBERY PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED YES
            CUDA_ARCHITECTURES "${RASBERY_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION ON
    )
endif ()

# Warnings
if (MSVC)
    # CXX-only: a bare /W4 reaching nvcc is parsed as an input file
    target_compile_options(RASBERY PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else ()
    target_compile_options(RASBERY PRIVATE
            -march=native
            -Wall -Wextra -Wpedantic
            -Wno-unused-variable
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-sign-compare
    )
endif ()

# Runtime paths
target_compile_definitions(RASBERY PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Tests
#
# The xsrecon consistency harness scores the shared host/device node function
# (XsReconKernel.h) against a verbatim quotation of the CPU loop, on synthetic
# arrays.  It includes nothing but that header, so it builds anywhere -- no
# HDF5, no CUDA.  -march=native mirrors the RASBERY release flags so the host
# compiler makes the same contraction choices it makes in the real build.
option(RASBERY_ENABLE_TESTS "Build the standalone consistency harnesses." OFF)

if (RASBERY_ENABLE_TESTS)
    enable_testing()

    add_executable(rasbery_xsrecon_consistency
            "${CMAKE_CURRENT_SOURCE_DIR}/test/xsrecon_consistency_test.cpp")
    target_compile_features(rasbery_xsrecon_consistency PRIVATE cxx_std_20)
    target_include_directories(rasbery_xsrecon_consistency PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if (NOT MSVC)
        target_compile_options(rasbery_xsrecon_consistency PRIVATE -march=native)
    endif ()

    add_test(NAME xsrecon_consistency COMMAND rasbery_xsrecon_consistency 4096)

    # Arena layout gate (Rev.7.1 Task 2 Step 1).  The layout calculator is pure
    # host arithmetic, so alignment, non-overlap, slot isolation and the
    # per-slot-per-phase scratch aliasing rule are all checkable with no device
    # -- which is the only place they get checked at all.  Includes
    # GpuPhysicsArenaLayout.h and nothing else, so it builds everywhere.
    add_executable(rasbery_gpu_arena_layout
            "${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_physics_arena_layout.cpp")
    target_compile_features(rasbery_gpu_arena_layout PRIVATE cxx_std_20)
    target_include_directories(rasbery_gpu_arena_layout PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")

    add_test(NAME gpu_arena_layout COMMAND rasbery_gpu_arena_layout)

    # Case-phase compaction gate (Rev.7.1 Task 3).  gpuClassifySerial is pure,
    # so the queue ordering, the padding, the bucket choice, both Sec 5.2 fatal
    # faults and the Sec 8.2 recycled-slot reset are all checkable exhaustively
    # without a device -- and that is the only place they get checked.
    add_executable(rasbery_gpu_phase_compaction
            "${CMAKE_CURRENT_SOURCE_DIR}/test/gpu_phase_compaction.cpp")
    target_compile_features(rasbery_gpu_phase_compaction PRIVATE cxx_std_20)
    target_include_directories(rasbery_gpu_phase_compaction PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")

    add_test(NAME gpu_phase_compaction COMMAND rasbery_gpu_phase_compaction)

    # Device arm of the same gate: nvcc's floating point vs g++'s, elementwise
    # on one call.  This is what the full-deck A/B cannot localize.
    if (RASBERY_ENABLE_CUDA)
        add_executable(rasbery_xsrecon_device_consistency
                "${CMAKE_CURRENT_SOURCE_DIR}/test/xsrecon_device_consistency_test.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/CudaXsReconBackend.cu")
        target_compile_features(rasbery_xsrecon_device_consistency PRIVATE cxx_std_20)
        target_include_directories(rasbery_xsrecon_device_consistency PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(rasbery_xsrecon_device_consistency PRIVATE CUDA::cudart)
        set_target_properties(rasbery_xsrecon_device_consistency PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED YES
                CUDA_ARCHITECTURES "${RASBERY_CUDA_ARCHITECTURES}"
                CUDA_SEPARABLE_COMPILATION ON)
        if (NOT MSVC)
            target_compile_options(rasbery_xsrecon_device_consistency PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        endif ()
        add_test(NAME xsrecon_device_consistency
                COMMAND rasbery_xsrecon_device_consistency 4096)
    endif ()

    # Flat-XS replay gate + contraction miner: scores the shared body of
    # FlatXsKernel.h against a RASBERY_FLATXS_DUMP production capture.  Needs
    # a capture on disk, so it is not an add_test -- run it by hand (see the
    # header comment in test/flatxs_replay.cpp).
    add_executable(rasbery_flatxs_replay
            "${CMAKE_CURRENT_SOURCE_DIR}/test/flatxs_replay.cpp")
    target_compile_features(rasbery_flatxs_replay PRIVATE cxx_std_20)
    target_include_directories(rasbery_flatxs_replay PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if (NOT MSVC)
        target_compile_options(rasbery_flatxs_replay PRIVATE -march=native -O3)
    endif ()
    if (OpenMP_CXX_FOUND)
        target_link_libraries(rasbery_flatxs_replay PRIVATE OpenMP::OpenMP_CXX)
    endif ()

    # Nodal replay gate + per-phase contraction miner (host-only; needs a
    # RASBERY_NODAL_DUMP capture on disk).
    add_executable(rasbery_nodal_replay
            "${CMAKE_CURRENT_SOURCE_DIR}/test/nodal_replay.cpp")
    target_compile_features(rasbery_nodal_replay PRIVATE cxx_std_20)
    target_include_directories(rasbery_nodal_replay PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if (NOT MSVC)
        target_compile_options(rasbery_nodal_replay PRIVATE -march=native -O3)
    endif ()

    # Device arm of the same gate: the CUDA build of the shared body against
    # the same capture, elementwise.  Compiled with --fmad=false exactly like
    # the production TU.
    if (RASBERY_ENABLE_CUDA)
        add_executable(rasbery_flatxs_device_replay
                "${CMAKE_CURRENT_SOURCE_DIR}/test/flatxs_device_replay.cu")
        target_compile_features(rasbery_flatxs_device_replay PRIVATE cxx_std_20)
        target_include_directories(rasbery_flatxs_device_replay PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(rasbery_flatxs_device_replay PRIVATE CUDA::cudart)
        set_target_properties(rasbery_flatxs_device_replay PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED YES
                CUDA_ARCHITECTURES "${RASBERY_CUDA_ARCHITECTURES}"
                CUDA_SEPARABLE_COMPILATION ON)
        set_source_files_properties(
                "${CMAKE_CURRENT_SOURCE_DIR}/test/flatxs_device_replay.cu"
                PROPERTIES COMPILE_OPTIONS "--fmad=false")
    endif ()

    if (RASBERY_ENABLE_CUDA)
        add_executable(rasbery_nodal_device_replay
                "${CMAKE_CURRENT_SOURCE_DIR}/test/nodal_device_replay.cu")
        target_compile_features(rasbery_nodal_device_replay PRIVATE cxx_std_20)
        target_include_directories(rasbery_nodal_device_replay PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(rasbery_nodal_device_replay PRIVATE CUDA::cudart)
        set_target_properties(rasbery_nodal_device_replay PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED YES
                CUDA_ARCHITECTURES "${RASBERY_CUDA_ARCHITECTURES}"
                CUDA_SEPARABLE_COMPILATION ON)
        set_source_files_properties(
                "${CMAKE_CURRENT_SOURCE_DIR}/test/nodal_device_replay.cu"
                PROPERTIES COMPILE_OPTIONS "--fmad=false")

        add_executable(rasbery_nodal_mine_device
                "${CMAKE_CURRENT_SOURCE_DIR}/test/nodal_mine_device.cu")
        target_compile_features(rasbery_nodal_mine_device PRIVATE cxx_std_20)
        target_include_directories(rasbery_nodal_mine_device PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(rasbery_nodal_mine_device PRIVATE CUDA::cudart)
        set_target_properties(rasbery_nodal_mine_device PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED YES
                CUDA_ARCHITECTURES "${RASBERY_CUDA_ARCHITECTURES}"
                CUDA_SEPARABLE_COMPILATION ON)
        set_source_files_properties(
                "${CMAKE_CURRENT_SOURCE_DIR}/test/nodal_mine_device.cu"
                PROPERTIES COMPILE_OPTIONS "--fmad=false")
    endif ()
endif ()
