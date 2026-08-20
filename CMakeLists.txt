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
    )
    list(APPEND RASBERY_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/CudaBICGBackend.cu"
    )
endif ()

add_executable(RASBERY ${RASBERY_SOURCES})
target_compile_features(RASBERY PRIVATE cxx_std_20)

target_include_directories(RASBERY PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/chiffon"
)

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
    target_compile_options(RASBERY PRIVATE /W4)
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
