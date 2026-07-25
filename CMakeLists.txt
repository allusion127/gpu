cmake_minimum_required(VERSION 3.25)

# Project
project(RASBERY LANGUAGES C CXX)

option(RASBERY_ENABLE_OPENMP "Enable OpenMP support when available." ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

add_executable(RASBERY ${RASBERY_SOURCES})
target_compile_features(RASBERY PRIVATE cxx_std_23)

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
