cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GPU_Programming LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add warning flags
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

# Assignment 1: Kronecker Product
add_executable(assignment1 src/assignment1.cu)
set_target_properties(assignment1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Assignment 2: Matrix Multiplication with Tiling
add_executable(assignment2 src/assignment2.cu)
set_target_properties(assignment2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Assignment 3: Activation Game (Graph Processing)
add_executable(assignment3 src/assignment3.cu)
set_target_properties(assignment3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable testing
enable_testing()

# Add test targets
add_test(NAME assignment1_test 
    COMMAND ${CMAKE_BINARY_DIR}/bin/assignment1
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

add_test(NAME assignment2_test 
    COMMAND ${CMAKE_BINARY_DIR}/bin/assignment2 
            ${CMAKE_SOURCE_DIR}/data/assignment2_input.txt 
            ${CMAKE_SOURCE_DIR}/tests/assignment2_output.txt
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

add_test(NAME assignment3_test 
    COMMAND ${CMAKE_BINARY_DIR}/bin/assignment3 
            ${CMAKE_SOURCE_DIR}/data/graph_input.txt
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

# Print configuration
message(STATUS "")
message(STATUS "GPU Programming Project Configuration:")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Binary Output: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
