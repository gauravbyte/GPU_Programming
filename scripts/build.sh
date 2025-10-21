#!/bin/bash

# GPU Programming Project - Build Script
# This script configures and builds all CUDA assignments

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "========================================="
echo "GPU Programming Project Build Script"
echo "========================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    echo "Please ensure CUDA toolkit is installed and in your PATH."
    exit 1
fi

echo "CUDA Compiler: $(nvcc --version | grep release)"
echo ""

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring project with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
echo ""
echo "Building project..."
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo ""
echo "Executables are in: ${BUILD_DIR}/bin/"
echo ""
echo "Available programs:"
echo "  - assignment1: Kronecker Product"
echo "  - assignment2: Matrix Multiplication with Tiling"
echo "  - assignment3: Activation Game (Graph Processing)"
echo ""
echo "Run individual assignments using scripts in scripts/ directory"
echo "or use the VS Code tasks."
