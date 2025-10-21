#!/bin/bash

# Run all tests

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found!"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

cd "$BUILD_DIR"

echo "========================================="
echo "Running All Tests"
echo "========================================="
echo ""

ctest --output-on-failure --verbose

echo ""
echo "========================================="
echo "Test Results Summary"
echo "========================================="
