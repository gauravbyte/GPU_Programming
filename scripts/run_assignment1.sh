#!/bin/bash

# Run Assignment 1: Kronecker Product

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTABLE="${PROJECT_ROOT}/build/bin/assignment1"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: assignment1 executable not found!"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

echo "========================================="
echo "Running Assignment 1: Kronecker Product"
echo "========================================="
echo ""

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file>"
    echo ""
    echo "Input format (via stdin or file):"
    echo "  Line 1: m n (dimensions)"
    echo "  Lines 2-(m*n+1): Matrix A elements"
    echo "  Lines (m*n+2)-end: Matrix B elements"
    echo ""
    echo "Example:"
    echo "  echo '2 3 1 2 3 4 5 6 7 8 9 10 11 12' | $EXECUTABLE"
    exit 1
fi

if [ -f "$1" ]; then
    cat "$1" | "$EXECUTABLE"
else
    echo "Error: Input file '$1' not found!"
    exit 1
fi

echo ""
echo "Output files generated:"
echo "  - kernel1.txt"
echo "  - kernel2.txt"
echo "  - kernel3.txt"
