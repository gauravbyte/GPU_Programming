#!/bin/bash

# Run Assignment 2: Matrix Multiplication with Tiling

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTABLE="${PROJECT_ROOT}/build/bin/assignment2"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: assignment2 executable not found!"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

echo "========================================="
echo "Running Assignment 2: Matrix Multiplication"
echo "========================================="
echo ""

# Check if input and output files are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    echo ""
    echo "Input format:"
    echo "  Line 1: p q r (dimensions)"
    echo "  Next p*q values: Matrix A"
    echo "  Next q*r values: Matrix B"
    echo "  Next p*q values: Matrix C"
    echo "  Next r*q values: Matrix D"
    echo ""
    echo "Computes: E = A*B + C*D^T"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

"$EXECUTABLE" "$INPUT_FILE" "$OUTPUT_FILE"

echo ""
echo "Result written to: $OUTPUT_FILE"
