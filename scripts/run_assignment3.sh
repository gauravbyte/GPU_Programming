#!/bin/bash

# Run Assignment 3: Activation Game (Graph Processing)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTABLE="${PROJECT_ROOT}/build/bin/assignment3"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: assignment3 executable not found!"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

echo "========================================="
echo "Running Assignment 3: Activation Game"
echo "========================================="
echo ""

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <graph_input_file> [verbose]"
    echo ""
    echo "The graph input file should contain:"
    echo "  - Graph structure in CSR format"
    echo "  - Activation point requirements"
    echo ""
    echo "Add 'verbose' as second argument to see detailed output"
    exit 1
fi

GRAPH_FILE="$1"
VERBOSE_FLAG=""

if [ ! -f "$GRAPH_FILE" ]; then
    echo "Error: Graph file '$GRAPH_FILE' not found!"
    exit 1
fi

if [ $# -eq 2 ] && [ "$2" == "verbose" ]; then
    VERBOSE_FLAG="1"
fi

if [ -z "$VERBOSE_FLAG" ]; then
    "$EXECUTABLE" "$GRAPH_FILE"
else
    "$EXECUTABLE" "$GRAPH_FILE" "1"
fi

echo ""
echo "Output written to: output.txt"
