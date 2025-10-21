#!/bin/bash

# Clean build artifacts

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Cleaning build artifacts..."

# Remove build directory
if [ -d "${PROJECT_ROOT}/build" ]; then
    rm -rf "${PROJECT_ROOT}/build"
    echo "✓ Removed build/ directory"
fi

# Remove output files
find "${PROJECT_ROOT}" -name "*.txt" -not -path "*/data/*" -not -name "README.md" -delete 2>/dev/null
echo "✓ Removed output files"

# Remove compiled binaries in root (if any)
find "${PROJECT_ROOT}" -maxdepth 1 -type f -executable -delete 2>/dev/null

echo ""
echo "Clean completed!"
