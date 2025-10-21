# Project Structure Overview

This document provides a complete overview of the restructured GPU Programming project.

## 📁 Complete Directory Tree

```
GPU_Programming/
│
├── 📄 CMakeLists.txt              # CMake build configuration
├── 📄 Makefile                    # Alternative Make build system
├── 📄 README.md                   # Main project documentation
├── 📄 SETUP.md                    # Installation and setup guide
├── 📄 .gitignore                  # Git ignore rules
│
├── 📁 src/                        # Source files
│   ├── assignment1.cu             # Kronecker Product (3 kernels)
│   ├── assignment2.cu             # Matrix Multiplication with Tiling
│   └── assignment3.cu             # Activation Game (Graph Processing)
│
├── 📁 include/                    # Header files
│   └── graph.hpp                  # Graph data structure (CSR format)
│
├── 📁 scripts/                    # Build and run automation
│   ├── build.sh                   # Build all assignments
│   ├── clean.sh                   # Clean build artifacts
│   ├── run_assignment1.sh         # Run Assignment 1
│   ├── run_assignment2.sh         # Run Assignment 2
│   ├── run_assignment3.sh         # Run Assignment 3
│   └── run_tests.sh               # Run all tests
│
├── 📁 data/                       # Test input data
│   ├── assignment1_input.txt      # Sample input for Assignment 1
│   ├── assignment2_input.txt      # Sample input for Assignment 2
│   ├── graph_input.txt            # Sample graph for Assignment 3
│   └── README.md                  # Data format documentation
│
├── 📁 tests/                      # Test outputs and configs
│   └── README.md                  # Testing documentation
│
├── 📁 .vscode/                    # VS Code configurations
│   ├── tasks.json                 # Build/run tasks
│   ├── launch.json                # Debug configurations
│   └── settings.json              # Editor settings
│
└── 📁 build/                      # Build output (generated)
    ├── bin/                       # Compiled executables
    │   ├── assignment1
    │   ├── assignment2
    │   └── assignment3
    └── [CMake files...]
```

## 🔧 Build Systems

### Option 1: CMake (Recommended)
```bash
./scripts/build.sh
# or
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

### Option 2: Make
```bash
make all
# or
make assignment1
make assignment2
make assignment3
```

## 🚀 Running Assignments

### Command Line:
```bash
# Assignment 1
cat data/assignment1_input.txt | ./build/bin/assignment1

# Assignment 2
./build/bin/assignment2 data/assignment2_input.txt output.txt

# Assignment 3
./build/bin/assignment3 data/graph_input.txt verbose
```

### Using Scripts:
```bash
./scripts/run_assignment1.sh data/assignment1_input.txt
./scripts/run_assignment2.sh data/assignment2_input.txt output.txt
./scripts/run_assignment3.sh data/graph_input.txt verbose
```

### Using VS Code:
1. `Cmd/Ctrl + Shift + P` → `Tasks: Run Task`
2. Select: `Run Assignment 1/2/3`

### Using Make:
```bash
make run1
make run2
make run3
```

## 🧪 Testing

### Run All Tests:
```bash
./scripts/run_tests.sh
# or
cd build && ctest --verbose
```

### VS Code:
`Tasks: Run Task` → `Run All Tests`

## 🐛 Debugging

### VS Code Debugging:
1. Open a `.cu` file
2. Set breakpoints (click left of line numbers)
3. Press `F5` or `Run > Start Debugging`
4. Select configuration (Assignment 1/2/3)

### CUDA-GDB:
```bash
cd build
cuda-gdb ./bin/assignment1
(gdb) run
(gdb) break kernel_function
(gdb) continue
```

### Nsight Systems Profiling:
```bash
nsys profile --stats=true ./build/bin/assignment1
```

### Nsight Compute Analysis:
```bash
ncu --set full ./build/bin/assignment1
```

## 📊 Key Features

### ✅ Professional Structure
- Separated source (`src/`), headers (`include/`), and build artifacts
- Clean project organization following C++ best practices

### ✅ Multiple Build Options
- **CMake**: Cross-platform, IDE integration
- **Makefile**: Quick compilation, traditional workflow
- **Scripts**: Automated build/run/test

### ✅ VS Code Integration
- **Tasks**: Build, run, clean with keyboard shortcuts
- **Launch configs**: Debug each assignment with F5
- **IntelliSense**: Code completion for CUDA
- **Problem matchers**: Inline error display

### ✅ Testing Framework
- CTest integration
- Individual test scripts
- Sample test data
- Automated test runner

### ✅ Documentation
- Comprehensive README
- Setup guide (SETUP.md)
- Per-directory documentation
- Inline code documentation

### ✅ Version Control
- Proper .gitignore
- Excludes build artifacts and outputs
- Includes necessary data files

## 📝 Assignment Details

### Assignment 1: Kronecker Product
- **File**: `src/assignment1.cu`
- **Kernels**: 3 (per-row, per-column, per-element)
- **Input**: Matrix dimensions and values (stdin)
- **Output**: `kernel1.txt`, `kernel2.txt`, `kernel3.txt`

### Assignment 2: Matrix Multiplication
- **File**: `src/assignment2.cu`
- **Algorithm**: Tiled matrix multiplication with shared memory
- **Formula**: E = A×B + C×D^T
- **Input**: File with matrices A, B, C, D
- **Output**: File with result matrix E

### Assignment 3: Activation Game
- **File**: `src/assignment3.cu`
- **Algorithm**: Graph activation propagation
- **Data structure**: CSR (Compressed Sparse Row)
- **Input**: Graph file with activation requirements
- **Output**: `output.txt` with active vertices per level

## 🔑 Quick Reference

### Build Commands:
```bash
./scripts/build.sh           # Build all
./scripts/clean.sh           # Clean
make all                     # Make build
make clean                   # Make clean
```

### Run Commands:
```bash
./scripts/run_assignment1.sh <input>
./scripts/run_assignment2.sh <input> <output>
./scripts/run_assignment3.sh <graph> [verbose]
```

### VS Code Shortcuts:
- `Cmd/Ctrl + Shift + B`: Default build task
- `F5`: Start debugging
- `Cmd/Ctrl + Shift + P`: Command palette

### Common Issues:
```bash
# CUDA not found
export PATH=/usr/local/cuda/bin:$PATH

# Scripts not executable
chmod +x scripts/*.sh

# Clean rebuild
./scripts/clean.sh && ./scripts/build.sh
```

## 📚 Additional Resources

- **Main README**: Project overview and usage
- **SETUP.md**: Installation and configuration
- **data/README.md**: Input data formats
- **tests/README.md**: Testing guide

## 🎯 Next Steps

1. **Install CUDA**: See SETUP.md for instructions
2. **Build**: Run `./scripts/build.sh`
3. **Test**: Run `./scripts/run_tests.sh`
4. **Develop**: Open in VS Code and use tasks
5. **Profile**: Use Nsight tools for optimization

---

**Project Status**: ✅ Fully Restructured
**Build System**: CMake + Make
**IDE Support**: VS Code (tasks, launch, debug)
**Testing**: Automated with CTest
**Documentation**: Complete
