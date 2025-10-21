# GPU Programming - CUDA Assignments

Course assignments for GPU Programming demonstrating various CUDA programming techniques and optimizations.

## 📋 Project Structure

```
GPU_Programming/
├── CMakeLists.txt          # CMake build configuration
├── README.md               # This file
├── .gitignore             # Git ignore rules
├── src/                   # Source files
│   ├── assignment1.cu     # Kronecker Product implementation
│   ├── assignment2.cu     # Matrix Multiplication with Tiling
│   └── assignment3.cu     # Activation Game (Graph Processing)
├── include/               # Header files
│   └── graph.hpp          # Graph data structure for Assignment 3
├── scripts/               # Build and run scripts
│   ├── build.sh          # Build all assignments
│   ├── clean.sh          # Clean build artifacts
│   ├── run_assignment1.sh
│   ├── run_assignment2.sh
│   ├── run_assignment3.sh
│   └── run_tests.sh      # Run all tests
├── data/                  # Test input data
│   ├── assignment1_input.txt
│   ├── assignment2_input.txt
│   ├── graph_input.txt
│   └── README.md
├── tests/                 # Test outputs and configurations
├── build/                 # Build output directory (generated)
│   └── bin/              # Compiled executables
└── .vscode/              # VS Code configurations
    ├── tasks.json        # Build and run tasks
    ├── launch.json       # Debug configurations
    └── settings.json     # Editor settings
```

## 🚀 Quick Start

### Prerequisites

- CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- C++14 compatible compiler
- NVIDIA GPU with compute capability 7.5+

### Building the Project

```bash
# Make scripts executable (first time only)
chmod +x scripts/*.sh

# Build all assignments
./scripts/build.sh

# Or use CMake directly
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j8
```

### Running Assignments

#### Using Scripts:

```bash
# Assignment 1: Kronecker Product
cat data/assignment1_input.txt | ./build/bin/assignment1

# Assignment 2: Matrix Multiplication
./scripts/run_assignment2.sh data/assignment2_input.txt tests/output.txt

# Assignment 3: Activation Game
./scripts/run_assignment3.sh data/graph_input.txt verbose
```

#### Using VS Code:

1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Select `Tasks: Run Task`
3. Choose from:
   - `Build All` - Build all assignments
   - `Run Assignment 1/2/3` - Run specific assignment
   - `Run All Tests` - Execute test suite
   - `Clean Build` - Remove build artifacts

## 📝 Assignments Overview

### Assignment 1: Kronecker Product

Implements three different CUDA kernel strategies for computing the Kronecker product:
- **Kernel 1**: Per-row processing (1D grid, 1D block)
- **Kernel 2**: Per-column processing (1D grid, 2D block)
- **Kernel 3**: Per-element processing (2D grid, 2D block)

**Input Format:**
```
m n
<m*n elements of matrix A>
<m*n elements of matrix B>
```

**Output:** `kernel1.txt`, `kernel2.txt`, `kernel3.txt`

### Assignment 2: Matrix Multiplication with Tiling

Optimized matrix multiplication using shared memory tiling:
- Computes `E = A*B + C*D^T`
- Uses shared memory to minimize global memory access
- Tile size: 32x32

**Input Format:**
```
p q r
<Matrix A: p×q>
<Matrix B: q×r>
<Matrix C: p×q>
<Matrix D: r×q>
```

**Output:** Result matrix E (p×r)

### Assignment 3: Activation Game (Graph Processing)

GPU-accelerated graph activation algorithm:
- CSR (Compressed Sparse Row) graph representation
- Parallel activation computation across graph levels
- Atomic operations for race-free updates

**Input Format:** Graph in CSR format with activation requirements

**Output:** `output.txt` - Active vertices per level

## 🧪 Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Or use CMake's test runner
cd build
ctest --output-on-failure --verbose
```

## 🐛 Debugging

### Using CUDA-GDB:

```bash
cd build
cuda-gdb ./bin/assignment1
```

### Using VS Code:

1. Set breakpoints in your `.cu` files
2. Press `F5` or select `Run > Start Debugging`
3. Choose the assignment you want to debug

### Profiling with Nsight:

```bash
# Profile execution
nsys profile --stats=true ./build/bin/assignment1

# Detailed kernel analysis
ncu --set full ./build/bin/assignment1
```

## 📊 Performance Optimization Tips

1. **Memory Access Patterns**: Ensure coalesced memory access
2. **Shared Memory**: Use for frequently accessed data
3. **Occupancy**: Balance threads per block for optimal occupancy
4. **Atomic Operations**: Minimize usage, consider alternatives
5. **Synchronization**: Use `__syncthreads()` judiciously

## 🔧 Configuration

### Adjusting CUDA Architectures:

Edit `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Adjust for your GPU
```

### Compiler Flags:

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
```

##  Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)

##  Contributing

When adding new assignments:

1. Add source file to `src/`
2. Update `CMakeLists.txt` with new executable
3. Create run script in `scripts/`
4. Add test data to `data/`
5. Update this README

## 📄 License

Course assignment materials - check with your institution for usage rights.

## 👥 Authors

- Course: GPU Programming (CS6023)
- Institution: IIT Madras / Similar
- Academic Year: 2023

##  Troubleshooting

### Common Issues:

**CUDA not found:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**CMake can't find CUDA:**
```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

**Permission denied on scripts:**
```bash
chmod +x scripts/*.sh
```

**Build fails:**
```bash
./scripts/clean.sh
./scripts/build.sh
```

---

**Happy GPU Programming! 🚀**
