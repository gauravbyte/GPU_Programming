# Setup Instructions for GPU Programming Project

## System Requirements

- **CUDA Toolkit**: Version 11.0 or later
- **CMake**: Version 3.18 or later
- **C++ Compiler**: GCC 7.5+ or Clang 8+
- **GPU**: NVIDIA GPU with Compute Capability 7.5 or higher

## CUDA Installation

### macOS

1. **Download CUDA Toolkit:**
   Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   
   Select:
   - Operating System: macOS
   - Architecture: x86_64 or arm64
   - Version: Your macOS version
   - Installer Type: dmg (local)

2. **Install CUDA:**
   ```bash
   # After downloading the .dmg file
   sudo installer -pkg /Volumes/CUDAMacOSInstaller/CUDAMacOSInstaller.pkg -target /
   ```

3. **Set Environment Variables:**
   Add to your `~/.zshrc` or `~/.bash_profile`:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
   ```

4. **Verify Installation:**
   ```bash
   nvcc --version
   ```

### Linux

1. **Download and Install:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

2. **Set Environment Variables:**
   Add to `~/.bashrc`:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **Verify:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Windows

1. Download CUDA Toolkit from NVIDIA website
2. Run the installer (requires Visual Studio)
3. Add CUDA to PATH (installer usually does this)
4. Verify with: `nvcc --version`

## CMake Installation

### macOS
```bash
brew install cmake
```

### Linux
```bash
sudo apt-get install cmake
```

### Windows
Download from [cmake.org](https://cmake.org/download/)

## Building the Project

Once CUDA and CMake are installed:

```bash
cd /path/to/GPU_Programming

# Make scripts executable
chmod +x scripts/*.sh

# Build the project
./scripts/build.sh
```

## VS Code Setup

### Recommended Extensions:

1. **C/C++** (ms-vscode.cpptools)
2. **CMake Tools** (ms-vscode.cmake-tools)
3. **CUDA C++** (nvidia.nsight-vscode-edition)
4. **Task Runner** (for easy build/run)

### Installing Extensions:
```bash
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
```

## Troubleshooting

### CUDA Not Found

**Error:** `nvcc: command not found`

**Solution:**
```bash
# Find CUDA installation
which nvcc
ls -la /usr/local/cuda

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### CMake Can't Find CUDA

**Error:** `Could NOT find CUDA`

**Solution:**
```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
      ..
```

### Compute Capability Mismatch

**Error:** `Unsupported gpu architecture`

**Solution:** Edit `CMakeLists.txt` and adjust:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Change to match your GPU
```

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Permission Denied on Scripts

```bash
chmod +x scripts/*.sh
```

## GPU Information

Check your GPU details:
```bash
nvidia-smi
```

For detailed GPU info:
```bash
./build/bin/deviceQuery  # If you have CUDA samples compiled
```

Or use:
```bash
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

## Next Steps

After successful setup:

1. Build the project: `./scripts/build.sh`
2. Run tests: `./scripts/run_tests.sh`
3. Try individual assignments using VS Code tasks
4. Profile your code with Nsight Systems

## Support

If you encounter issues:

1. Check CUDA compatibility with your GPU
2. Verify CUDA toolkit version matches GPU driver
3. Ensure CMake version is 3.18 or later
4. Check compiler compatibility with CUDA version

## Useful Links

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CMake CUDA Support](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
