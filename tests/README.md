# Test Configuration

This directory contains test outputs and configurations.

## Running Tests

Tests can be executed in several ways:

### 1. Using Test Script:
```bash
cd /path/to/GPU_Programming
./scripts/run_tests.sh
```

### 2. Using CMake CTest:
```bash
cd build
ctest --output-on-failure --verbose
```

### 3. Using VS Code:
- Open Command Palette (`Cmd+Shift+P`)
- Select `Tasks: Run Task`
- Choose `Run All Tests`

## Test Cases

### Assignment 1 Test
- **Input**: stdin (from test data)
- **Expected**: Generates kernel1.txt, kernel2.txt, kernel3.txt
- **Validation**: Check matrix dimensions and values

### Assignment 2 Test
- **Input**: `data/assignment2_input.txt`
- **Output**: `tests/assignment2_output.txt`
- **Expected**: Correct matrix E = A*B + C*D^T

### Assignment 3 Test
- **Input**: `data/graph_input.txt`
- **Output**: `output.txt`
- **Expected**: Correct active vertex counts per level

## Adding New Tests

1. Create input files in `data/`
2. Add expected output files here
3. Update `CMakeLists.txt`:
   ```cmake
   add_test(NAME my_test 
       COMMAND ${CMAKE_BINARY_DIR}/bin/assignment1
       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
   )
   ```

## Test Results

Results from the last test run will be stored here.
