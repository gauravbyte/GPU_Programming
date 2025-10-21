# GPU Programming Test Data

This directory contains test input files for each assignment.

## Files:

- `assignment1_input.txt`: Sample input for Kronecker Product (Assignment 1)
- `assignment2_input.txt`: Sample input for Matrix Multiplication (Assignment 2)
- `graph_input.txt`: Sample graph input for Activation Game (Assignment 3)

## Adding Your Own Test Data:

1. Create new input files following the format specified in each assignment
2. Update the run scripts in `scripts/` to point to your test files
3. Update `CMakeLists.txt` test configurations if needed

## Input Formats:

### Assignment 1:
```
m n
<m*n values for matrix A>
<m*n values for matrix B>
```

### Assignment 2:
```
p q r
<p*q values for matrix A>
<q*r values for matrix B>
<p*q values for matrix C>
<r*q values for matrix D>
```

### Assignment 3:
```
<Graph structure in CSR format>
<Activation point requirements>
```
(Exact format depends on your graph.hpp implementation)
