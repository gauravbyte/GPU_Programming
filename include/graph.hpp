#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

/**
 * Graph class for CSR (Compressed Sparse Row) representation
 * Used in Assignment 3 for graph processing on GPU
 */
class graph {
private:
    int num_vertices;
    int num_edges;
    int levels;
    int *offset_array;
    int *csr_array;
    int *apr_array;  // Activation Point Requirement array
    
public:
    // Constructor
    graph(const char* filename) {
        // Initialize with file reading logic
        // This is a placeholder - implement according to your input format
        num_vertices = 0;
        num_edges = 0;
        levels = 0;
        offset_array = nullptr;
        csr_array = nullptr;
        apr_array = nullptr;
    }
    
    // Destructor
    ~graph() {
        if (offset_array) delete[] offset_array;
        if (csr_array) delete[] csr_array;
        if (apr_array) delete[] apr_array;
    }
    
    // Parse graph from input file and create CSR representation
    void parseGraph() {
        // Implement graph parsing logic
        // Read adjacency list and convert to CSR format
    }
    
    // Getters
    int num_nodes() const { return num_vertices; }
    int num_edges() const { return num_edges; }
    int get_level() const { return levels; }
    
    int* get_offset() { return offset_array; }
    int* get_csr() { return csr_array; }
    int* get_aprArray() { return apr_array; }
    
    // Setters (for testing)
    void set_levels(int l) { levels = l; }
};

#endif // GRAPH_HPP
