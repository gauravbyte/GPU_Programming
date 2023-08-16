/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
#include <limits.h>
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

//This is a CUDA kernel function that calculates the active in-degree of vertices in a
// The function takes in several parameters:
// - v1: the starting vertex ID to calculate in-degree for
// - v2: the ending vertex ID to calculate in-degree for
// - level: unused parameter
// - d_offset: a pointer to the offset array of the CSR graph representation
// - d_csrList: a pointer to the CSR list array of the CSR graph representation
// - d_apr: a pointer to an array that stores the current in-degree of each vertex
// - d_aid: a pointer to an array that will store the updated in-degree of each vertex after the calculati

__global__ void calculate_indegree(int v1,int v2, int *d_offset,int *d_csrList,int *d_apr,int *d_aid){
    // Calculate the thread ID based on the block ID and thread ID
    int threadId = blockIdx.x*blockDim.x + threadIdx.x;
    // Calculate the vertex ID to process based on the thread ID
    int vid = v1 + threadId;
    // Check if the vertex ID is within the range to process
    if(vid<v2){
        // Check if the active in-degree of the vertex is greater than or equal activation point requirement 
        if(d_aid[vid]>=d_apr[vid]){ 
            // Loop through the neighbors of the vertex and increment their active in-degree
            for(int i=d_offset[vid]; i<d_offset[vid+1]; i++){
                atomicAdd(&(d_aid[d_csrList[i]]),1);
            }
        }
    }
}
  
// This function is executed on the GPU and is responsible for calculating the number of active based on the second rule of the algorithm.
// The function takes in the starting and ending vertices (vs and ve), the current level (l), the array of active node IDs (d_aid), the array of active node parents (d_apr), and the array of active vertices (d_activeVertex).
__global__ void calculate_active_nodes_2nd_rule(int vs,int ve,int l,int *d_aid,int *d_apr, int *d_activeVertex){
        // Calculate the thread ID based on the block ID and thread ID.
        int threadId = blockIdx.x*1024 + threadIdx.x;
        // Calculate the vertex ID based on the starting vertex and thread ID.
        int vid = vs+threadId;
        int vl,vr;
        // Check if the current vertex is active.
        bool is_active_vid = (d_aid[vid]>=d_apr[vid]?1:0);
        // If the current vertex is active and not at the beginning or end of the range of vertices...
        if(is_active_vid && vid >vs && vid <ve-1){
            // Calculate the IDs of the vertices to the left and right of the current vertex.
            vl = vid-1;
            vr = vid+1;

            // Check if the vertices to the left and right of the current vertex are active.
            bool is_active_vl = (d_aid[vl]>=d_apr[vl]?1:0);
            bool is_active_vr = (d_aid[vr]>=d_apr[vr]?1:0);
            // If either of the vertices to the left or right of the current vertex are active...
            if(is_active_vl || is_active_vr){
                // Increment the number of active vertices at the current level.
                atomicAdd(&d_activeVertex[l],1);                
            }
            // If neither of the vertices to the left or right of the current vertex are active...
            else {
              // Mark the current vertex as inactive.
              d_aid[vid] = -1;
            }
        }

        // If the current vertex is active and at the beginning or end of the range of vertices...
        else if(is_active_vid && (vid == vs || vid == ve-1)){

            // Increment the number of active vertices at the current level.
            atomicAdd(&d_activeVertex[l],1);
        }
}

/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
    
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // active in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));
    cudaMemset(d_aid,0,V*sizeof(int));
    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
  cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemset(d_activeVertex, 0, L*sizeof(int));

/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

// Initialize minimum to the maximum integer value
int minimum = INT_MAX;

// Initialize the level array with 0 at index 0
int level[L+1];
level[0] = 0;

// Initialize l to 1
int l=1;

// Loop through all vertices
for(int i=0; i<V; i++){
    // Update minimum with the minimum value between the current vertex's offset and the current minimum
    minimum = min(h_csrList[h_offset[i]],minimum);

    // If the minimum value is less than or equal to the current vertex index and l is less than L
    if(minimum <= i && l<L){
        // Set the current level to the current vertex index
        level[l] = i;
        // Increment l
        l++;
        // Reset minimum to the maximum integer value
        minimum = INT_MAX;
    }
}

// Set the last index of the level array to the total number of vertices
level[L] = V;

// Loop through all levels except the last one
for(int i=0 ; i<L-1; i++){
    // Set v1 to the current level's starting vertex index and v2 to the next level's starting vertex index
    int v1 = level[i];
    int v2 = level[i+1];

    // Calculate the number of nodes in the current level and the number of blocks needed to process them
    int nodes_in_cur_level = v2-v1;
    int n_blocks = ceil(nodes_in_cur_level/1024.0);

    // Call the calculate_indegree kernel to calculate the active -indegree of each vertex in the current level
    calculate_indegree<<<n_blocks,min(1024, nodes_in_cur_level)>>>(  v1,  v2,    d_offset,   d_csrList,   d_apr,   d_aid);
    cudaDeviceSynchronize();

    // Set vs to the next level's starting vertex index and ve to the next level's ending vertex index
    int vs = level[i+1];
    int ve = level[i+2];

    // Calculate the number of nodes in the next level and the number of blocks needed to process them
    int nodes_in_next_level = ve-vs;
    int nn_blocks = ceil(nodes_in_next_level*(1.0)/1024);

    // Call the calculate_active_nodes_2nd_rule kernel to calculate the active nodes in the next level based on the second rule
    calculate_active_nodes_2nd_rule<<<nn_blocks,min(1024, nodes_in_next_level)>>>(  vs,  ve,  i+1,  d_aid,  d_apr,   d_activeVertex);
    cudaDeviceSynchronize();
}

// Copy the activeVertex array from device to host
cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);

// Set the first index of the activeVertex array to the number of vertices in the first level
h_activeVertex[0] = level[1]-level[0];
/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
    if(argc>2)
    {
        for(int i=0; i<L; i++)
        {
            printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
        }
    }

    return 0;
}
