#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    int aindx,bindx;
    //divide threadid from 0 to m
    //give blockid from 0 to m
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int cindx = bid*m*n*n+tid;
    //cindx is unique and decidesd by
    for(int i=0; i<n; i++){ 
        aindx = bid*n +i;
        cindx = bid*m*n*n + tid + i*m;
        for(int j=0; j<n; j++){
          bindx = tid*n + j;
          C[cindx] = A[aindx] * B[bindx];
          cindx+=m*n;             
        }
    }
}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    int tid = threadIdx.x + blockDim.x*threadIdx.y;
    int tno = blockIdx.x * 1024 + tid;
    if(tno<n*n){
      int acol = tno%(n);
      int bcol = tno/(n);
      int aindx,bindx,cindx;
      cindx = tno*m;
      aindx = acol;
      //total m*m column wise multiplication 
      for(int i=0; i<m; i++){
        bindx = bcol;
        cindx = tno*m + i*m*n*n;
        for(int j=0; j<m; j++){
          C[cindx] = A[aindx] * B[bindx];
          cindx+=1;
          bindx+=n;
        }
        aindx+=n;      
      }
    }
}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    /* row number 
and column number 
row 1 column 1
a indx = (row/n)*n + (column/m)
B_indx = (row%n)    +  (column % m)*n
 */
    int bid = gridDim.x*blockIdx.y + blockIdx.x;
    int tid = blockDim.x*threadIdx.y + threadIdx.x;
    int tno = bid*1024+tid;
    if (tno < m*n*m*n){   
      //thread numbers will setup each c indx value uniquely
      int cindx = tno;
      int row = tno/(m*n);
      int col = tno%(m*n);
      int aindx = (row/n)*n + (col/m);
      int bindx = (row%n) + (col%m)*n;
      C[tno] = A[aindx] * B[bindx];
    }

    // printf("cindx %d = %d \n",tno,A[aindx]*B[bindx]);
}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
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
    long int m,n;	
    cin>>m>>n;	
 
    // Host_arrays 
    long int *h_a,*h_b,*h_c;

    // Device arrays 
    long int *d_a,*d_b,*d_c;
	
    // Allocating space for the host_arrays 
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    // Allocating memory for the device arrays 
    // --> Allocate memory for A on device 
    cudaMalloc(&d_a, m * n * sizeof(long int));
    // --> Allocate memory for B on device 
    cudaMalloc(&d_b, m * n * sizeof(long int));
    // --> Allocate memory for C on device 
    cudaMalloc(&d_c, m*m*n*n*sizeof(long int));

    // Read the input matrix A 

    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    //Read the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // h_a[]={4,-3,1,2,6,-8};
    // h_b[]={-5,0,4,6,9,-2};
    // Transfer the input host arrays to the device 
    // --> Copy A from Host to Device
    cudaMemcpy(d_a,h_a, n* m * sizeof(long int),cudaMemcpyHostToDevice);
    // --> Copy B from Host to Device 
    cudaMemcpy(d_b,h_b, n * m * sizeof(long int),cudaMemcpyHostToDevice);

    long int gridDimx, gridDimy;
    // Launch the kernels
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
     **/
    
    // --> Set the launch configuration 
    
    double starttime = rtclock();  
    per_row_AB_kernel<<<m,m>>>(d_a,d_b,d_c,m,n);
    // --> Launch the kernel 
    
    cudaDeviceSynchronize();                                                           

    double endtime = rtclock(); 
	printtime("GPU Kernel-1 time: ", starttime, endtime);  

    // --> Copy C from Device to Host 
    cudaMemcpy(h_c,d_c,m*m*n*n*sizeof(long int),cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/
    
    // --> Set the launch configuration 
  
    dim3 block2(64,16);
    int block_count = ceil(float(n*n)/1024);
    starttime = rtclock(); 
    
    // --> Launch the kernel 
    per_column_AB_kernel<<<block_count,dim3(64,16)>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();


    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  
 
    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,m*m*n*n*sizeof(long int),cudaMemcpyDeviceToHost);   
    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/
    gridDimx = ceil(float(n * n) / 16);
    gridDimy = ceil(float(m * m) / 64);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);
    
    starttime = rtclock();  

    // --> Launch the kernel
    per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();                                                              

    endtime = rtclock();  
	printtime("GPU Kernel-3 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,m*m*n*n*sizeof(long int),cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel3.txt");

    return 0;
}