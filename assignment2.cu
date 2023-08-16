#include <iostream>
#include <sys/time.h>
#include <cuda.h>
using namespace std;
#define TILE_SIZE 32

__global__ void Multiply_AB(int *A, int *B, int *E, int p, int q, int r)
{
	// Shared memory for tiles of A and B
	__shared__ int sA[TILE_SIZE][TILE_SIZE];
	__shared__ int sB[TILE_SIZE][TILE_SIZE];

	// get the row and column of E matrix
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	//tile row and tile column depends thread block dim
	int Tile_row = threadIdx.y;
	int Tile_col = threadIdx.x;
	// Initialize total to zero
	int sum = 0;

	// Loop over tiles of A and B
	for (int i = 0; i < q+1; i++)
	{
		// import A tile in shared memory and assign the empty elements to be zero
		if (row < p && i * TILE_SIZE + Tile_col < q)
		{
			sA[Tile_row][Tile_col] = A[row * q + i * TILE_SIZE + Tile_col];
			//get the qth row and ith tile in that get element whose index is Tile_col

		}
		else
		{
			sA[Tile_row][Tile_col] = 0;
		}

		// import B tile in shared memory and assign the empty elements to be zero
		if (col < r && i * TILE_SIZE + Tile_row < q)
		{
			sB[Tile_row][Tile_col] = B[(i * TILE_SIZE + Tile_row) * r + col];
		}
		else
		{
			sB[Tile_row][Tile_col] = 0;
		}

		//barrier to make sure all threads wait till all data loaded in tiles 
		__syncthreads();

		// multiply tiles of A and B
		for (int j = 0; j < TILE_SIZE; j++)
		{
			
			sum += sA[Tile_row][j] * sB[j][Tile_col];
		}

		// barrier to make all multiplication in block is calculated
		__syncthreads();
	}

	// Write result element to global memory
	if (row < p && col < r)
	{
		// printf("%d\n",sum);
		E[row * r + col] = sum;
	}
}

__global__ void Multiply_CDt(int *C, int *D, int *E, int p, int q, int r)
{
	__shared__ int sC[1024];
	int row = blockIdx.x;
	int col = threadIdx.x;
	//loading C matrix in Shared memory to reduce memory latency
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < q; i++)
		{
			sC[i] = C[row * q + i];
		}
	}
	//waiting for all threadds till C stored in shared memory
	__syncthreads();
	if (row < p && col <= r)
	{
		int sum = 0;
		for (int k = 0; k < q; k++)
		{
			// sum += C[row * q + k] * D[col * q + k];
			sum += sC[k] * D[col * q + k];
			//D matrix is already Coalesced cause stored in Column wise order
		}
		E[row * r + col] += sum;
	}
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB,
			 int *h_matrixC, int *h_matrixD, int *h_matrixE)
{
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;

	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */

	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid(r / TILE_SIZE + 1, p / TILE_SIZE + 1);
	Multiply_AB<<<dimGrid, dimBlock>>>(d_matrixA, d_matrixB, d_matrixE, p, q, r);
	Multiply_CDt<<<p, r>>>(d_matrixC, d_matrixD, d_matrixE, p, q, r);

	/* ****************************************************************** */
	// matrixAddition<<<gridDim, blockDim>>>(d_matrixC, d_matrixD, d_matrixE, p, q);
	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fscanf(inputFilePtr, "%d", &matrix[i * cols + j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fprintf(outputFilePtr, "%d ", matrix[i * cols + j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv)
{
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;

	inputFilePtr = fopen(inputFileName, "r");
	if (inputFilePtr == NULL)
	{
		printf("Failed to open the input file.!!\n");
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int *)malloc(p * q * sizeof(int));
	matrixB = (int *)malloc(q * r * sizeof(int));
	matrixC = (int *)malloc(p * q * sizeof(int));
	matrixD = (int *)malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int *)malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000 * seconds + microSeconds / 1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
