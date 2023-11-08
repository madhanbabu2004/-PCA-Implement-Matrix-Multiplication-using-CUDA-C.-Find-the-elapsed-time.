# PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time.
 Implement Matrix Multiplication using GPU.

## Aim:
To implement Matrix Multiplication using GPU.

## Procedure:
Step 1 :
Include the required files and library.

Step 2 :
Declare the block size and the size of elements .

Step 3 :
Introduce Kernel function to perform matrix multiplication.In the kernal function,decalre the row column size and initialize the sum to be 0,then using for loop calculate the sum.

Step 4 :
Intoduce a Main function, in the main method declare the required variables and Initialize the matrices 'a' and 'b'.Allocate memory on the device and then copy the input matrices from host to device memory and set the grid and block sizes . Launch the kernel,Copy the result matrix from device to host memory ,Print the result matrix and the elapsed time followed by freeing the device memory.

Step 5 :
Save the program and execute it .
# PROGRAM:
Developed By: MADHAN BABU P
Register no.: 212222230075


#include <stdio.h>
#include <sys/time.h>
#define SIZE 4
#define BLOCK_SIZE 2

// Kernel function to perform matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < size; ++k)
    {
        sum += a[row * size + k] * b[k * size + col];
    }

    c[row * size + col] = sum;
}

int main()
{
    int a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE];
    int *dev_a, *dev_b, *dev_c;
    int size = SIZE * SIZE * sizeof(int);

    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("Result Matrix:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
## Output:
root@MidPC:/home/student/Desktop# nvcc first.cu
root@MidPC:/home/student/Desktop# ./a.out
Result Matrix:
14 8 2 -4 
20 10 0 -10 
26 12 -2 -16 
32 14 -4 -22 
Elapsed Time: 0.000023 seconds
root@MidPC:/home/student/Desktop# nvprof ./a.out
==18221== NVPROF is profiling process 18221, command: ./a.out
Result Matrix:
14 8 2 -4 
20 10 0 -10 
26 12 -2 -16 
32 14 -4 -22 
Elapsed Time: 0.000037 seconds
==18221== Profiling application: ./a.out
==18221== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.90%  2.5280us         1  2.5280us  2.5280us  2.5280us  matrixMultiply(int*, int*, int*, int)
                   38.89%  2.4640us         2  1.2320us     928ns  1.5360us  [CUDA memcpy HtoD]
                   21.21%  1.3440us         1  1.3440us  1.3440us  1.3440us  [CUDA memcpy DtoH]
      API calls:   99.38%  126.78ms         3  42.262ms  2.2600us  126.78ms  cudaMalloc
                    0.28%  356.84us         1  356.84us  356.84us  356.84us  cuDeviceTotalMem
                    0.20%  252.08us        97  2.5980us     210ns  107.52us  cuDeviceGetAttribute
                    0.07%  87.360us         3  29.120us  2.5700us  79.360us  cudaFree
                    0.03%  36.470us         1  36.470us  36.470us  36.470us  cuDeviceGetName
                    0.02%  29.180us         3  9.7260us  5.9900us  12.080us  cudaMemcpy
                    0.02%  23.180us         1  23.180us  23.180us  23.180us  cudaLaunchKernel
                    0.00%  4.5900us         1  4.5900us  4.5900us  4.5900us  cuDeviceGetPCIBusId
                    0.00%  2.4000us         3     800ns     250ns  1.8100us  cuDeviceGetCount
                    0.00%     930ns         2     465ns     210ns     720ns  cuDeviceGet
                    0.00%     310ns         1     310ns     310ns     310ns  cuDeviceGetUuid



## Result:
![PCA 1](https://github.com/MURALI22008445/-PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time./assets/119643767/becb0217-e800-4a79-a6b8-e7fe08b28889)

