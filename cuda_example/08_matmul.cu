#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err) {
        const char *errorStr = cudaGetErrorString(err);

        fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}

#define block_size 32

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

__global__ void gemm(float *out, float *a, float *b, 
                     int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int idxM = bx * block_size + tx;
    int idxN = by * block_size + ty;

    float result = 0;
    for(int i = 0; i < K; i++) {
        int idxA = idxM * K + i;
        int idxB = N * i + idxN;
        result += a[idxA] * b[idxB];
    }
    out[idxM * N + idxN] = result;
}

int main() {
    int M = 5120, N = 4096, K = 4096;

    float *matrixA, *matrixB, *matrixC;
    float *dMatrixA, *dMatrixB, *dMatrixC;

    // Allocate host memory.
    checkCudaErrors(cudaMallocHost(&matrixA, sizeof(float) * M * K));
    checkCudaErrors(cudaMallocHost(&matrixB, sizeof(float) * K * N));
    checkCudaErrors(cudaMallocHost(&matrixC, sizeof(float) * M * N));

    // Allocate device memory.
    checkCudaErrors(cudaMalloc((void**)&dMatrixA, sizeof(float) * M * K));
    checkCudaErrors(cudaMalloc((void**)&dMatrixB, sizeof(float) * K * N));
    checkCudaErrors(cudaMalloc((void**)&dMatrixC, sizeof(float) * M * N));

    // Initialize matrixA
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            matrixA[i * K + j] = float(rand()) / RAND_MAX;
        }
    }
    // Initialize matrixB
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            matrixB[i * N + j] = float(rand()) / RAND_MAX;
      }
    }
    
    // Transfer data from host to device.
    checkCudaErrors(cudaMemcpy(dMatrixA, matrixA, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dMatrixB, matrixB, sizeof(float) * K * N, cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    dim3 threads(block_size, block_size);
    dim3 grids((M + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    checkCudaErrors(cudaEventRecord(start, 0));
    gemm<<<grids, threads>>>(dMatrixC, dMatrixA, dMatrixB, M, N, K);
    checkCudaErrors(cudaEventRecord(end, 0));
    checkCudaErrors(cudaEventSynchronize(end));

    float deviceUsedTime;
    checkCudaErrors(cudaEventElapsedTime(&deviceUsedTime, start, end));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(end));
    
    long workload = long(M) * N * K * 2;
    double gflops = (double(workload) / 1e9) / (double(deviceUsedTime) / 1e3);
    printf("GPU time: %f, performances: %f.\n", deviceUsedTime, gflops);

    // Transfer data from device to host.
    checkCudaErrors(cudaMemcpy(matrixC, dMatrixC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    // Compute the golden data.
    bool error = false;

    // Use stride for i and j to check less results to reduce the waiting time.
    for(int i = 0; i < M; i += 3) {
        for(int j = 0; j < N; j += 7) {
            float tmp = 0;
            for (int k = 0; k < K; k++) {
                tmp += matrixA[i * K + k] * matrixB[k * N + j];
            }
            if(fabs(matrixC[i * N + j] - tmp) > 1e-3) {
                printf("i: %d, j: %d, matrixC: %f, goldenC: %f.\n", i, j, matrixC[i * N + j], tmp);
                error = true;
                break;
            }
        }
        if(error)
          break;
    }

    // Free host memory.
    checkCudaErrors(cudaFreeHost(matrixA));
    checkCudaErrors(cudaFreeHost(matrixB));
    checkCudaErrors(cudaFreeHost(matrixC));

    // Free device memory.
    checkCudaErrors(cudaFree(dMatrixA));
    checkCudaErrors(cudaFree(dMatrixB));
    checkCudaErrors(cudaFree(dMatrixC));

    if(error)
        printf("Incorrect result!\n");
    else
        printf("Successful.\n");
    
    return 0;
}