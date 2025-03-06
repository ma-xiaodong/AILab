#include <stdio.h>
#include <sys/time.h>

#define M (1 << 10)
#define N (1 << 12)
#define K (1 << 10)

__global__ void vector_add(float *out, float *a, float *b) {
    int idxN = threadIdx.x;
    int idxM = blockIdx.x;
    float result = 0;

    for(int i = 0; i < K; i++) {
        result += a[idxM * K + i] * b[N * i + idxN];
    }

    out[idxM * N + idxN] = result;
}

int main() {
    float *matrixA, *matrixB, *matrixC;
    float *goldenC;
    float *dMatrixA, *dMatrixB, *dMatrixC;

    // Allocate host memory.
    matrixA = (float *)malloc(sizeof(float) * M * K);
    matrixB = (float *)malloc(sizeof(float) * K * N);
    matrixC = (float *)malloc(sizeof(float) * M * N);
    goldenC = (float *)malloc(sizeof(float) * M * N);

    // Allocate device memory.
    cudaMalloc((void**)&dMatrixA, sizeof(float) * M * K);
    cudaMalloc((void**)&dMatrixB, sizeof(float) * K * N);
    cudaMalloc((void**)&dMatrixC, sizeof(float) * M * N);

    // Initialize matrixA
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
          if(j % 3 == 0)
              matrixA[i * K + i] = 0;
          else
              matrixA[i * K + i] = 1.2;
        }
    }
    // Initialize matrixB
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            if(j % 5 == 0)
                matrixB[i * N + i] = 0;
            else
                matrixB[i * N + i] = 2.1;
      }
    }
    
    // Transfer data from host to device.
    cudaMemcpy(dMatrixA, matrixA, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dMatrixB, matrixB, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // int block_size = 256;
    // int grid_size = (N + block_size) / block_size;
    vector_add<<<M, N>>>(dMatrixC, dMatrixA, dMatrixB);
    // cudaDeviceSynchronize();

    // Transfer data from device to host.
    cudaMemcpy(matrixC, dMatrixC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Compute the golden data.
    struct timeval tv_begin, tv_end;
    float timeUsed;
    gettimeofday(&tv_begin, NULL);
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            goldenC[i * N + j]= 0;
            for (int k = 0; k < K; k++) {
                goldenC[i * N + j] += matrixA[i * K + k] * matrixB[k * N + j];
            }
        }
    }
    gettimeofday(&tv_end, NULL);
    timeUsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
    printf("CPU time: %.2fms.\n", timeUsed);

    // Check the result.
    bool error = false;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            if(matrixC[i * N + j] != goldenC[i * N + j]) {
                printf("i: %d, j: %d, matrixC: %f, goldenC: %f.\n", i, j, matrixC[i * N + j], goldenC[i * N + j]);
                error = true;
                break;
            }
        }
    }

    // Free host memory.
    free(matrixA);
    free(matrixB);
    free(matrixC);

    // Free device memory.
    cudaFree(dMatrixA);
    cudaFree(dMatrixB);
    cudaFree(dMatrixC);

    if(error)
        printf("Incorrect result!\n");
    else
        printf("Successful.\n");
    
    return 0;
}