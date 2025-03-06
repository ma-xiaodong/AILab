#include <stdio.h>

#define N (1 << 28)

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
        return;

    out[idx] = a[idx] + b[idx];
}

int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate host memory.
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Allocate device memory.
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    for(int i = 0; i < N; i++) {
        a[i] = 1;
        b[i] = 2;
    }
    
    // Transfer data from host to device.
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size) / block_size;
    vector_add<<<grid_size, block_size>>>(d_out, d_a, d_b, N);
    //cudaDeviceSynchronize();

    // Transfer data from device to host.
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // check the results
    bool error = false;
    for(int i = 0; i < N; i++) {
        if(out[i] != 3) {
            error = true;
            break;
        }

    }

    // Free host and device memory.
    free(a);
    free(b);
    free(out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    if(error)
        printf("Incorrect result!\n");
    else
        printf("Successful.\n");
    
    return 0;
}