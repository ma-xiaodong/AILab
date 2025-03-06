#include <iostream>
#include <math.h>

__global__ void add(int n, float *a, float *b, float *out)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < n)
        out[idx] = a[idx] + b[idx];
}

int main(void)
{
    int N = 1 << 25;
    float *a, *b, *out;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
   
    // Run kernel on 1M elements on the GPU
    int block_size = 32;
    int grid_size = (N + block_size - 1) / block_size;
    add<<<grid_size, block_size>>>(N, a, b, out);
   
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(out[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
  
    return 0;
}
