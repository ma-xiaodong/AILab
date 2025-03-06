#include <iostream>
#include <math.h>

__global__ void add(int n, float *a, float *b, float *out)
{
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for(int i = idx; i < n; i += stride)
        out[i] = a[i] + b[i];
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
    add<<<1, 256>>>(N, a, b, out);
   
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
