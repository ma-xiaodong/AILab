#include <iostream>
#include <math.h>

#define FLOAT4(a)    *(float4*)(&(a))
#define CEIL(a,b)    ((a+b-1)/(b))

__global__ void add(int n, float *a, float *b, float *out)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

    if(idx > n)
        return;

    float4 va, vb, vc;
    va = FLOAT4(a[idx]);
    vb = FLOAT4(b[idx]);

    vc.x = va.x + vb.x;
    vc.y = va.y + vb.y;
    vc.z = va.z + vb.z;
    vc.w = va.w + vb.w;

    FLOAT4(out[idx]) = vc;
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
    int block_size = 1024;
    int grid_size = CEIL(CEIL(N, 4), block_size);
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
