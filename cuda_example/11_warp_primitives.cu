#include "error.cuh"
#include <stdio.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

const int N = 256;
const int M = sizeof(float) * N;

void __global__ reduce_shfl(float *device_data, int len)
{
    const unsigned FULL_MASK = 0xffffffff;
    __shared__ float s_mem[32];

    if (blockDim.x > 32)
    {
        return;
    }

    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    s_mem[threadIdx.x] = device_data[global_idx];
    __syncthreads();

    float val = s_mem[threadIdx.x];
/*
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset, 32);
    }
*/
    int offset = 10;
    val += __shfl_down_sync(FULL_MASK, val, offset, 32);
    
    device_data[global_idx] = val;
    return;
} 

int main(void)
{
    float *host_data = (float *)malloc(M);
    for (int n = 0; n < N; ++n)
    {
        if (n % 32 == 0)
        {
            printf("\n");
        }
        host_data[n] = n % 32;
        printf("%4.1f ", host_data[n]);
    }
    printf("\n--------------------------\n");

    float *device_data;
    CHECK(cudaMalloc(&device_data, M));
    CHECK(cudaMemcpy(device_data, host_data, M, cudaMemcpyHostToDevice));

    const int block_size = 32;
    const int grid_size = (N + block_size - 1) / block_size;

    reduce_shfl<<<grid_size, block_size>>>(device_data, N);

    CHECK(cudaMemcpy(host_data, device_data, M, cudaMemcpyDeviceToHost));
    for (int idx = 0; idx < N; idx++)
    {
        if (idx % 32 == 0)
        {
            printf("\n");
        }
        printf("%4.1f ", host_data[idx]);
    }
    printf("\n");

    free(host_data);
    CHECK(cudaFree(device_data));
    return 0;
}