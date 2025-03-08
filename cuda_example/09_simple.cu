#include <stdio.h>

#define BLOCK_SIZE 6
#define BLOCK_NUM  3
__global__ void simple_kernel() {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int iX = tx + BLOCK_SIZE * bx;
    int iY = ty + BLOCK_SIZE * by;
    printf("bx: %d, by: %d, tx: %d, ty: %d\n", bx, by, tx, ty);
    // printf("iX: %d, iY: %d.\n", iX, iY);
}

int main() {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grids(2, 4);

    simple_kernel<<<grids, threads>>>();
    cudaDeviceSynchronize();
    printf("Finished!\n"); 
}