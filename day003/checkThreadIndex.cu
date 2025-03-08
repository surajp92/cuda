#include <cuda_runtime.h>
#include <stdio.h>

void initialize (int *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny) {
    printf("\nMatrix: (%3d.%3d)\n", nx, ny);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%3d ", C[i * nx + j]);
        }
        printf("\n");
    }
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;

    printf("thread_id (%d %d) block_id (%d %d) coordinate (%d %d) global index %2d ival %2d\n",
    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}


__global__ void checkThreadIndex() {

    printf("threadIdx: (%d %d %d) blockIdx: (%d %d %d)  blockDim: (%d %d %d) gridDim: (%d %d %d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}

int main() {
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    int *h_A;
    h_A = (int *)malloc(nBytes);
    initialize(h_A, nxy);
    printMatrix(h_A, nx, ny);

    int *d_A;
    cudaMalloc((int **)&d_A, nBytes);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    int blocksizex = 4;
    int blocksizey = 2;

    dim3 block(blocksizex, blocksizey);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // checkThreadIndex <<<grid, block>>> ();
    printThreadIndex <<<grid, block>>> (d_A, nx, ny);
    cudaDeviceSynchronize();

    cudaFree(d_A);

    free(h_A);

    cudaDeviceReset();

    return 0;
}