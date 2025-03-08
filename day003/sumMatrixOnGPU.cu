#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double(tp.tv_sec) + double(tp.tv_usec) / 1e6;
} 


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

void checkResult(int *hostRef, int *gpuRef, const int nx, const int ny) {
    double epsilon = 1.0E-8;
    bool match = 1;
    int nxny = nx * ny;
    for (int i = 0; i < nxny; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("host %d gpu %d\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match) {
        printf("Arrays match.\n\n");
    } else {
        printf("Arrays do not match.\n\n");
    }
}


void sumMatrixOnHost(int *A, int *B, int *C, const int nx, const int ny) {
    int nxy = nx * ny;
    for (int i = 0; i < nxy; i++) {
        C[i] = A[i] + B[i];
    }
    // for (int i=0; i<ny; i++){
    //     for (int j=0; j<nx; j++){
    //         int idx = i * nx + j;
    //         C[idx] = A[idx] + B[idx];
    //     }
    // }
}

__global__ void sumMatrixOnGPU(int *A, int *B, int*C, const int nx, const int ny) {
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumMatrixOnGPU1D(int *A, int *B, int *C, const int nx, const int ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < nx) {
        for (int iy = 0; iy < ny; iy++) {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__ void sumMatrixOnGPUMix(int *A, int *B, int*C, const int nx, const int ny) {
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y;
    int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}



int main() {
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    int *h_A;
    int *h_B;
    int *hostRef;
    int *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);
    initialize(h_A, nxy);
    initialize(h_B, nxy);

    double istart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    double iElaps = cpuSecond() - istart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);
    
    int *d_A;
    int *d_B;
    int *d_C;
    cudaMalloc((int **)&d_A, nBytes);
    cudaMalloc((int **)&d_B, nBytes);
    cudaMalloc((int **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int blocksizex = 16;
    int blocksizey = 16;

    dim3 block(blocksizex, blocksizey);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    istart = cpuSecond();
    sumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    iElaps = cpuSecond() - istart;
    printf("sumMatrixOnGPU <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nx, ny);
    cudaDeviceSynchronize();

    int blocksize = 32;
    dim3 block1D(blocksize);
    dim3 grid1D((nx + block1D.x - 1) / block1D.x);

    istart = cpuSecond();
    sumMatrixOnGPU1D<<<grid1D, block1D>>>(d_A, d_B, d_C, nx, ny);
    iElaps = cpuSecond() - istart;
    printf("sumMatrixOnGPU1D <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid1D.x, grid1D.y, block1D.x, block1D.y, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nx, ny);
    cudaDeviceSynchronize();

    dim3 blockMix(blocksize);
    dim3 gridMix((nx + block1D.x - 1) / block1D.x, ny);

    istart = cpuSecond();
    sumMatrixOnGPUMix<<<gridMix, blockMix>>>(d_A, d_B, d_C, nx, ny);
    iElaps = cpuSecond() - istart;
    printf("sumMatrixOnGPUMix <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", gridMix.x, gridMix.y, blockMix.x, blockMix.y, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nx, ny);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();

    return 0;
}