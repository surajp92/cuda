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


void transposeOnHost(int *in, int *out, const int nx, const int ny) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            out[j * ny + i] = in[i * nx + j];
        }
    }
}

__global__ void transposeOnGPURow(int *in, int *out, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] =  in[iy * nx + ix];
    }
}

__global__ void transposeOnGPUCol(int *in, int *out, int nx, int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

__global__ void transposeOnGPURowUnroll4(int *in, int *out, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int tid = iy * nx + ix;
    int tod = ix * ny + iy;

    if (ix + blockDim.x * 3 < nx && iy < ny) {
        out[tod] =  in[tid];
        out[tod + ny * blockDim.x] =  in[tid + blockDim.x];
        out[tod + ny * blockDim.x * 2] =  in[tid + blockDim.x * 2];
        out[tod + ny * blockDim.x * 3] =  in[tid + blockDim.x * 3];
    }
}

__global__ void transposeOnGPUColUnroll4(int *in, int *out, int nx, int ny) {
    unsigned int ix = blockDim.x * blockIdx.x * 3 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    int tid = ix * ny + iy;
    int tod = iy * nx + ix;

    if (ix + blockDim.x * 3 < nx && iy < ny) {
        out[tod] = in[tid];
        out[tod + blockDim.x] = in[tid + ny * blockDim.x];
        out[tod + blockDim.x * 2] = in[tid + ny * blockDim.x * 2];
        out[tod + blockDim.x * 3] = in[tid + ny * blockDim.x * 3];
    }

}

int main() {
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    int *h_in;
    int *h_out;
    int *gpuRef;
    h_in = (int *)malloc(nBytes);
    h_out = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);
    initialize(h_in, nxy);
    initialize(h_out, nxy);

    double istart = cpuSecond();
    transposeOnHost(h_in, h_out, nx, ny);
    double iElaps = cpuSecond() - istart;
    printf("transposeOnHost elapsed %f sec\n", iElaps);

    int *d_in;
    int *d_out;
    cudaMalloc((int **)&d_in, nBytes);
    cudaMalloc((int **)&d_out, nBytes);

    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    int blocksizex = 16;
    int blocksizey = 16;

    dim3 block(blocksizex, blocksizey);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    istart = cpuSecond();
    transposeOnGPURow<<<grid, block>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("warmup <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    istart = cpuSecond();
    transposeOnGPURow<<<grid, block>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("transposeOnGPURow <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    istart = cpuSecond();
    transposeOnGPUCol<<<grid, block>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("transposeOnGPUCol <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    
    grid.x = (nx + 4 * block.x - 1) / (4 * block.x);

    istart = cpuSecond();
    transposeOnGPURowUnroll4<<<grid, block>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("transposeOnGPURowUnroll4 <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    istart = cpuSecond();
    transposeOnGPUColUnroll4<<<grid, block>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("transposeOnGPUColUnroll4 <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost);
    checkResult(h_out, gpuRef, nx, ny);

    // printMatrix(h_in, nx, ny);
    // printMatrix(h_out, ny, nx);
    // printMatrix(gpuRef, ny, nx);
    
    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_out);
    free(gpuRef);

    cudaDeviceReset();

    return 0;
}