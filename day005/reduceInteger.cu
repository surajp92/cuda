#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double(tp.tv_sec) + double(tp.tv_usec) / 1e6;
} 

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define FUN(a, b) ((a) + (b))

int recursiveReduce(int *A, int size){
    if (size == 1) return A[0];
    int stride = size / 2;
    for (int i = 0; i < stride; i++){
        A[i] = FUN(A[i], A[i + stride]);
    }
    return recursiveReduce(A, stride);
}


__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            g_idata[idx] = FUN(g_idata[idx], g_idata[idx + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = g_idata[idx];
    }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    int *idata = g_idata + blockIdx.x * blockDim.x; // offset to access each block's data

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        idx = 2 * stride * tid;
        if (idx < blockDim.x) {
            idata[idx] = FUN(idata[idx], idata[idx + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceNeighboredInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    int *idata = g_idata + blockIdx.x * blockDim.x; // offset to access each block's data

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] = FUN(idata[tid], idata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

int main(int argc, char **argv){

    int dev = 0;
    cudaSetDevice(dev);

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("With array size %d\n", size);

    // execution configuration
    int blockSize = 64;
    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }
    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    for (int i = 0; i < size; i++){
        h_idata[i] = (int)(rand() & 0xFF); // 0 - 255
    }
    memcpy(tmp, h_idata, bytes);

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

    // cpu reduction
    double iStart = cpuSecond();
    int cpuResult = recursiveReduce(tmp, size);
    double iElaps = cpuSecond() - iStart;
    printf("CPU reduce elapsed %f sec cpuResult: %d\n", iElaps, cpuResult);

    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum = FUN(gpu_sum, h_odata[i]);
    }
    printf("reduceNeighbored <<<%d, %d>>> elapsed %f sec result %d\n", grid.x, block.x, iElaps, gpu_sum);

    // kernel 2: reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum = FUN(gpu_sum, h_odata[i]);
    }
    printf("reduceNeighboredLess <<<%d, %d>>> elapsed %f sec result %d\n", grid.x, block.x, iElaps, gpu_sum);

    // kernel 3: reduceNeighboredInterleaved
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighboredInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum = FUN(gpu_sum, h_odata[i]);
    }
    printf("reduceNeighboredInterleaved <<<%d, %d>>> elapsed %f sec result %d\n", grid.x, block.x, iElaps, gpu_sum);


    free(h_idata);
    free(h_odata);
    free(tmp);

    cudaFree(d_idata);
    cudaFree(d_odata);

    bResult = (gpu_sum == cpuResult);
    if (!bResult) printf("Test failed\n");
    return EXIT_SUCCESS;

}
