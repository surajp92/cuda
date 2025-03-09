#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double(tp.tv_sec) + double(tp.tv_usec) / 1e6;
} 

__global__ void mathkernel1 (float *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    C[tid] = a + b;
}

__global__ void mathkernel2 (float *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    C[tid] = a + b;
}

__global__ void mathkernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        ia = 100.0f;
    }

    if (!ipred)
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathkernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)    {
        ia = 100.0f;
    } else    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

int main(int argc, char **argv) {

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    int size = 64;
    int blockSize = 64;
    if (argc > 1) blockSize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);
    printf("Data size, Block size %d %d\n", size, blockSize);

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    double iStart = cpuSecond();
    mathkernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("Execution time  mathkernel1 %f sec\n", iElaps);

    iStart = cpuSecond();
    mathkernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Execution time  mathkernel2 %f sec\n", iElaps);

    iStart = cpuSecond();
    mathkernel3<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Execution time  mathkernel3 %f sec\n", iElaps);

    iStart = cpuSecond();
    mathkernel4<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Execution time  mathkernel4 %f sec\n", iElaps);

    cudaFree(d_C);
    cudaDeviceReset();
    return EXIT_SUCCESS;

}