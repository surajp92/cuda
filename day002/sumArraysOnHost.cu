#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

inline void CHECK(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", file, line);
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

#define CHECK_CALL(call) CHECK((call), __FILE__, __LINE__)

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double EPSILON = 1.0E-8;
    bool match = 1;
    for (int i=0; i<N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > EPSILON) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match. \n\n");
}

void initialData (float *ip, int size) {
    // generate diferent seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost (float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N, bool printDim) {

    if (printDim) {
        printf("threadIdx: (%d %d %d) blockIdx: (%d %d %d)  blockDim: (%d %d %d) gridDim: (%d %d %d)\n", 
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

void printArray(float *op, const int N) {
    for (int i=0; i<N; i++) {
        printf("%f ", op[i]);
    }
    printf("\n");
}


int main (int argc, char **argv) {
    printf("Starting %s...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // set up data size of vectors
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data on host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int blockSize = 16; // number of threads in a block
    dim3 block(blockSize);
    dim3 grid((nElem + block.x - 1) / block.x);

    bool printDim = false;
    sumArraysOnGPU <<<grid, block >>> (d_A, d_B, d_C, nElem, printDim);
    printf("Executing configuration <<< %d, %d>>>\n", grid.x, block.x);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // printArray(h_A,nElem); 
    // printArray(h_B,nElem); 
    // printArray(hostRef,nElem); 
    
    return 0;
}