#include <stdio.h>
#include <cuda_runtime.h>

__device__ float devData;

__global__ void checkGlobalVariable() {
    printf("Device: the value of devData is %f\n", devData);

    devData += 2.0;
}

int main(void) {
    float value = 3.14;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host: copied %f to the global variable\n", value);

    checkGlobalVariable<<<1, 1>>>();

    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by kernel to %f\n", value);

    cudaDeviceReset();
    return 0;
} 

