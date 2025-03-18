#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    unsigned int size = 1 << 22;
    unsigned int bytes = size * sizeof(int);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device: %s\n", deviceProp.name);
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

    // allocate host memory
    float *h_a = (float *)malloc(bytes);

    // allocate device memory
    float *d_a;
    cudaMalloc((float **)&d_a, bytes);

    // initialize host memory
    for (unsigned int i = 0; i < size; i++) {
        h_a[i] = 0.1f;
    }
    
    // transfer data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    // transfer data from device to host
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    
    // free memory
    cudaFree(d_a);
    free(h_a);

    cudaDeviceReset();
    return 0;
}