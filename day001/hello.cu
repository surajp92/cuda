#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("Hello, World from GPU! threadIdx: %d\n", threadIdx.x);
}

int main()
{
    printf("Hello, World from CPU!\n");
    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    // cudaDeviceSynchronize();
    return 0;
}