#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost (float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

void printArray(float *op, const int N) {
    for (int i=0; i<N; i++) {
        printf("%f ", op[i]);
    }
    printf("\n");
}

void initialData (float *ip, int size) {
    // generate diferent seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main (int argc, char **argv) {
    // float *ptr;
    // ptr = (float *)malloc(nBytes);
    // *ptr = 1.0;
    // float **addr = &ptr;
    // printf("%f\n", *ptr);
    // printf("%p\n", ptr);
    // printf("%p\n", &ptr);
    // printf("%p\n", *ptr);
    // printf("%p\n", *addr);
    // printf("%p\n", &addr);

    int nElem = 8;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // 
    float *d_A, *d_B, *d_C;

    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    // sumArraysOnHost(h_A, h_B, h_C, nElem);

    // printArray(h_A,nElem); 
    // printArray(h_B,nElem); 
    // printArray(h_C,nElem); 

    // free(h_A);
    // free(h_B);
    // free(h_C);

    return 0;

}