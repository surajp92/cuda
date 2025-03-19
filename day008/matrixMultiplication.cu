#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double(tp.tv_sec) + double(tp.tv_usec) / 1e6;
}


void printMatrix(float *h_A, unsigned int M, unsigned int N) {
    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            printf("%f ", h_A[i * N + j]);
        }
        printf("\n");
    }
}

void check(float *h_C, float *h_C_gpu, unsigned int M, unsigned int N) {
    double epsilon = 1.0E-8;

    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            if (abs(h_C[i * N + j] - h_C_gpu[i * N + j]) > epsilon) {
                printf("Error at (%d, %d)\n", i, j);
                printf("h_C: %f, h_C_gpu: %f, diff: %f\n", h_C[i * N + j], h_C_gpu[i * N + j], h_C[i * N + j] - h_C_gpu[i * N + j]);
                return;
            }
        }
    }
    printf("Success\n");

}

void cpuMultiplication(float *h_A, float *h_B, float *h_C,  int M,  int K,  int N) {
    for ( int i = 0; i < M; i++) {
        for ( int j = 0; j < N; j++) {
            float sum = 0.0;
            for ( int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C[i * N + j] = sum;
        }
    }
}


__global__ void gpuMultiplication(float *d_A, float *d_B, float *d_C,  int M,  int K,  int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float sum = 0.0;
        for ( int k = 0; k < K; k++) {
            sum += d_A[i * K + k] * d_B[k * N + j];
        }
        d_C[i * N + j] = sum;
    }
}


int main() {
    int M = 1024;
    int K = 1024;
    int N = 1024;

    float *h_A, *h_B, *h_C, *h_C_gpu;

    h_A = (float *)malloc(M * K * sizeof(float));
    h_B = (float *)malloc(K * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));
    h_C_gpu = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) {
        // h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_A[i] = int(rand() & 0xFF);

    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = int(rand() & 0xFF);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    double iStart = cpuSecond();
    cpuMultiplication(h_A, h_B, h_C, M, K, N);
    double iElaps = cpuSecond() - iStart;
    printf("CPU elapsed %f sec\n", iElaps);

    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    printf("block.x: %d, block.y: %d\n", block.x, block.y);
    printf("grid.x: %d, grid.y: %d\n", grid.x, grid.y);

    iStart = cpuSecond();
    gpuMultiplication<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    iElaps = cpuSecond() - iStart;
    printf("GPU elapsed  on <<<%d, %d>>> %f sec\n", grid.x, block.x, iElaps);

    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // printMatrix(h_A, M, K);
    // printf("\n");
    // printMatrix(h_B, K, N);
    // printf("\n");
    // printMatrix(h_C, M, N);
    // printf("\n");
    // printMatrix(h_C_gpu, M, N);
    
    check(h_C, h_C_gpu, M, N);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}