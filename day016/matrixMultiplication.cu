#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double(tp.tv_sec) + double(tp.tv_usec) / 1e6;
}


void printMatrix(float *h_A, int B, int H, int M, int N) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < M; k++) {
                for (int l = 0; l < N; l++) {
                    printf("%f ", h_A[i * H * M * N + j * M * N + k * N + l]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void initializeMatric(float *A, int B, int H, int M, int N) {
    int size = B * H * M * N;
    for (int i = 0; i < size; i++) {
        A[i] = float(rand()) / float(RAND_MAX);
    }
}

void check(float *h_C, float *h_C_gpu, int B, int H, int M, int N) {
    double epsilon = 1.0E-3;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < M; k++) {
                for (int l = 0; l < N; l++) {
                    if (abs(h_C[i * H * M * N + j * M * N + k * N + l] - h_C_gpu[i * H * M * N + j * M * N + k * N + l]) > epsilon) {
                        printf("Error at (%d, %d, %d, %d)\n", i, j, k, l);
                        printf("h_C: %f, h_C_gpu: %f, diff: %f\n", h_C[i * H * M * N + j * M * N + k * N + l], h_C_gpu[i * H * M * N + j * M * N + k * N + l], h_C[i * H * M * N + j * M * N + k * N + l] - h_C_gpu[i * H * M * N + j * M * N + k * N + l]);
                        return;
                    }
                }
            }
        }
    }
    printf("Success\n");

}

void cpuMultiplication(float *h_A, float *h_B, float *h_C, int B, int H, int M,  int K,  int N) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < H; j++) {
            for ( int k = 0; k < M; k++) {
                for ( int l = 0; l < N; l++) {
                    float sum = 0.0;
                    for ( int m = 0; m < K; m++) {
                        sum += h_A[i * H * M * K + j * M * K + k * K + m] * h_B[i * H * K * N + j * K * N + m * N + l];
                    }
                    h_C[i * H * M * N + j * M * N + k * N + l] = sum;
                }
            }
        }
    }
}


__global__ void batchedMatrixMultiplication(float *d_A, float *d_B, float *d_C, int B, int H, int M, int K, int N) {
    // Compute batch and height indices from blockIdx.z
    int batch = blockIdx.z / H;  // Batch index
    int height = blockIdx.z % H; // Height index

    // Compute row and column indices
    int row = threadIdx.y + blockIdx.y * blockDim.y; // Row index in M
    int col = threadIdx.x + blockIdx.x * blockDim.x; // Column index in N

    if (row < M && col < N) {
        float sum = 0.0;

        // Perform dot product for the current element
        for (int k = 0; k < K; k++) {
            sum += d_A[batch * H * M * K + height * M * K + row * K + k] *
                   d_B[batch * H * K * N + height * K * N + k * N + col];
        }

        // Store the result in d_C
        d_C[batch * H * M * N + height * M * N + row * N + col] = sum;
    }
}


int main() {
    int B = 10;
    int H = 10;
    int M = 512;
    int K = 256;
    int N = 512;

    float *h_A, *h_B, *h_C, *h_C_gpu;

    h_A = (float *)malloc(B * H * M * K * sizeof(float));
    h_B = (float *)malloc(B * H * K * N * sizeof(float));
    h_C = (float *)malloc(B * H * M * N * sizeof(float));
    h_C_gpu = (float *)malloc(B * H * M * N * sizeof(float));

    initializeMatric(h_A, B, H, M, K);
    initializeMatric(h_B, B, H, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, B * H * M * K * sizeof(float));
    cudaMalloc((void **)&d_B, B * H * K * N * sizeof(float));
    cudaMalloc((void **)&d_C, B * H * M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, B * H * M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B * H * K * N * sizeof(float), cudaMemcpyHostToDevice);

    double iStart = cpuSecond();
    cpuMultiplication(h_A, h_B, h_C, B, H, M, K, N);
    double iElaps = cpuSecond() - iStart;
    printf("CPU elapsed %f sec\n", iElaps);

    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y, B * H);
    printf("block.x: %d, block.y: %d, block.z: %d\n", block.x, block.y, block.z);
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    iStart = cpuSecond();
    batchedMatrixMultiplication<<<grid, block>>>(d_A, d_B, d_C, B, H, M, K, N);
    iElaps = cpuSecond() - iStart;
    printf("GPU elapsed  on <<<%d, %d>>> %f sec\n", grid.x, block.x, iElaps);

    cudaMemcpy(h_C_gpu, d_C, B * H * M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // printMatrix(h_A, B, H, M, K);
    // printf("\n");
    // printMatrix(h_B, B, H, K, N);
    // printf("\n");
    // printMatrix(h_C, B, H, M, N);
    // printf("\n");
    // printMatrix(h_C_gpu, B, H, M, N);
    
    check(h_C, h_C_gpu, B, H, M, N);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}