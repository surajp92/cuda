#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double(tp.tv_sec) + double(tp.tv_usec) / 1e6;
} 

void initialdata(float *A, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++) {
        A[i] = (float)(rand()) / RAND_MAX;
    }
}

void printMatrix(float *C, const int ny, const int nx) {
    printf("\nMatrix: (%3d.%3d)\n", ny, nx);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%3.2f ", C[i * nx + j]);
        }
        printf("\n");
    }
}

void checkResult(float *hostRef, float *gpuRef, int nx, int ny) {
    double epsilon = 1.0E-6;
    bool match = 1;
    long int nxny = nx * ny;
    for (long int i = 0; i < nxny; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match) {
        printf("Arrays match.\n\n");
    } else {
        printf("Arrays do not match.\n\n");
    }
}

void layerNorm(float *input, float *output, float *gamma, float *beta, int m, int n) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0;
        float mean, var;
        for (int j = 0; j < n; j++) {
            sum += input[i * n + j];
        }
        mean = sum / n;
        sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += (input[i * n + j] - mean) * (input[i * n + j] - mean);
        }
        var = sum / n;
        float stddev = sqrtf(var + 1e-7);
        for (int j = 0; j < n; j++) {
            output[i * n + j] = gamma[j] * (input[i * n + j] - mean) / stddev + beta[j];
        }
    }
}

__global__ void layerNormDevice(float *input, float *output, float *gamma, float *beta, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float sum = 0.0;
        float mean, var;
        for (int j = 0; j < n; j++) {
            sum += input[i * n + j];
        }
        mean = sum / n;
        sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += (input[i * n + j] - mean) * (input[i * n + j] - mean);
        }
        var = sum / n;
        float stddev = sqrtf(var + 1e-7);
        for (int j = 0; j < n; j++) {
            output[i * n + j] = gamma[j] * (input[i * n + j] - mean) / stddev + beta[j];
        }
    }
}


int main() {
    int m = 1 << 10;
    int n = 1 << 10;
    int nElem = m * n;

    float *h_input;
    float *h_output;
    float *h_gamma;
    float *h_beta;
    float *gpuRef;

    h_input = (float *)malloc(nElem * sizeof(float));
    h_output = (float *)malloc(nElem * sizeof(float));
    h_gamma = (float *)malloc(n * sizeof(float));
    h_beta = (float *)malloc(n * sizeof(float));
    gpuRef = (float *)malloc(nElem * sizeof(float));

    initialdata(h_input, nElem);

    for (int i = 0; i < n; i++) {
        h_gamma[i] = 1.0;
        h_beta[i] = 0.0;
    }

    float *d_input;
    float *d_output;
    float *d_gamma;
    float *d_beta;

    cudaMalloc((float **)&d_input, nElem * sizeof(float));
    cudaMalloc((float **)&d_output, nElem * sizeof(float));
    cudaMalloc((float **)&d_gamma, n * sizeof(float));
    cudaMalloc((float **)&d_beta, n * sizeof(float));

    cudaMemcpy(d_input, h_input, nElem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, n * sizeof(float), cudaMemcpyHostToDevice);
    
    double iStart = cpuSecond();
    layerNorm(h_input, h_output, h_gamma, h_beta, m, n);
    double iElaps = cpuSecond() - iStart;
    printf("layerNorm on host elapsed %f sec\n", iElaps);

    int blockSize = 128;
    dim3 block(blockSize);
    dim3 grid((m + block.x - 1) / block.x);

    printf("grid.x: %d, grid.y: %d\n", grid.x, grid.y);
    printf("block.x: %d, block.y: %d\n", block.x, block.y);

    iStart = cpuSecond();
    layerNormDevice<<<grid, block>>>(d_input, d_output, d_gamma, d_beta, m, n);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("layerNorm on device elapsed %f sec\n", iElaps);

    cudaMemcpy(gpuRef, d_output, nElem * sizeof(float), cudaMemcpyDeviceToHost);

    checkResult(h_output, gpuRef, m, n);

    // printMatrix(h_input, m, n);
    // printMatrix(h_output, m, n);
    // printMatrix(gpuRef, m, n);

    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);
    free(gpuRef);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}
