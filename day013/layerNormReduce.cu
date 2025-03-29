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
            printf("%3.4f ", C[i * nx + j]);
        }
        printf("\n");
    }
}

void checkResult(float *hostRef, float *gpuRef, int nx, int ny) {
    double epsilon = 1.0E-4;
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
        // printf("mean: %f\n", mean);
        sum = 0.0;
        for (int j = 0; j < n; j++) {
            // output[i * n + j] = (input[i * n + j] - mean) * (input[i * n + j] - mean);
            sum += (input[i * n + j] - mean) * (input[i * n + j] - mean);
        }
        var = sum / n;
        // printf("var: %f\n", var);

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

__global__ void computeMeanAndVarianceLayerNorm(const float *input, float *output, const float *gamma, const float *beta, float *mean, float *var, int M, int N) {
    // Shared memory for partial sums
    extern __shared__ float sharedData[];

    // Get the row index this block is responsible for
    int row = blockIdx.x;

    // Get the thread index within the block
    int tid = threadIdx.x;

    // Initialize shared memory
    sharedData[tid] = 0.0f;

    // Each thread sums a subset of the row
    for (int col = tid; col < N; col += blockDim.x) {
        sharedData[tid] += input[row * N + col];
    }

    // Synchronize threads to ensure all partial sums are written to shared memory
    __syncthreads();

    // Perform reduction to compute the total sum for the row
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // The first thread in the block computes the mean
    float rowMean = 0.0f;
    if (tid == 0) {
        rowMean = sharedData[0] / N;
        mean[row] = sharedData[0] / N;
    }

    // Broadcast the mean to all threads in the block
    __syncthreads();

    // Step 2: Compute the variance
    sharedData[tid] = 0.0f;

    // Each thread computes the squared difference for its subset of elements
    for (int col = tid; col < N; col += blockDim.x) {
        float diff = input[row * N + col] - mean[row];
        sharedData[tid] += diff * diff;
    }

    // Synchronize threads to ensure all partial squared differences are written to shared memory
    __syncthreads();

    // Perform reduction to compute the total squared difference for the row
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // The first thread in the block computes the variance
    float rowVariance = 0.0f;
    if (tid == 0) {
        rowVariance = sharedData[0] / N;
        var[row] = rowVariance;
    }

    // Broadcast the variance to all threads in the block
    __syncthreads();

    // Step 3: Compute the layer normalization
    float rowStddev = sqrtf(var[row] + 1e-7f); // Add epsilon for numerical stability
    for (int col = tid; col < N; col += blockDim.x) {
        float normalized = (input[row * N + col] - mean[row]) / rowStddev;
        output[row * N + col] = gamma[col] * normalized + beta[col];
    }
}

int main() {
    int m = 1 << 10;
    int n = 1 << 16;
    int blockSize_x = 256;
    int nElem = m * n;

    float *h_input;
    float *h_output;
    float *h_gamma;
    float *h_beta;
    float *gpuRef;
    float *mean;
    float *var;

    h_input = (float *)malloc(nElem * sizeof(float));
    h_output = (float *)malloc(nElem * sizeof(float));
    h_gamma = (float *)malloc(n * sizeof(float));
    h_beta = (float *)malloc(n * sizeof(float));
    gpuRef = (float *)malloc(nElem * sizeof(float));
    mean = (float *)malloc(m * sizeof(float));
    var = (float *)malloc(m * sizeof(float));

    initialdata(h_input, nElem);

    for (int i = 0; i < n; i++) {
        h_gamma[i] = 1.0;
        h_beta[i] = 0.0;
    }

    for (int i = 0; i < m; i++) {
        mean[i] = 0.0;
        var[i] = 0.0;
    }

    float *d_input;
    float *d_output;
    float *d_gamma;
    float *d_beta;
    float *d_mean;
    float *d_var;

    cudaMalloc((float **)&d_input, nElem * sizeof(float));
    cudaMalloc((float **)&d_output, nElem * sizeof(float));
    cudaMalloc((float **)&d_gamma, n * sizeof(float));
    cudaMalloc((float **)&d_beta, n * sizeof(float));
    cudaMalloc((float **)&d_mean, m * sizeof(float));
    cudaMalloc((float **)&d_var, m * sizeof(float));

    cudaMemcpy(d_input, h_input, nElem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var, var, m * sizeof(float), cudaMemcpyHostToDevice);
    
    double iStart = cpuSecond();
    layerNorm(h_input, h_output, h_gamma, h_beta, m, n);
    double iElaps = cpuSecond() - iStart;
    printf("layerNorm on host elapsed %f sec\n", iElaps);

    dim3 grid(m);       // One block per row
    dim3 block(blockSize_x);    // Number of threads per block
    size_t sharedMemSize = block.x * sizeof(float);

    iStart = cpuSecond();
    computeMeanAndVarianceLayerNorm<<<grid, block, sharedMemSize>>>(d_input, d_output, d_gamma, d_beta, d_mean, d_var, m, n);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("computeMeanAndVariance on device elapsed %f sec\n", iElaps);

    iStart = cpuSecond();
    layerNormDevice<<<grid, block, sharedMemSize>>>(d_input, d_output, d_gamma, d_beta, m, n);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("layerNormDevice on device elapsed %f sec\n", iElaps);

    cudaMemcpy(gpuRef, d_output, nElem * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, d_mean, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(var, d_var, m * sizeof(float), cudaMemcpyDeviceToHost);

    checkResult(h_output, gpuRef, m, n);

    // printMatrix(h_input, m, n);
    // printMatrix(h_output, m, n);
    // printMatrix(gpuRef, m, n);
    // printMatrix(mean, m, 1);
    // printMatrix(var, m, 1);

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
