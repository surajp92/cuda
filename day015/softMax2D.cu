#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

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

void softMaxHost(float *input, float *output, int M, int N) {
    for (int i = 0; i < M; i++) {
        float maxVal = -INFINITY;
        // Find the maximum value in the row for numerical stability
        for (int j = 0; j < N; j++) {
            if (input[i * N + j] > maxVal) {
                maxVal = input[i * N + j];
            }
        }

        // Compute the sum of exponentials
        float sumExp = 0.0f;
        for (int j = 0; j < N; j++) {
            sumExp += expf(input[i * N + j] - maxVal);
        }

        // Compute the softmax values
        for (int j = 0; j < N; j++) {
            output[i * N + j] = expf(input[i * N + j] - maxVal) / sumExp;
        }
    }
}

__global__ void checkIndex() {

    printf("threadIdx: (%d %d %d) blockIdx: (%d %d %d)  blockDim: (%d %d %d) gridDim: (%d %d %d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}

__device__ float reduceOperation(float *sharedData, float value, int blockSize, bool isMax) {
    // Store the value in shared memory
    sharedData[threadIdx.x] = value;
    __syncthreads();

    // Perform reduction
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (isMax) {
                sharedData[threadIdx.x] = fmaxf(sharedData[threadIdx.x], sharedData[threadIdx.x + stride]);
            } else {
                sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Return the reduced value
    return sharedData[0];
}

__global__ void softmaxKernel(float *input, float *output, int M, int N) {
    extern __shared__ float sharedData[]; // Shared memory for intermediate results

    // Compute row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    // Shared memory for the current row
    float *rowShared = &sharedData[threadIdx.y * blockDim.x];

    if (row < M && col < N) {
        // Step 1: Compute the maximum value for the row
        float maxVal = -INFINITY;
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            maxVal = fmaxf(maxVal, input[row * N + j]);
        }
        maxVal = reduceOperation(rowShared, maxVal, blockDim.x, true);

        // Step 2: Compute the sum of exponentials
        float sumExp = 0.0f;
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            sumExp += expf(input[row * N + j] - maxVal);
        }
        sumExp = reduceOperation(rowShared, sumExp, blockDim.x, false);

        // Step 3: Compute the softmax values
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            output[row * N + j] = expf(input[row * N + j] - maxVal) / sumExp;
        }
    }
}


int main() {
    int m = 1 << 10;
    int n = 1 << 16;
    int nElem = m * n;

    float *h_input;
    float *h_output;
    float *gpuRef;

    h_input = (float *)malloc(nElem * sizeof(float));
    h_output = (float *)malloc(nElem * sizeof(float));
    gpuRef = (float *)malloc(nElem * sizeof(float));

    initialdata(h_input, nElem);

    float *d_input;
    float *d_output;

    cudaMalloc((float **)&d_input, nElem * sizeof(float));
    cudaMalloc((float **)&d_output, nElem * sizeof(float));

    cudaMemcpy(d_input, h_input, nElem * sizeof(float), cudaMemcpyHostToDevice);

    double iStart = cpuSecond();
    softMaxHost(h_input, h_output, m, n);
    double iElaps = cpuSecond() - iStart;
    printf("softmax on host elapsed %f sec\n", iElaps);

    int blockSize_x = 1024; // Threads per row (columns)
    int blockSize_y = 1;  // Threads per block processing multiple rows
    dim3 blockS(blockSize_x, blockSize_y); // 2D block
    dim3 gridS((n + blockSize_x - 1) / blockSize_x, (m + blockSize_y - 1) / blockSize_y); // 2D grid

    size_t sharedMemSize = blockSize_y * blockSize_x * sizeof(float); // Shared memory for multiple rows
    
    // checkIndex<<<gridS, blockS>>>();

    iStart = cpuSecond();
    softmaxKernel<<<gridS, blockS, sharedMemSize>>>(d_input, d_output, m, n);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("softmax on device with shared memory elapsed %f sec\n", iElaps);

    cudaMemcpy(gpuRef, d_output, nElem * sizeof(float), cudaMemcpyDeviceToHost);

    checkResult(h_output, gpuRef, m, n);

    free(h_input);
    free(h_output);
    free(gpuRef);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}