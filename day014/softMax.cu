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

__global__ void softMaxDevice(float *input, float *output, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
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

__global__ void softMaxDeviceSharedMem(float *input, float *output, int m, int n) {
    extern __shared__ float sharedData[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row > m) return;

    // step 1: calculate the maximum value in the row
    float maxVal = -INFINITY;
    for (int j = tid; j < n; j += blockDim.x) {
        maxVal = fmaxf(maxVal, input[row * n + j]);
    }

    // reduce maxVal across threads in the block
    sharedData[tid] = maxVal;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] = fmaxf(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }
    maxVal = sharedData[0];

    // step 2: calculate the sum of exponentials
    float sumExp = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        sumExp += expf(input[row * n + j] - maxVal);
    }

    // reduce sumExp across threads in the block
    sharedData[tid] = sumExp;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    sumExp = sharedData[0];

    // step 3: calculate the softmax values
    for (int j = tid; j < n; j += blockDim.x) {
        output[row * n + j] = expf(input[row * n + j] - maxVal) / sumExp;
    }
}

int main() {
    int m = 1 << 10;
    int n = 1 << 16;
    int blockSize = 512;
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

    dim3 block(blockSize);
    dim3 grid((m + block.x - 1) / block.x);

    iStart = cpuSecond();
    softMaxDevice<<<grid, block>>>(d_input, d_output, m, n);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("softmax on device elapsed %f sec\n", iElaps);

    dim3 gridS(m);       // One block per row
    dim3 blockS(blockSize);    // Number of threads per block
    size_t sharedMemSize = block.x * sizeof(float);

    // checkIndex<<<gridS, blockS>>>();

    iStart = cpuSecond();
    softMaxDeviceSharedMem<<<gridS, blockS, sharedMemSize>>>(d_input, d_output, m, n);
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