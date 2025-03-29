# Layer Normalization Implementation

## Overview

This file, `layerNormReduce.cu`, implements Layer Normalization on both the host and device using CUDA. It includes:

- A host-based implementation of Layer Normalization.
- A naive device-based implementation of Layer Normalization.
- An optimized device-based implementation using shared memory for mean and variance computation.

The program compares the performance and correctness of the host and device implementations.

## Functions

### Utility Functions
- **`double cpuSecond()`**: Measures elapsed time.
- **`void initialdata(float *A, int size)`**: Initializes an array with random values.
- **`void printMatrix(float *C, const int ny, const int nx)`**: Prints a 2D matrix.
- **`void checkResult(float *hostRef, float *gpuRef, int nx, int ny)`**: Compares results between host and device.

### Layer Normalization Implementations
- **`void layerNorm(float *input, float *output, float *gamma, float *beta, int m, int n)`**: Host implementation of Layer Normalization.
- **`__global__ void layerNormDevice(float *input, float *output, float *gamma, float *beta, int m, int n)`**: Naive device implementation of Layer Normalization.
- **`__global__ void computeMeanAndVarianceLayerNorm(const float *input, float *output, const float *gamma, const float *beta, float *mean, float *var, int M, int N)`**: Optimized device implementation using shared memory and reduction operation for mean and variance.

## Main Workflow

1. Allocates and initializes host and device memory.
2. Executes the host and device implementations of Layer Normalization.
3. Measures and compares the execution time of each implementation.
4. Verifies the correctness of the device implementation against the host implementation.

## Parameters

- **`m`**: Number of rows in the input matrix.
- **`n`**: Number of columns in the input matrix.
- **`blockSize_x`**: Number of threads per block for CUDA kernels.
- **`nElem`**: Total number of elements in the input matrix.

## Notes

- The optimized device implementation uses shared memory to compute the mean and variance for each row.
- Numerical stability is ensured by adding a small epsilon value (`1e-7`) during variance computation.
- The program assumes that the input matrix is row-major.

## Usage

### Compilation
```bash
nvcc -o layerNormReduce layerNormReduce.cu
```

### Execution
```bash
./layerNormReduce
```
