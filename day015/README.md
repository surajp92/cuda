# SoftMax2D.cu

This file contains the implementation of a 2D SoftMax operation using CUDA. The primary function, `softmaxKernel`, is designed to compute the softmax activation for a 2D input tensor in a highly parallelized manner, leveraging the power of GPUs.

## Overview of SoftMax

SoftMax is a mathematical function that converts a vector of numbers into probabilities. It is commonly used in machine learning for classification tasks. The formula for the SoftMax function is:

\[
\text{SoftMax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
\]

## Implementation Details

### `softmaxKernel` Function

The `softmaxKernel` function is implemented to compute the SoftMax operation row-wise for a 2D input tensor. Below is a detailed explanation of its steps:

1. **Thread and Block Mapping**:
    - Each thread computes its row and column indices (row and col) based on the thread and block indices. 
    - Each thread processes a subset of the row's elements, and the results are stored in shared memory.

2. **Input Normalization**:
    - To improve numerical stability, the maximum value of each row is first computed using a parallel reduction. This prevents large exponentials from causing overflow.

3. **Exponentiation and Sum**:
    - Each thread computes the exponential of its corresponding element (after subtracting the row maximum).
    - A parallel reduction is then used to compute the sum of these exponentials for the row.

4. **SoftMax Calculation**:
    - Each thread divides its exponential value by the computed sum to produce the final SoftMax value.

5. **Memory Access**:
    - Shared memory is used to store intermediate results (e.g., row maximum and sum of exponentials) to minimize global memory access and improve performance.

### Pseudocode

```cpp
__global__ void softmaxKernel(float* input, float* output, int rows, int cols) {
     // Step 1: Compute row maximum using parallel reduction
     // Step 2: Compute exponentials and their sum
     // Step 3: Normalize each element by dividing by the sum
}
```

### Key Optimizations

- **Shared Memory**: Used for intermediate computations to reduce latency.
- **Parallel Reduction**: Efficiently computes row-wise maximum and sum of exponentials.
- **Thread Synchronization**: Ensures correctness during shared memory operations.

## Usage

To use the `softmaxKernel` function, compile the file with `nvcc` and call the kernel with appropriate grid and block dimensions. Ensure the input tensor is flattened and resides in GPU memory.

```bash
nvcc -o softmax2D SoftMax2D.cu
```

## Example

Given a 2D tensor:

```
[ [1.0, 2.0, 3.0],
  [1.0, 3.0, 5.0] ]
```

The output after applying SoftMax row-wise will be:

```
[ [0.0900, 0.2447, 0.6652],
  [0.0159, 0.1173, 0.8668] ]
```

## Conclusion

The `softmaxKernel` function in `SoftMax2D.cu` demonstrates an efficient CUDA implementation of the SoftMax operation, optimized for 2D tensors. It is a critical building block for GPU-accelerated machine learning workflows.  