# Day 79: Portability & Scalability Best Practices

Adapting CUDA code to work efficiently on various GPU architectures is critical for building portable and scalable HPC applications. Hardcoding parameters or optimizations for a single GPU (e.g., targeting a specific SM version) can result in poor performance when deployed on different hardware. In this lesson, we explore strategies to write adaptable CUDA code by querying device properties, using conditional compilation, and designing kernels that gracefully scale across multiple architectures.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Why Portability and Scalability Matter](#2-why-portability-and-scalability-matter)  
3. [Challenges of Hardcoding for a Single GPU](#3-challenges-of-hardcoding-for-a-single-gpu)  
4. [Techniques for Portability](#4-techniques-for-portability)  
   - [a) Querying Device Properties](#a-querying-device-properties)  
   - [b) Conditional Compilation and Macros](#b-conditional-compilation-and-macros)  
   - [c) Parameterized Kernel Launches](#c-parameterized-kernel-launches)  
   - [d) Use of Unified Memory and Dynamic Parallelism](#d-use-of-unified-memory-and-dynamic-parallelism)  
5. [Implementation Guidelines and Best Practices](#5-implementation-guidelines-and-best-practices)  
6. [Code Example: Portable Kernel Example](#6-code-example-portable-kernel-example)  
7. [Conceptual Diagrams](#7-conceptual-diagrams)  
   - [Diagram 1: Device Query and Adaptation Flow](#diagram-1-device-query-and-adaptation-flow)  
   - [Diagram 2: Conditional Kernel Launch Based on SM Version](#diagram-2-conditional-kernel-launch-based-on-sm-version)  
   - [Diagram 3: Scalable Design – Code Abstraction for Multiple Architectures](#diagram-3-scalable-design-code-abstraction-for-multiple-architectures)  
8. [References & Further Reading](#8-references--further-reading)  
9. [Conclusion](#9-conclusion)  
10. [Next Steps](#10-next-steps)

---

## 1. Overview

Portability in CUDA programming means writing code that can efficiently run on a variety of GPU architectures (e.g., different SM versions). Scalability means that your application can take advantage of future hardware improvements without requiring major rewrites. This lesson focuses on techniques to avoid hardcoding architecture-specific parameters, ensuring your code adapts dynamically and remains robust across CUDA Toolkit releases.

---

## 2. Why Portability and Scalability Matter

- **Broader Deployment**: Applications that adapt to different GPUs ensure that users with various hardware generations can run your code efficiently.
- **Future-Proofing**: As new GPU architectures emerge with higher compute capabilities or different resource limits, adaptable code can leverage these improvements immediately.
- **Performance Consistency**: By tailoring optimizations based on the hardware, you can achieve near-optimal performance across diverse platforms without maintaining multiple codebases.

---

## 3. Challenges of Hardcoding for a Single GPU

- **Static Optimizations**: Hardcoding parameters like thread block size, shared memory allocation, or specific kernel unrolling factors for one SM version may not be ideal for another architecture.
- **Resource Mismatch**: Different GPUs have varying numbers of registers, shared memory sizes, and maximum concurrent threads per SM.
- **Maintenance Overhead**: Updating code for each new architecture becomes labor-intensive and error-prone if not designed for portability.

---

## 4. Techniques for Portability

### a) Querying Device Properties
- Use `cudaGetDeviceProperties()` to determine the current GPU’s compute capability, number of SMs, registers per SM, and shared memory available.
- Tailor your kernel launch parameters and optimizations based on these queried values.

### b) Conditional Compilation and Macros
- Define macros or use preprocessor directives (`#if`, `#ifdef`) to enable different code paths based on the CUDA architecture.
- For example, use different block sizes or unroll factors if compiling for SM 7.x vs. SM 8.x.

### c) Parameterized Kernel Launches
- Instead of hardcoding grid and block dimensions, pass them as parameters computed at runtime based on device properties.
- Write kernels that use flexible memory allocation strategies and dynamic shared memory to adapt to varying resource limits.

### d) Use of Unified Memory and Dynamic Parallelism
- **Unified Memory (UM)** abstracts away the physical location of memory, enabling code to run on different devices without explicit data transfers.
- **Dynamic Parallelism** allows kernels to launch child kernels, which can be tuned dynamically according to the device’s capabilities.

---

## 5. Implementation Guidelines and Best Practices

- **Device Querying**: Always query device properties at the start of your application. Use these properties to adjust kernel configurations.
- **Modular Code Design**: Write functions that accept configuration parameters for block size, shared memory, and other resources.
- **Conditional Paths**: Use conditional logic to select optimized routines based on the current GPU’s compute capability.
- **Benchmarking**: Profile your code on multiple devices and use tools like Nsight Compute to compare performance, ensuring your dynamic adjustments are effective.
- **Documentation**: Clearly document which parameters are expected to change with different architectures, aiding future maintenance and updates.

---

## 6. Code Example: Portable Kernel Example

Below is a simple example demonstrating how to query device properties and launch a kernel with parameters tailored for the current GPU. This example multiplies two matrices, with block size chosen based on the SM version.

```cpp
// File: portable_kernel_example.cu

#include <cuda_runtime.h>
#include <stdio.h>

// Simple matrix multiplication kernel (naive version for illustration)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if(row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Query device properties
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);
    printf("Device %d: %s (Compute Capability: %d.%d)\n", device, prop.name, prop.major, prop.minor);

    // Set parameters based on device compute capability
    dim3 blockDim, gridDim;
    if(prop.major >= 7) {
        // For newer architectures, larger blocks might work better
        blockDim = dim3(32, 32);
    } else {
        // For older architectures, use a smaller block size
        blockDim = dim3(16, 16);
    }
    int N = 1024; // matrix size N x N
    gridDim = dim3((N + blockDim.x - 1) / blockDim.x,
                   (N + blockDim.y - 1) / blockDim.y);

    // Allocate matrices (for simplicity, assume square matrices of size N x N)
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize matrices
    for(int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f; // dummy data
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with dynamically chosen block dimensions
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Sample result: h_C[0] = %f\n", h_C[0]);

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

### Explanation & Comments
- **Device Query**: The program first queries the device properties to determine the compute capability.
- **Conditional Block Size**: The kernel launch parameters are chosen based on whether the device is of a newer architecture (e.g., SM 7.x or above) or older.
- **Kernel Launch**: A simple matrix multiplication kernel is launched with dynamically determined block and grid dimensions.
- **Result Validation**: The output is copied back to the host for verification.

---

## 7. Multiple Conceptual Diagrams

### Diagram 1: Device Query and Adaptive Configuration

```mermaid
flowchart TD
    A[Start Application]
    B[Query Device Properties (cudaGetDeviceProperties)]
    C{Compute Capability}
    D[High (>=7.x): Use 32x32 blocks]
    E[Low (<7.x): Use 16x16 blocks]
    F[Set grid dimensions based on block size]
    A --> B
    B --> C
    C -- High --> D
    C -- Low --> E
    D --> F
    E --> F
```

**Explanation**:  
The application queries the GPU properties and selects kernel launch parameters accordingly.

---

### Diagram 2: Modular Kernel Design for Portability

```mermaid
flowchart LR
    A[Device Query]
    B[Conditional Compilation / Macros]
    C[Parameterize Kernel Launch (grid, block dimensions)]
    D[Kernel Execution]
    A --> B
    B --> C
    C --> D
```

**Explanation**:  
This diagram emphasizes modular design: using device queries and conditional macros to parameterize and adapt kernel execution for different architectures.

---

### Diagram 3: End-to-End Portable Workflow

```mermaid
flowchart TD
    A[Host allocates matrices]
    B[Query GPU properties and set configuration]
    C[Launch kernel with dynamic block/grid sizes]
    D[Kernel executes on GPU (optimized per device)]
    E[Results copied back to host]
    A --> B
    B --> C
    C --> D
    D --> E
```

**Explanation**:  
An end-to-end workflow showing how portability is achieved from memory allocation to kernel execution and result retrieval, all based on dynamic configuration.

---

## 8. References & Further Reading

- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)  
- [CUDA C Programming Guide – Multiple GPUs & Device Properties](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multiple-gpus)  
- [CUDA C Best Practices Guide – Code Portability and Performance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)  
- [Nsight Systems and Compute Tools](https://docs.nvidia.com/nsight-systems/)

---

## 9. Conclusion

**Day 79** emphasizes that writing portable and scalable CUDA code is essential for long-term success. By dynamically querying device properties and adapting kernel configurations (such as block sizes and shared memory usage), you can write code that performs efficiently across different GPU architectures. Avoiding hardcoded values prevents performance pitfalls and improves portability, ensuring that your applications remain robust as hardware evolves.

---

## 10. Next Steps

1. **Extend Device Queries**: Further refine your code to adjust for additional device properties (e.g., maximum threads per SM, shared memory per block).
2. **Benchmark Across Architectures**: Test your kernels on different GPUs to ensure performance scales.
3. **Integrate Conditional Compilation**: Use macros to maintain different code paths for SM-specific optimizations.
4. **Profile with Nsight Tools**: Confirm that your dynamic configurations result in optimal occupancy and throughput.
5. **Document Assumptions**: Clearly document which parameters are device-dependent for easier future maintenance.

```
