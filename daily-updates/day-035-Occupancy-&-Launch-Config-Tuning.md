# Day 35: Occupancy & Launch Configuration Tuning

In this lesson, we explore **occupancy**—a critical concept in CUDA that measures how many warps can be active on a Streaming Multiprocessor (SM) at once compared to the maximum possible. We’ll learn how to use **the Occupancy Calculator** (provided in Nsight Compute or as a standalone spreadsheet) to refine block size and other kernel launch parameters for better SM utilization. Additionally, we will discuss how shared memory usage and other resource considerations (e.g., registers, warp scheduling) affect occupancy. By carefully tuning these parameters, developers can often unlock significant performance gains.

**Key References**:  
- [CUDA C Best Practices Guide – Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)  
- [NVIDIA Nsight Compute – Occupancy Analysis](https://docs.nvidia.com/nsight-compute/)  
- [CUDA C Programming Guide – Shared Memory & Kernel Launch Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

---

## Table of Contents

1. [Overview](#1-overview)  
2. [What is Occupancy?](#2-what-is-occupancy)  
3. [Occupancy Calculator & Key Factors](#3-occupancy-calculator--key-factors)  
   - [a) Registers & Shared Memory Usage](#a-registers--shared-memory-usage)  
   - [b) Block Size & Thread Count](#b-block-size--thread-count)  
   - [c) Grid Configuration](#c-grid-configuration)  
4. [Practical Exercise: Tuning Kernel Launch Config](#4-practical-exercise-tuning-kernel-launch-config)  
   - [a) Example Kernel Code](#a-example-kernel-code)  
   - [b) Using the Occupancy Calculator](#b-using-the-occupancy-calculator)  
   - [c) Testing Different Configurations](#c-testing-different-configurations)  
5. [Conceptual Diagrams](#5-conceptual-diagrams)  
6. [Common Debugging Pitfalls & Best Practices](#6-common-debugging-pitfalls--best-practices)  
7. [References & Further Reading](#7-references--further-reading)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)

---

## 1. Overview

**Occupancy** is defined as the ratio of active warps per SM to the maximum number of warps that can be active. High occupancy helps hide memory latency but does not guarantee maximum performance by itself—there are kernels that perform better at slightly lower occupancy if they make efficient use of shared memory and registers. The **Occupancy Calculator** helps us predict what block size (and shared memory usage) yields the highest occupancy for a given kernel on a specific GPU architecture.

**Key Goals**:
- Understand how occupancy is calculated.
- Learn to adjust block sizes, shared memory usage, and registers to maximize occupancy.
- Use the Occupancy Calculator (from Nsight Compute or the spreadsheet) to find near-optimal kernel launch configurations.

---

## 2. What is Occupancy?

- **Definition**: Occupancy = \(\frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}\)
- **Warp**: Group of 32 threads that execute instructions in lockstep on the GPU.
- **Active Warps**: Number of warps that can be scheduled and swapped by the SM to hide latency.
- **Why Occupancy Matters**:  
  If occupancy is too low, the GPU may idle waiting for memory operations. If it’s very high but the kernel is not memory-bound, you might not get additional performance benefits.

**Balancing Occupancy**:
- If your kernel is heavily memory-bound, moderate occupancy may be enough because adding more warps could saturate memory bandwidth.
- If your kernel is compute-bound, higher occupancy might help keep the SM busy while other warps wait for instructions.

---

## 3. Occupancy Calculator & Key Factors

The **Occupancy Calculator** is usually found in:
- **Nsight Compute**: Occupancy analysis sections per kernel.
- **Spreadsheet**: Provided by NVIDIA for manual calculations.

### a) Registers & Shared Memory Usage

1. **Registers**:  
   Each SM has a finite pool of registers. If your kernel uses many registers per thread, you can’t fit as many threads on the SM, reducing occupancy.

2. **Shared Memory**:  
   Shared memory is allocated per block. If your kernel uses significant shared memory per block, fewer blocks can reside concurrently on the same SM, also affecting occupancy.

### b) Block Size & Thread Count

- **Thread count** is crucial because a warp is always 32 threads. You typically want block sizes that are multiples of 32 to avoid partially filled warps.
- **Large block sizes** might be beneficial if your kernel can handle high concurrency, but it may also increase register usage or shared memory usage per block, capping occupancy.

### c) Grid Configuration

- The **number of blocks** (grid dimension) impacts concurrency on each SM. If there aren’t enough blocks to keep all SMs busy, you might see lower occupancy overall.
- **Launch Bounds**: Using \(\_\_launch\_bounds()\_\_) can help tune register usage for certain block sizes on certain architectures.

---

## 4. Practical Exercise: Tuning Kernel Launch Config

### a) Example Kernel Code

Here’s a simple kernel that uses some shared memory for partial sums (an example of a partial reduction). We will tune block size and shared memory usage:

```cpp
// day35_occupancyExample.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void partialReductionKernel(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory if in range
    sdata[tid] = (globalIndex < N) ? input[globalIndex] : 0.0f;
    __syncthreads();

    // Simple partial sum in shared memory
    // e.g., half threads add to other half iteratively
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write the block’s partial sum to the output array
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1 << 20; // 1M
    size_t size = N * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc( ( (N + 255) / 256 ) * sizeof(float));
    for(int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // or some random data
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, ((N+255)/256)*sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // We'll vary blockDim.x to see how it affects occupancy
    // We'll also pass blockDim.x * sizeof(float) as shared memory
    // to store partial sums.
    int blockSize = 256; // We'll test e.g. 128, 256, 512, etc
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch with dynamic shared memory
    partialReductionKernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, gridSize*sizeof(float), cudaMemcpyDeviceToHost);

    // Summation check
    float finalSum = 0.0f;
    for(int i = 0; i < gridSize; i++){
        finalSum += h_output[i];
    }
    printf("Final partial reduction sum = %f\n", finalSum);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
```

### b) Using the Occupancy Calculator

Steps:
1. **Profile** the kernel or refer to your code to find:
   - Number of registers used per thread (from compilation or Nsight Compute).
   - Shared memory usage (here, `blockSize * sizeof(float)`).
   - Block size (threads per block).
2. **Plug** these parameters and the GPU architecture (e.g., **sm_80** for Ampere) into the Occupancy Calculator.
3. **Observe** the occupancy: see how adjusting block size or changing shared memory usage changes the number of concurrent blocks per SM.

### c) Testing Different Configurations

Experiment by adjusting **blockSize** (128, 256, 512, etc.) and observe:
- How many registers are used per thread?
- How does that limit concurrent warps per SM?
- How does shared memory usage per block limit the number of blocks that can fit per SM?

Measure **execution times** for each configuration to find an **optimal** or near-optimal block size for your kernel on your specific GPU.

---

## 5. Conceptual Diagrams

### Diagram 1: Occupancy Factors

```mermaid
flowchart TD
    A[Kernel Resource Usage] --> B[Registers Used per Thread]
    A --> C[Shared Memory Used per Block]
    B --> D[Occupancy (Active Warps per SM)]
    C --> D
    D --> E[Kernel Performance]

    E -->|Additionally| F[Block Size & Grid Size Tuning]
```

*Explanation:*  
- Kernel resource usage (registers per thread, shared memory per block) determines how many blocks can fit on an SM, thus affecting occupancy.
- Occupancy strongly influences kernel performance.

### Diagram 2: Using the Occupancy Calculator

```mermaid
flowchart TD
    A[Compile Kernel] --> B[Get Resource Usage (NVCC or Nsight Compute)]
    B --> C[Input Data into Occupancy Calculator (Registers, Shared Mem, Block Size)]
    C --> D[Calculator Predicts Max Blocks per SM, Occupancy Percentage]
    D --> E[Adjust Block Size / Shared Memory?]
    E --> C
    D --> F[Pick Launch Config with Good Occupancy]
```

*Explanation:*  
- Shows how the iterative process works: gather resource usage, plug into the Occupancy Calculator, adjust block size or shared memory usage, and refine.

---

## 6. Common Debugging Pitfalls & Best Practices

| **Pitfall**                                    | **Solution**                                                                     |
|------------------------------------------------|----------------------------------------------------------------------------------|
| **Overlooking Shared Memory**                  | Carefully account for shared memory usage in the kernel; if you use a large amount, fewer blocks can run per SM. |
| **Blindly Maximizing Occupancy**               | While high occupancy can help, it might not always yield the best performance if you’re memory-bound. Profile your kernel to confirm. |
| **Ignoring Registers**                         | Check how many registers each thread uses. If it’s too high, reduce usage via optimizations or `__launch_bounds__()`. |
| **Not verifying performance improvements**      | After adjusting block size or resources, always measure kernel execution time to ensure real gains. |
| **Inconsistent kernel usage**                  | If your kernel changes (e.g., added shared memory), re-check occupancy. |

---

## 7. References & Further Reading

1. **CUDA C Best Practices Guide – Occupancy**  
   [CUDA C Best Practices: Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)  
2. **NVIDIA Nsight Compute**  
   [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)  
3. **"Programming Massively Parallel Processors" by David B. Kirk and Wen-mei W. Hwu**  
4. **NVIDIA Developer Blog**  
   [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)

---

## 8. Conclusion

In **Day 35**, we delved into **Occupancy & Launch Configuration Tuning**. We explored:
- **Occupancy** and how it’s calculated.
- The role of **registers** and **shared memory** in determining occupancy.
- How to use the **Occupancy Calculator** to refine block sizes for better SM utilization.
- Why **overlooking shared memory** usage can reduce occupancy.
- Provided conceptual diagrams and a practical code example demonstrating partial reduction with variable shared memory usage.

---

## 9. Next Steps

- **Gather Resource Usage**: Use Nsight Compute to find registers per thread and shared memory usage per kernel.  
- **Experiment**: Try multiple block sizes and measure performance. Compare predicted occupancy vs. actual throughput.  
- **Optimize**: If you discover an optimum block size from the Occupancy Calculator, confirm improvements by real-world kernel timing.  
- **Refine**: Combine occupancy tuning with memory coalescing, warp efficiency, and concurrency strategies for maximal performance.

Happy CUDA coding, and keep optimizing your kernels for higher occupancy and faster execution!
```
