# Day 81: Cooperative Groups – Advanced Patterns

In this lesson, we dive into **advanced cooperative group patterns** by leveraging grid-level synchronization in CUDA. By coordinating all blocks in a kernel launch, you can implement complex parallel patterns—such as global reductions, hierarchical data exchange, or dynamic load balancing—that require a full grid synchronization. However, **grid-level sync** is only supported on select GPUs and requires careful design to avoid pitfalls when hardware or driver support is lacking.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Understanding Cooperative Groups](#2-understanding-cooperative-groups)  
   - [a) What Are Cooperative Groups?](#a-what-are-cooperative-groups)  
   - [b) Grid-Level Cooperative Groups](#b-grid-level-cooperative-groups)  
3. [Advanced Patterns with Grid-Level Sync](#3-advanced-patterns-with-grid-level-sync)  
   - [a) Use Cases](#a-use-cases)  
   - [b) Hardware Considerations](#b-hardware-considerations)  
4. [Implementation Approach](#4-implementation-approach)  
   - [a) Setting Up a Cooperative Kernel](#a-setting-up-a-cooperative-kernel)  
   - [b) Synchronization and Data Merging](#b-synchronization-and-data-merging)  
5. [Code Example: Grid-Level Cooperative Reduction](#5-code-example-grid-level-cooperative-reduction)  
   - [Explanation & Comments](#explanation--comments)  
6. [Conceptual Diagrams](#6-conceptual-diagrams)  
   - [Diagram 1: Grid-Level Sync Flow](#diagram-1-grid-level-sync-flow)  
   - [Diagram 2: Hierarchical Reduction with Cooperative Groups](#diagram-2-hierarchical-reduction-with-cooperative-groups)  
   - [Diagram 3: Advanced Pattern for Global Data Exchange](#diagram-3-advanced-pattern-for-global-data-exchange)  
7. [Common Pitfalls](#7-common-pitfalls)  
8. [References & Further Reading](#8-references--further-reading)  
9. [Conclusion](#9-conclusion)  
10. [Next Steps](#10-next-steps)

---

## 1. Overview

Modern CUDA applications increasingly require **global synchronization** across all blocks to implement advanced parallel patterns. Cooperative groups enable a **grid-level synchronization** within a single kernel launch, allowing threads from different blocks to synchronize and share data directly. This powerful feature is essential for algorithms such as global reductions or complex multi-phase computations. However, because **not all GPUs support grid-level sync**, it’s important to query device capabilities and design fallbacks.

---

## 2. Understanding Cooperative Groups

### a) What Are Cooperative Groups?

Cooperative groups are a set of CUDA APIs that provide mechanisms for grouping threads together beyond the traditional block or warp level. They allow for:
- **Fine-Grained Synchronization**: Within a block, warp, or even across the entire grid.
- **Data Exchange**: Facilitate direct sharing of data between threads in different blocks without resorting to the host.

### b) Grid-Level Cooperative Groups

A **grid group** aggregates all threads from all blocks in a kernel launch. By calling `cg::this_grid()`, you obtain a grid-level group object that can invoke `grid.sync()` to synchronize across the entire grid. This feature is critical for:
- Global reductions
- Coordinated data transfers
- Adaptive algorithm patterns

---

## 3. Advanced Patterns with Grid-Level Sync

### a) Use Cases

- **Global Reduction**: Each block computes a partial sum, and grid-level sync allows a designated block to merge all partial sums.
- **Dynamic Work Redistribution**: After an initial computation phase, tasks can be redistributed based on collective results.
- **Global Data Exchange**: Synchronize all blocks before performing a final update across the entire dataset.

### b) Hardware Considerations

- **Support Variability**: Not all GPUs support grid-level synchronization. Devices must support cooperative launches, so always check device properties before using these features.
- **Driver Requirements**: Ensure your CUDA Toolkit and drivers support cooperative groups at the grid level.

---

## 4. Implementation Approach

### a) Setting Up a Cooperative Kernel

- **Enable Cooperative Launch**: Use APIs such as `cudaLaunchCooperativeKernel` or ensure your device supports grid sync.
- **Create Grid Group**: Inside the kernel, call `cooperative_groups::this_grid()` to obtain the grid group.
- **Synchronization Point**: Use `grid.sync()` to enforce that all blocks have completed their work before proceeding.

### b) Synchronization and Data Merging

- **Local Reductions**: Each block performs local computations (e.g., partial sums).
- **Global Merge**: After `grid.sync()`, one designated block (often block 0) can merge the partial results.
- **Ensure Correctness**: Synchronization must occur before the merging phase to avoid data races or incomplete computations.

---

## 5. Code Example: Grid-Level Cooperative Reduction

Below is a simplified example demonstrating a grid-level reduction using cooperative groups.

```cpp
// File: cooperative_reduction.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
namespace cg = cooperative_groups;

__launch_bounds__(256)
__global__ void cooperativeReductionKernel(const float* input, float* output, int N) {
    // Create a grid group to synchronize all blocks
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory, handling boundary conditions
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Perform reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }

    // Synchronize all blocks in the grid
    grid.sync();

    // Final reduction by block 0
    if (blockIdx.x == 0) {
        // Use shared memory to combine all block results
        if (tid < gridDim.x) {
            sdata[tid] = output[tid];
        }
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < gridDim.x) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[0] = sdata[0]; // Final result
        }
    }
}

int main() {
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, (N / 256 + 1) * sizeof(float)); // one per block

    // Initialize input on host and copy to device (omitted for brevity)

    // Launch cooperative kernel with shared memory size = blockDim.x * sizeof(float)
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    cooperativeReductionKernel<<<grid, block, block.x * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Retrieve result from d_output[0] (omitted for brevity)

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
```

### Explanation & Comments

- **Grid Group**: `cg::this_grid()` creates a grid-level group to synchronize all blocks.
- **Shared Memory Reduction**: Each block reduces its portion of the data in shared memory.
- **Grid Synchronization**: `grid.sync()` ensures that all partial results are written before the final merge.
- **Final Merge**: Block 0 then uses shared memory to combine all partial sums into a final result.

---

## 6. Multiple Conceptual Diagrams

### Diagram 1: Grid-Level Sync Flow

```mermaid
flowchart TD
    A[Each Block: Compute Partial Sum]
    B[Write Partial Sum to Global Memory]
    C[grid.sync() - Synchronize all blocks]
    D[Block 0: Merge Partial Sums]
    A --> B
    B --> C
    C --> D
```

**Explanation**:  
All blocks compute partial sums and write to global memory. After a grid-wide sync, block 0 aggregates these results.

---

### Diagram 2: Hierarchical Reduction using Cooperative Groups

```mermaid
flowchart LR
    subgraph Block-Level Reduction
    A1[Threads reduce to a single value]
    A2[Write partial result]
    end
    subgraph Grid-Level Merge
    B1[grid.sync()]
    B2[Block 0 reads all partial results]
    B3[Final reduction in shared memory]
    end
    A1 --> A2 --> B1
    B1 --> B2 --> B3
```

**Explanation**:  
The reduction is performed in two stages: first, within each block, then globally by block 0 after a grid sync.

---

### Diagram 3: Advanced Pattern for Global Data Exchange

```mermaid
flowchart TD
    A[Kernel launches: each block performs computation]
    B[grid.sync() ensures all data is available]
    C[Designated block (block 0) aggregates global results]
    D[Final output stored in global memory]
    A --> B
    B --> C
    C --> D
```

**Explanation**:  
This diagram shows a more general pattern where global synchronization allows one block to safely gather and combine results from all blocks.

---

## 7. Common Pitfalls

- **Hardware Limitations**: Not all GPUs support grid-level sync. Always verify that your hardware supports cooperative launches.  
- **Incomplete Synchronization**: Failing to call `grid.sync()` can lead to race conditions, as some blocks may still be computing when the final merge begins.  
- **Excessive Overhead**: Overusing grid-level sync in small kernels can add unnecessary overhead. Use it judiciously when global coordination is essential.

---

## 8. References & Further Reading

- [CUDA C Programming Guide – Cooperative Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)  
- [Nsight Systems – Profiling Cooperative Groups](https://docs.nvidia.com/nsight-systems/)  
- [NVIDIA Developer Blog – Advanced Cooperative Group Techniques](https://developer.nvidia.com/blog/cooperative-groups/)

---

## 9. Conclusion

**Day 81** explores advanced cooperative group patterns by using grid-level synchronization. By coordinating all blocks within a single kernel launch, you can perform complex global operations—like reductions or data exchanges—without multiple kernel launches. However, the success of such strategies depends on hardware support and careful synchronization, as missing sync calls can lead to race conditions or incomplete results.

---

## 10. Next Steps

1. **Verify Hardware Support**: Query your GPU to confirm it supports grid-level cooperative groups.  
2. **Profile**: Use Nsight Systems to examine the overhead and benefits of grid-level sync in your applications.  
3. **Optimize**: Experiment with different block sizes and shared memory configurations to balance local reduction and global merging performance.  
4. **Extend Patterns**: Apply grid-level sync to other advanced patterns such as global prefix sum or adaptive load balancing.  
5. **Documentation**: Clearly document your kernel design to aid in future maintenance and debugging.

```
