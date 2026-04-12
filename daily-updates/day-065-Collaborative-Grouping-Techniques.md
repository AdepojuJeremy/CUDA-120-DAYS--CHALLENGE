# Day 65: Collaborative Grouping Techniques

**Collaborative groups** in CUDA offer powerful synchronization and partitioning primitives beyond the traditional block- or warp-level scope. When implemented correctly, they enable advanced algorithms such as multi-block or grid-wide reductions, collective communications, and hierarchical data sharing. In this lesson, we explore how to use **cooperative groups** for sophisticated parallel reductions, discuss potential hardware limitations (not all GPUs support advanced group features), and provide illustrative code examples and diagrams.

---

## Table of Contents
1. [Overview](#1-overview)  
2. [Introduction to Cooperative Groups](#2-introduction-to-cooperative-groups)  
   - [a) Why Cooperative Groups?](#a-why-cooperative-groups)  
   - [b) Scope Levels (thread_block, tiled_partition, grid_group)](#b-scope-levels-thread_block-tiled_partition-grid_group)  
3. [Advanced Reductions Using Cooperative Groups](#3-advanced-reductions-using-cooperative-groups)  
   - [a) Extending Beyond a Single Block](#a-extending-beyond-a-single-block)  
   - [b) Using Grid-Level Synchronization](#b-using-grid-level-synchronization)  
4. [Implementation Steps](#4-implementation-steps)  
   - [a) Checking GPU Support](#a-checking-gpu-support)  
   - [b) Kernel Design](#b-kernel-design)  
   - [c) Launch Configuration & Host Code](#c-launch-configuration--host-code)  
5. [Code Example](#5-code-example)  
   - [Explanation & Comments](#explanation--comments)  
6. [Hardware Limitations & Common Pitfalls](#6-hardware-limitations--common-pitfalls)  
7. [Conceptual Diagrams](#7-conceptual-diagrams)  
8. [References & Further Reading](#8-references--further-reading)  
9. [Conclusion](#9-conclusion)  
10. [Next Steps](#10-next-steps)

---

## 1. Overview
While warp- and block-level primitives (like `__shfl_sync` or shared memory reduction) handle many HPC workloads, cooperative groups allow you to orchestrate **larger-scale concurrency**. This unlocks:
- **Grid-level synchronization** for data aggregation across blocks in a single kernel launch.
- **Flexible sub-block partitioning** (via tiled partitions) for custom compute patterns.
- **Hierarchical algorithms** (e.g., multi-block reduction) that do not require separate kernel launches.

However, **not all GPUs** support grid-wide synchronization or cooperative launch fully. Checking hardware and driver compatibility is a must.

---

## 2. Introduction to Cooperative Groups

### a) Why Cooperative Groups?
- **Fine-grained synchronization**: Beyond block-level sync, enabling threads in different blocks to coordinate without returning to the host.
- **Hierarchical decomposition**: Partitioning thread blocks into smaller sub-blocks (tiled partitions) for localized cooperation.
- **Advanced patterns**: Single-kernel multi-block reductions, prefix sums, or BFS expansions can be coded more naturally.

### b) Scope Levels (thread_block, tiled_partition, grid_group)
- **thread_block**: Standard block-level group, typically used for shared memory collaboration.
- **tiled_partition**: Sub-block partitioning to handle smaller subgroups within a block (e.g., groups of 32 threads).
- **grid_group**: Potentially synchronizes across the entire grid, but requires hardware that supports cooperative launch.

---

## 3. Advanced Reductions Using Cooperative Groups

### a) Extending Beyond a Single Block
Traditionally, large-scale reductions (over thousands of blocks) require multiple kernel launches or atomic operations. With **cooperative groups**, it’s possible to:
- Perform partial reductions within each block.
- Use a grid-level synchronization call (`this_grid().sync()`) to ensure all blocks complete their partial reductions.
- Complete the final accumulation in a single kernel launch, avoiding extra global sync or atomic overhead.

### b) Using Grid-Level Synchronization
Grid-level synchronization with `this_grid().sync()` is only supported on GPUs and driver configurations that allow **cooperative launch**. This approach eliminates the need for multiple kernel launches to perform hierarchical reductions, but must be used carefully to avoid performance pitfalls or unsupported architectures.

---

## 4. Implementation Steps

### a) Checking GPU Support
1. **Driver & Hardware**: Use `cudaDeviceGetAttribute()` (e.g., `cudaDevAttrCooperativeLaunch`) to verify that your GPU supports cooperative groups and grid-wide sync.  
2. **Launch Mode**: Use the correct launch parameters (e.g., `cudaLaunchCooperativeKernel`) if needed.

### b) Kernel Design
- **Declare** a group object inside your kernel (e.g., `cg::grid_group grid = cg::this_grid();`).
- **Perform** partial reductions in sub-block or warp-level partitions as usual.
- **Synchronize** the entire grid at specific points using `grid.sync()` if needed.

### c) Launch Configuration & Host Code
- **Use** the cooperative launch APIs or set the necessary flags for cooperative kernel launching.  
- **Allocate** device buffers for partial and final results.  
- **Profile** the kernel with Nsight Compute or Nsight Systems to confirm concurrency benefits.

---

## 5. Code Example

Below is a simplified demonstration of how to do a grid-wide reduction using cooperative groups. This example assumes a large input array that is too big for a single-block reduction.

```cpp
// File: cooperative_reduction.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Kernel that performs a multi-block reduction using cooperative groups
__global__ void cooperativeReductionKernel(const float *input, float *output, int N) {
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = (idx < N) ? input[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Each block writes partial sums to output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }

    // Synchronize the entire grid so that partial sums are visible to all
    grid.sync();

    // Let block 0 finalize the reduction
    if (blockIdx.x == 0) {
        // For simplicity, reuse sdata
        if (tid < gridDim.x) {
            sdata[tid] = (tid < gridDim.x) ? output[tid] : 0.0f;
        }
        __syncthreads();
        
        // Another reduction pass
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < gridDim.x) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        // The final sum is in sdata[0]
        if (tid == 0) {
            output[0] = sdata[0];
        }
    }
}

int main(){
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);

    // Allocate host memory and initialize
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // e.g., all ones
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float) * 1024); // Enough to hold partial sums from each block

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Check if the GPU supports cooperative groups (grid sync)
    // [Implementation to check GPU attributes omitted for brevity]

    // Launch cooperative kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // For cooperative launch, the kernel must be launched in a mode that supports it
    // [Implementation details for cooperative launch omitted for brevity]
    cooperativeReductionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Copy final result (sum) back to host
    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Final reduction result: %f\n", result);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    return 0;
}
```

### Explanation & Comments

1. **grid.sync()**  
   Used to synchronize all blocks in the grid, ensuring partial sums are fully written before the final pass.
2. **Partial Sums**  
   Each block performs a local reduction and writes to `output[blockIdx.x]`.  
3. **Final Summation**  
   Block 0 performs a second pass of reduction over the partial sums to get the global result.

---

## 6. Hardware Limitations & Common Pitfalls

- **Not All GPUs Support Grid Sync**: Some older architectures do not allow cooperative groups at a grid level. Always check `cudaDeviceCanAccessPeer` and related attributes for cooperative launch capabilities.
- **Launch Configuration**: Must ensure the kernel is launched in a cooperative mode (`cudaLaunchCooperativeKernel`) if needed.
- **Scalability**: If the kernel uses grid sync repeatedly, concurrency might suffer if not carefully orchestrated.
- **Error Checking**: Failing to record or wait for the correct sync can cause partial sums to be read prematurely.

---

## 7. Conceptual Diagrams

### Diagram 1: Multi-Block Reduction Flow with Cooperative Groups

```mermaid
flowchart TD
    A[Block-level partial reductions]
    B[grid.sync() ensures all partial sums are ready]
    C[Block 0 finalizes the reduction over partial sums]
    D[Write final result to global memory]

    A --> B
    B --> C
    C --> D
```

Explanation: Each block performs a local reduction, writes partial results to global memory, then grid sync ensures correctness before block 0 does a final pass.

### Diagram 2: Hierarchical Partitioning

```mermaid
flowchart LR
    subgraph Block
    direction TB
    T1[Tiled Partition (e.g., warp-level group)]
    T2[Perform local reduction within tile]
    T3[Block-level sync and partial result]
    end

    subgraph Grid
    direction TB
    G1[global partial sums in device memory]
    G2[grid.sync() => all partial sums ready]
    G3[final pass by block 0]
    end
```

Explanation: This diagram shows a hierarchical approach, where warp-level or tile-level cooperative groups help reduce data within a block, then grid-level synchronization coordinates the final pass.

---

## 8. References & Further Reading

- [CUDA C Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [NVIDIA Developer Blog – Cooperative Groups Tutorials](https://developer.nvidia.com/blog/cooperative-groups/)
- [Nsight Compute – Analyzing Multi-Block Kernels](https://developer.nvidia.com/nsight-compute)

---

## 9. Conclusion

Collaborative grouping techniques enable advanced concurrency patterns like multi-block reductions and grid-level synchronization. By carefully partitioning your kernel’s data processing and ensuring hardware support for grid-wide operations, you can efficiently aggregate large data sets without multiple kernel launches or complex atomic operations. However, these features require careful planning, as not all GPUs support them, and improper synchronization can degrade performance or produce incorrect results.

---

## 10. Next Steps

1. **Check GPU Attributes**: Implement logic to confirm that your GPU supports cooperative launch and grid-level sync.  
2. **Experiment**: Try rewriting multi-block reductions or scans using cooperative groups to see if you can streamline your code and reduce kernel launches.  
3. **Profile**: Use Nsight Systems or Nsight Compute to verify concurrency and measure performance improvements.  
4. **Tiled Groups**: Explore warp-level or sub-block partitions for specialized HPC tasks (e.g., advanced matrix operations).  
5. **Grid Sync**: Apply grid-level sync sparingly to avoid overhead, ensuring it solves a clear concurrency challenge.

```
