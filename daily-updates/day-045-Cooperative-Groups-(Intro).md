# Day 45: Cooperative Groups (Intro)

**Objective:**  
Explore **Cooperative Groups** in CUDA, which provide more flexible synchronization models than traditional block- or warp-level sync. Using cooperative groups, you can form custom groups of threads—like an entire grid, multiple blocks, or subsets of threads within a block—to coordinate computation. This enables more advanced parallel patterns. However, not all GPUs support advanced cooperative group features (like device-wide sync) in older architectures.

**Key Reference**:  
- [CUDA C Programming Guide – “Cooperative Groups”](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)

---

## Table of Contents

1. [Overview](#1-overview)  
2. [What Are Cooperative Groups?](#2-what-are-cooperative-groups)  
3. [Types of Cooperative Groups](#3-types-of-cooperative-groups)  
4. [Practical Example: Grouping Threads & Synchronization](#4-practical-example-grouping-threads--synchronization)  
   - [a) Code Snippet](#a-code-snippet)  
   - [b) Observing Group Sync](#b-observing-group-sync)  
5. [Conceptual Diagrams (Lots!)](#5-conceptual-diagrams-lots)  
   - [Diagram 1: Cooperative Group Hierarchy](#diagram-1-cooperative-group-hierarchy)  
   - [Diagram 2: Block-Level Group vs. Thread-Group Subsets](#diagram-2-block-level-group-vs-thread-group-subsets)  
   - [Diagram 3: Grid-Synchronization Flow (If Supported)](#diagram-3-grid-synchronization-flow-if-supported)  
6. [Common Pitfalls & Best Practices](#6-common-pitfalls--best-practices)  
7. [References & Further Reading](#7-references--further-reading)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)

---

## 1. Overview

Traditional CUDA synchronization occurs at:
- **Warp-level** automatically (execution in lockstep, though partial warp sync is complex).
- **Block-level** with `__syncthreads()`.
- **Device** or **grid** level** only** from the host side (e.g., kernel completion or `cudaDeviceSynchronize()`).

**Cooperative Groups** extends synchronization scopes:
- You can create **sub-block thread groups** (like tile groups).
- Potentially entire **grid groups** (for kernels launched in “cooperative” mode on supported GPUs).
- Use simpler, more flexible sync APIs (like `group.sync()`) and query group membership with a high-level API.

**However**:
- Full grid-level sync from device code requires hardware that supports **cooperative launch** and certain environment constraints.  
- If not supported, fallback or partial usage is needed.

---

## 2. What Are Cooperative Groups?

A **cooperative group** is a set of threads that can coordinate via advanced synchronization or gather thread indices in a more flexible manner. The library (`#include <cooperative_groups.h>`) defines classes like:

- `cooperative_groups::thread_block`: all threads in a block.  
- `cooperative_groups::thread_tile<tileSize>`: a subgroup of a block.  
- `cooperative_groups::grid_group`: if the entire grid is cooperating.  

**APIs**:
- `auto g = this_thread_block();` // block-level group, then `g.sync()`.  
- `auto tile = tiled_partition<16>(g);` // subdivide block into tiles of 16 threads, tile.sync().  
- `auto grid = this_grid();` // grid group for device-wide sync (special launch config needed).

---

## 3. Types of Cooperative Groups

1. **Thread Block Group** (`this_thread_block()`): Equivalent to blockDim.x * blockDim.y * blockDim.z threads. You can call `group.sync()` instead of `__syncthreads()`.  
2. **Tile Partition**: Subdivide a block into smaller tile groups (like warp- or half-warp sized). E.g. `thread_tile<32>` for warp-level sync.  
3. **Grid Group** (`this_grid()`): Potentially includes *all threads in the kernel launch*, so you can do a grid-wide sync in device code. Only valid on GPUs that support it and if the kernel is launched in a cooperative manner (host sets up “cooperative launch” if possible).  

---

## 4. Practical Example: Grouping Threads & Synchronization

### a) Code Snippet

```cpp
/**** day45_CooperativeGroups.cu ****/
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
namespace cg = cooperative_groups;

__global__ void tileReduceKernel(const float *input, float *output, int N) {
    // get block group
    cg::thread_block block = cg::this_thread_block();
    // subdivide block into tiles of 16 threads
    auto tile16 = cg::tiled_partition<16>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx<N) ? input[idx] : 0.0f;

    // do partial reduce in tile
    // each tile has tileSize=16, so local tile sync
    for(int offset=tile16.size()/2; offset>0; offset>>=1){
        val += tile16.shfl_down(val, offset);
    }

    // tile leader writes partial sum to shared mem or block-level array
    __shared__ float sdata[256]; // blockDim.x <=256 for demonstration
    int tid = threadIdx.x;
    if(tile16.thread_rank() ==0){
        sdata[tid] = val; // each tile leader writes
    }
    block.sync();

    // now block-level reduce the partial sums from each tile
    // e.g. if blockDim=64, that’s 4 tiles
    if(tid<16){
        float sum = sdata[tid];
        // do a final partial sum among tile leaders
        // for simplicity, each tile leader wrote sdata[tile*16], we need to loop if needed
        // omitted for brevity
        output[blockIdx.x] = sum; // store block sum
    }
}

int main(){
    int N=256;
    size_t size= N*sizeof(float);
    float *h_in=(float*)malloc(size);
    for(int i=0;i<N;i++){
        h_in[i]=(float)(rand()%100);
    }
    float *d_in,*d_out;
    cudaMalloc(&d_in,size);
    cudaMalloc(&d_out, (N/64)*sizeof(float)); // partial sums
    cudaMemcpy(d_in,h_in,size,cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid((N+block.x-1)/block.x);

    tileReduceKernel<<<grid, block>>>(d_in,d_out,N);
    cudaDeviceSynchronize();

    // Then reduce partial sums from d_out on host or in another kernel
    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

**Explanation**:  
- We use `cooperative_groups::this_thread_block()` to get block group, then `tiled_partition<16>` for sub-block synchronization.  
- `tile16.sync()` is implied with `shfl_down()`, but you can also do `tile16.sync()` explicitly if needed.  
- The final partial sums can be done at the block group or grid group level if you support grid-level sync.

### b) Observing Group Sync

If you replaced sub-block sync with `__syncthreads()`, you might lose the fine-grained tile approach. The advantage is you can easily manage multiple tile subgroups in a block.

---

## 5. Conceptual Diagrams (Lots!)

### Diagram 1: Cooperative Group Hierarchy

```mermaid
flowchart TD
    A[Grid Group: this_grid()] --> B[Block Group: this_thread_block()]
    B --> C[Tile Group <tileSize>]
    C --> D[Individual threads]
```

*Explanation*:
- Cooperative groups provide nested grouping: you can subdivide a block into tiles or treat entire grid as a group (if supported).

---

### Diagram 2: Block-Level Group vs. Thread-Group Subsets

```mermaid
flowchart LR
    subgraph Block of 64 threads
    direction TB
    A[thread_block group (64 threads)]
    B[tile_partition<16>] --> B1[tile0(16 threads)]
    B --> B2[tile1(16 threads)]
    B --> B3[tile2(16 threads)]
    B --> B4[tile3(16 threads)]
    end
```

*Explanation*:
- You can partition a block of 64 threads into four 16-thread tiles, each tile can `tile.sync()`.

---

### Diagram 3: Grid-Synchronization Flow (If Supported)

```mermaid
flowchart TD
    A[grid_group all threads in entire kernel] --> B[grid.sync()]
    B --> C[All threads proceed after device-wide sync]
```

*Explanation*:
- For a kernel launched in “cooperative mode,” `this_grid()` can represent all threads, letting them perform a grid-wide sync. This is only possible on certain architectures and requires specific launch parameters.

---

## 6. Common Pitfalls & Best Practices

1. **GPU Support**  
   - Not all GPUs or driver environments support **grid-group** synchronization (kernel must be launched cooperatively).  
2. **Performance Overhead**  
   - Overusing tile partitions or group sync can cause frequent synchronization => overhead.  
3. **Large Warps**  
   - Some tile partitions may or may not align with warp boundaries, leading to partial warp usage. Carefully handle that.  
4. **API Availability**  
   - Must include `<cooperative_groups.h>`, ensure compilation with appropriate flags if using advanced features.  
5. **Grid Sync**  
   - Use it carefully. A device-wide synchronization can drastically reduce concurrency if used frequently.

---

## 7. References & Further Reading

1. **CUDA C Programming Guide – Cooperative Groups**  
   [Documentation Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)  
2. **NVIDIA Developer Blog** – Articles on cooperative groups usage, advanced sync patterns.  
3. **Programming Massively Parallel Processors** – Chapter on cooperative groups & new synchronization scopes.

---

## 8. Conclusion

**Day 45** introduces **Cooperative Groups**:
- They provide a more flexible synchronization scope than `__syncthreads()` or warp-level automatic lockstep.
- You can form tile partitions within a block (`tiled_partition<tileSize>`) or even grid-level groups for full device sync if the GPU supports it.
- This can enable advanced parallel patterns or hierarchical designs, but must be used carefully to avoid overhead or rely on hardware features not always present.

**Key Takeaway**:  
Cooperative groups unify advanced sync mechanisms and partitioning. They can simplify sub-block tile synchronization or let an entire grid coordinate from within device code. Always confirm **hardware** and **driver** support for the full range of cooperative group features.

---

## 9. Next Steps

- **Practice**: Partition blocks into multiple tile groups for specialized computations.  
- **Measure**: Evaluate overhead of group sync vs. standard `__syncthreads()`.  
- **Explore**: If your GPU supports **cooperative launch**, attempt a kernel that uses `this_grid().sync()`. Compare performance vs. host-level synchronization.  
- **Integrate**: Combine with warp-level intrinsics or multi-stream concurrency for even more advanced GPU workflows.
```
