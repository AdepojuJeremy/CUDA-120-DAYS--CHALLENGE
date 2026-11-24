# Capstone Project #3: Multi-Stream Data Processing with Overlapped Transfers & Kernels for Real-Time Feeds

**Objective:**  
Design a multi-stream, real-time data processing pipeline where continuous data feeds are transferred to the GPU in chunks, processed by custom or library-based kernels, and then streamed back to the host (or consumed on the GPU) with minimal latency. This **Capstone Project #3** encapsulates techniques learned throughout our CUDA journey: concurrent data transfers, multi-stream overlapping, kernel concurrency, chunk processing, and robust design practices to handle real-time constraints.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Project Goals & Key Concepts](#2-project-goals--key-concepts)  
3. [High-Level Pipeline Design](#3-high-level-pipeline-design)  
4. [Detailed Implementation Steps](#4-detailed-implementation-steps)  
   - [a) Data Acquisition / Host Buffering](#a-data-acquisition--host-buffering)  
   - [b) Chunk-Based Transfer & Stream Management](#b-chunk-based-transfer--stream-management)  
   - [c) GPU Processing Kernels](#c-gpu-processing-kernels)  
   - [d) Output Consumption / Host Post-Processing](#d-output-consumption--host-post-processing)  
5. [Sample Code: Real-Time Multi-Stream Pipeline](#5-sample-code-real-time-multi-stream-pipeline)  
   - [a) Code Explanation](#a-code-explanation)  
   - [b) Full Example](#b-full-example)  
6. [Mermaid Diagram: Multi-Stream Overlap Logic](#6-mermaid-diagram-multi-stream-overlap-logic)  
7. [Performance Considerations & Tips](#7-performance-considerations--tips)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)

---

## 1. Overview

In real-time data processing scenarios—such as live analytics, video streams, or sensor-driven HPC tasks—**latency** and **throughput** are paramount. By employing multiple CUDA streams, you can overlap data transfers with GPU kernel execution, ensuring minimal idle time on both the host and device sides.

**Core Principles**:
- **Chunking**: Process data in small segments that can fit in GPU memory or be processed within a time slice.  
- **Multi-Stream Concurrency**: Overlap memory copies and kernel execution across different streams.  
- **Synchronization**: Use events or stream sync calls to coordinate chunk availability and processing stages.

This project integrates all these topics into a cohesive pipeline that continuously receives data, processes it on the GPU, and returns (or uses) the results in near real-time.

---

## 2. Project Goals & Key Concepts

1. **Continuous Data Feeds**: Emulate or connect to a real-time source of data (e.g., sensor streams, networked data feed).  
2. **Overlap Transfers & Kernels**: Exploit multi-stream concurrency to hide data transfer latency behind kernel execution.  
3. **Chunk-Based Processing**: Divide data into chunks suitable for GPU memory constraints or preferred kernel sizes.  
4. **Minimal Latency**: Leverage asynchronous APIs and efficient synchronization (cuda events, stream callbacks) to reduce overall latency.  
5. **Scalable Throughput**: Adapt chunk sizes and concurrency levels based on performance measurement to handle high data rates.

---

## 3. High-Level Pipeline Design

1. **Data Acquisition**: Host receives real-time chunks (e.g., from a socket, file stream, or sensor interface).  
2. **Chunk Buffering**: Each chunk is queued in host memory to be processed.  
3. **Streamed Transfer**: An asynchronous copy of each chunk to the GPU using `cudaMemcpyAsync()` in a dedicated stream.  
4. **Kernel Execution**: A custom kernel (or library routine) transforms / processes the chunk (e.g., filtering, transformations).  
5. **Overlap**: While GPU is processing chunk *n*, chunk *n+1* is being copied to the device. Meanwhile, processed chunk *n-1* might be copied back to the host.  
6. **Host Consumption**: The result is returned to the host for post-processing, display, or saving.  
7. **Performance Feedback**: Profiling and monitoring help adjust chunk size or concurrency to meet latency/throughput targets.

---

## 4. Detailed Implementation Steps

### a) Data Acquisition / Host Buffering
- **Real-Time Feeds**: Typically read in small increments (e.g., 1 MB or 4 MB segments) from a socket or pipeline.
- **Buffer Queues**: Maintain a queue of host buffers ready for GPU transfers, so the CPU can read next chunks while GPU processes current ones.

### b) Chunk-Based Transfer & Stream Management
- **Allocate pinned (page-locked) memory** for host buffers to reduce transfer latency.
- **Spawn multiple streams** to pipeline operations:
  - **Stream 1**: Transfer chunk *n* → GPU and launch kernel *n*.  
  - **Stream 2**: Transfer chunk *n+1* → GPU and launch kernel *n+1*.
- **Synchronization**: Use events to ensure chunk *n* is fully transferred before kernel execution and to signal host when kernel is done to retrieve results.

### c) GPU Processing Kernels
- A custom kernel might filter data, transform it, or aggregate it. Examples:
  - **Signal Processing**: A simple transform kernel applying a filter or scale.
  - **Reduction / Summaries**: Summarize chunk data for real-time stats.
- **Edge Cases**: If the last chunk is partial, ensure the kernel’s indexing logic does not read beyond the chunk.

### d) Output Consumption / Host Post-Processing
- After the kernel finishes, results can be:
  - **Asynchronously copied** to a host output buffer for immediate consumption.
  - **Left on the device** for subsequent GPU-based post-processing or library calls.

---

## 5. Sample Code: Real-Time Multi-Stream Pipeline

### a) Code Explanation

- We simulate real-time feed data using a loop that “generates” random chunks.
- Each chunk is queued for transfer to GPU memory in its own stream.
- A custom kernel processes the chunk.
- We overlap chunk *n* processing with chunk *n+1* transfer to the GPU.
- We store the processed results on the device and/or copy them back asynchronously for host consumption.

### b) Full Example

```cpp
// File: capstone_project_multi_stream.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA Error: %s (line %d)\n",         \
                cudaGetErrorString(err), __LINE__);           \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// Sample kernel: simple transform, e.g., scale + offset
__global__ void transformKernel(const float *in, float *out, int chunkSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunkSize) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

int main(){
    // Simulated real-time data feed parameters
    int totalFeedSize = 1 << 22; // 4 million floats
    int chunkSize = 1 << 20;     // 1 million floats per chunk
    int numChunks = (totalFeedSize + chunkSize - 1) / chunkSize;

    // Host buffers: pinned memory for better transfer performance
    float *h_in, *h_out;
    size_t chunkBytes = chunkSize * sizeof(float);
    CUDA_CHECK(cudaMallocHost((void**)&h_in, chunkBytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, chunkBytes));

    // Device memory for 2-chunk ping-pong approach
    float *d_in[2], *d_out[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc(&d_in[i], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_out[i], chunkBytes));
    }

    // Create 2 streams for overlapping
    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    // Simulation of real-time feed
    srand(time(NULL));
    for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        // Simulate reading from a feed: fill host pinned buffer with random data
        int currentChunkSize = ((chunkIdx + 1) * chunkSize > totalFeedSize)
                                ? (totalFeedSize - chunkIdx * chunkSize)
                                : chunkSize;
        for (int j = 0; j < currentChunkSize; j++) {
            h_in[j] = (float)(rand() % 100);
        }

        // Use ping-pong indexing: chunkIdx % 2 to pick device buffers
        int bufIndex = chunkIdx % 2;

        // Asynchronously copy chunk data to device
        CUDA_CHECK(cudaMemcpyAsync(d_in[bufIndex], h_in,
                                   currentChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream[bufIndex]));

        // Launch transformKernel in the same stream
        transformKernel<<<(currentChunkSize + threadsPerBlock - 1)/threadsPerBlock, 
                          threadsPerBlock, 0, stream[bufIndex]>>>(
                          d_in[bufIndex], d_out[bufIndex], currentChunkSize);

        // Optionally copy results back to host asynchronously
        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out[bufIndex],
                                   currentChunkSize * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream[bufIndex]));

        // (Optional) Use or process h_out data before next chunk
        // If real-time display or further host-level analysis is needed:
        // e.g. wait or do partial checks
        // For demonstration, let's not block here, we'll rely on stream sync if needed
    }

    // Wait for all streams to finish
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamSynchronize(stream[i]));
    }

    // Print sample
    printf("Last chunk processed: h_out[0] = %f\n", h_out[0]);

    // Cleanup
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));

    return 0;
}
```

**Explanation Highlights**:
- **Ping-Pong Buffers**: We use two sets of `d_in`/`d_out` arrays to overlap chunk processing.  
- **Multi-Stream**: `stream[0]` and `stream[1]` handle alternating chunks to enable concurrency.  
- **Real-Time Simulation**: For each chunk, we randomize data in `h_in`, copy to a device buffer, run a kernel, and optionally copy results back asynchronously.  
- **Final Sync**: We synchronize both streams at the end to ensure all chunks are processed before exiting.

---

## 6. Mermaid Diagram: Multi-Stream Overlap Logic

```mermaid
flowchart LR
    subgraph Stream0
      direction TB
      S0a[Copy chunkN => d_in[0]] --> K0[transformKernel(chunkN)] --> R0[Copy d_out[0] => h_out]
    end

    subgraph Stream1
      direction TB
      S1a[Copy chunkN+1 => d_in[1]] --> K1[transformKernel(chunkN+1)] --> R1[Copy d_out[1] => h_out]
    end

    Host[Host: fill h_in for chunkN, chunkN+1, ... in a loop]
    Host --> S0a
    Host --> S1a
    S0a --> K0
    S1a --> K1
    K0 --> R0
    K1 --> R1
```

**Explanation:**  
- The host provides chunk *N* to `stream[0]`, while chunk *N+1* can be processed by `stream[1]`, enabling concurrency.  
- Each stream asynchronously handles its chunk’s copy, kernel execution, and optional result transfer.

---

## 7. Performance Considerations & Tips

1. **Chunk Size Tuning**:  
   - Too large chunks might reduce concurrency.  
   - Too small chunks might increase overhead from frequent kernel launches and transfers.

2. **Pinned Memory**:  
   - Using pinned (page-locked) host memory can improve transfer speeds, vital for real-time feeds.

3. **Kernel Efficiency**:  
   - Optimize kernel to handle chunk edges properly and ensure minimal overhead for partial data.

4. **Overlapping**:  
   - Evaluate concurrency: ensure the GPU has enough bandwidth and SM resources to run multiple streams effectively.

5. **Scaling**:  
   - For extremely large data or multiple real-time streams, more advanced scheduling or partitioning across multiple GPUs might be necessary.

---

## 8. Conclusion

**Capstone Project #3** showcases a **multi-stream data processing pipeline** aimed at real-time feeds. By overlapping data transfers and kernel executions in separate streams, the GPU can process incoming chunks with minimal idle time, thereby reducing end-to-end latency and increasing throughput. Proper chunk management, pinned host memory usage, and kernel design are essential to handle large or continuous data sets effectively.

---

## 9. Next Steps

1. **Refine the Pipeline**:  
   - Adjust chunk size, concurrency level, and pinned memory usage for your data rate and GPU resources.
2. **Incorporate Libraries**:  
   - Use cuBLAS, cuFFT, or custom kernels for more complex transformations on each chunk.
3. **Extend for Real Input Streams**:  
   - Integrate actual data sources (e.g., network or sensor) instead of random data simulation.
4. **Profile**:  
   - Employ Nsight Systems to confirm overlapping data transfers and kernel computation.
5. **Scale Up**:  
   - Expand to multiple GPUs if single-GPU concurrency cannot handle the real-time data load.

**Happy Capstone coding and best of luck optimizing your multi-stream real-time pipelines!**
```
