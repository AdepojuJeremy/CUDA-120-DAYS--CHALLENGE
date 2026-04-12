# Day 71: Advanced Streams & Overlapping

When working on **complex HPC pipelines** or **real-time data feeds**, leveraging **multiple streams** can enable the GPU to overlap various tasks (kernels, data transfers, even CPU tasks) to maximize concurrency. Day 71 focuses on **advanced usage of CUDA streams** and how to effectively chain multiple operations in a pipeline, carefully avoiding race conditions by synchronizing at critical points. We also explore how missing or incorrect synchronization can lead to silent data corruption or partial results, emphasizing the need for diligent concurrency management.

---

## Table of Contents
1. [Overview](#1-overview)  
2. [Why Overlapping Matters](#2-why-overlapping-matters)  
3. [Advanced Stream Concepts](#3-advanced-stream-concepts)  
   - [a) Concurrent Kernel Execution](#a-concurrent-kernel-execution)  
   - [b) Overlapping Data Transfers and Kernels](#b-overlapping-data-transfers-and-kernels)  
   - [c) Integrating CPU-Side Tasks](#c-integrating-cpu-side-tasks)  
4. [Implementation Approach](#4-implementation-approach)  
   - [a) Stream Creation and Event Usage](#a-stream-creation-and-event-usage)  
   - [b) Chaining Operations](#b-chaining-operations)  
5. [Code Example: Multi-Stream Overlap](#5-code-example-multi-stream-overlap)  
   - [Explanation & Comments](#explanation--comments)  
6. [Common Pitfalls & Synchronization Issues](#6-common-pitfalls--synchronization-issues)  
7. [Conceptual Diagram](#7-conceptual-diagram)  
8. [References & Further Reading](#8-references--further-reading)  
9. [Conclusion](#9-conclusion)  
10. [Next Steps](#10-next-steps)

---

## 1. Overview

By default, CUDA operations in a single stream are **serialized**. However, by creating **multiple streams**, you can overlap:
- **Kernel launches** so that multiple kernels run concurrently if the GPU’s hardware resources allow.
- **Host↔Device transfers** with kernel execution, hiding data transfer time behind GPU compute.
- **CPU tasks** in parallel with GPU tasks, if the CPU code doesn’t block or require the same data that’s in use by the GPU.

Yet concurrency is only beneficial if carefully planned. **Missing sync** calls, incorrect usage of events, or oversubscription can degrade performance or produce data races. This lesson digs deeper into advanced overlap scenarios that push GPU concurrency to its limits.

---

## 2. Why Overlapping Matters

- **Latency Hiding**: If a kernel in one stream is waiting on memory, the scheduler can run a kernel from another stream.  
- **Efficiency**: In real-time or batch pipelines, streaming input data while the GPU processes the previous batch can significantly boost throughput.  
- **CPU-GPU Parallelism**: The CPU can post-process the results from the last iteration while the GPU tackles the next iteration.

---

## 3. Advanced Stream Concepts

### a) Concurrent Kernel Execution
- Modern GPUs can schedule multiple kernels in parallel if resources remain available. Streams with independent data let the device run them simultaneously.

### b) Overlapping Data Transfers and Kernels
- By calling `cudaMemcpyAsync(..., streamX)`, you can copy data in parallel with kernel execution in other streams.  
- Use pinned host memory to maximize transfer bandwidth and maintain concurrency.

### c) Integrating CPU-Side Tasks
- The CPU can run computations or manage I/O while the GPU processes data. This synergy is especially valuable if your workflow is partially CPU-bound or requires specialized CPU tasks like compression or network I/O.

---

## 4. Implementation Approach

### a) Stream Creation and Event Usage
1. **Multiple Streams**: Create separate streams for data transfers, kernel A, kernel B, etc.  
2. **Events**: Use `cudaEventRecord()` to mark a stream’s completion point. Another stream or CPU can wait on that event via `cudaStreamWaitEvent()` or `cudaEventSynchronize()`, ensuring correct ordering.

### b) Chaining Operations
1. **Producer Stream**: Copies new data to the device and launches a production kernel.  
2. **Consumer Stream**: Waits on an event signifying the production kernel is done, then processes that data or merges partial results.  
3. **CPU Tasks**: Potentially run concurrently if your code does not require immediate results from the GPU or pinned memory.

---

## 5. Code Example: Multi-Stream Overlap

Below is a **multi-stream** snippet demonstrating how to overlap kernel execution, data transfers, and CPU tasks. We have:
1. Stream for **kernelA**.  
2. Stream for **data copy**.  
3. A CPU function that runs concurrently.  
4. Another stream for **kernelB** which depends on the result from **kernelA**.

```cpp
// File: multi_stream_overlap.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <thread>
#include <chrono>

// Simple kernel that increments each element
__global__ void kernelA(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f;
    }
}

// Another kernel that doubles each element
__global__ void kernelB(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

void cpuSideTask() {
    // Simulate a CPU task that runs concurrently with GPU
    printf("CPU task started...\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    printf("CPU task completed.\n");
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // Allocate pinned host buffer
    float* h_data;
    cudaMallocHost((void**)&h_data, size);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device buffer
    float* d_data;
    cudaMalloc(&d_data, size);

    // Create streams
    cudaStream_t streamA, streamCopy, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamCopy);
    cudaStreamCreate(&streamB);

    // 1) Asynchronous copy host -> device in streamCopy
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, streamCopy);

    // Launch a thread for CPU side task concurrently
    std::thread cpuThread(cpuSideTask);

    // Wait for copy in streamCopy to complete, then launch kernelA in streamA
    // Use an event to chain these streams
    cudaEvent_t copyDone;
    cudaEventCreate(&copyDone);
    cudaEventRecord(copyDone, streamCopy);
    cudaStreamWaitEvent(streamA, copyDone, 0);
    kernelA<<<(N+255)/256, 256, 0, streamA>>>(d_data, N);

    // Another event after kernelA completes
    cudaEvent_t kernelA_done;
    cudaEventCreate(&kernelA_done);
    cudaEventRecord(kernelA_done, streamA);

    // kernelB in streamB waits for kernelA_done
    cudaStreamWaitEvent(streamB, kernelA_done, 0);
    kernelB<<<(N+255)/256, 256, 0, streamB>>>(d_data, N);

    // Wait for all GPU ops
    cudaStreamSynchronize(streamCopy);
    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);

    // Join CPU thread
    cpuThread.join();

    // Copy result back in same or new stream (for brevity, do synchronous copy)
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    printf("Sample result: h_data[0] = %f\n", h_data[0]);

    // Cleanup
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamCopy);
    cudaStreamDestroy(streamB);
    cudaEventDestroy(copyDone);
    cudaEventDestroy(kernelA_done);
    return 0;
}
```

### Explanation & Comments

1. **Asynchronous Copy**: `cudaMemcpyAsync` in `streamCopy` loads data to the GPU while the CPU does a separate task.  
2. **CPU Thread**: A simple CPU function runs concurrently with GPU actions.  
3. **Event Chaining**:  
   - `copyDone` ensures kernelA waits for the data.  
   - `kernelA_done` ensures kernelB only starts after kernelA.  
4. **Streams**: `streamA` for kernelA, `streamB` for kernelB, `streamCopy` for data transfers.

---

## 6. Common Pitfalls & Synchronization Issues

- **Overlapping CPU Code**: The CPU might attempt to read pinned memory before the GPU finishes writing. Use events and synchronization if that data is needed by the CPU.  
- **Misused Streams**: Launching kernels that rely on the same data in separate streams without an event-based sync can cause data hazards.  
- **Oversubscription**: Creating too many concurrent streams can degrade performance if the GPU is saturated.

---

## 7. Conceptual Diagram

```mermaid
flowchart TD
    subgraph Host
    H1[Start CPU task in separate thread]
    end

    subgraph StreamCopy
    C1[cudaMemcpyAsync Host->Device]
    C2[Event copyDone]
    end

    subgraph StreamA
    A1[Wait event copyDone]
    A2[kernelA(d_data)]
    A3[Event kernelA_done]
    end

    subgraph StreamB
    B1[Wait event kernelA_done]
    B2[kernelB(d_data)]
    end

    H1 --- C1
    C1 --> C2 --> A1 --> A2 --> A3 --> B1 --> B2
```

**Explanation**:  
- StreamCopy transfers data in parallel with a CPU task.  
- kernelA in StreamA waits for `copyDone` event.  
- kernelB in StreamB waits for `kernelA_done` event, forming a pipeline.

---

## 8. References & Further Reading

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/) – Profiling concurrency timelines.  
- [CUDA C Programming Guide – Streams & Concurrency](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-kernel-execution)  
- [NVIDIA Developer Blog – Advanced Multi-Stream Techniques](https://developer.nvidia.com/blog)

---

## 9. Conclusion

**Day 71** spotlights **advanced stream usage** where multiple kernels, data transfers, and even CPU tasks overlap for maximal concurrency. Proper event chaining is essential to avoid data hazards. By dedicating separate streams to distinct operations (data copy, kernel stages), and optionally letting the CPU do concurrent tasks, HPC pipelines can hide latencies and boost throughput. Yet, the synergy among streams only succeeds if events are used to tie their data dependencies together.

---

## 10. Next Steps

1. **Profile Overlap**: Use **Nsight Systems** to confirm actual concurrency and identify potential serialization points.  
2. **Experiment**: Try rearranging streams for data staging, kernel chaining, or CPU tasks, measuring real speedups.  
3. **Refine**: If concurrency is not giving expected gains, check if the GPU or memory bus is saturated or if sync calls are too frequent.  
4. **Apply**: Extend multi-stream concurrency to multi-GPU or MPS scenarios for more advanced HPC pipelines.

```
