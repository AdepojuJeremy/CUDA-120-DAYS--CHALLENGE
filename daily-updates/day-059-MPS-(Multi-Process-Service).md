# Day 59: NVIDIA Multi-Process Service (MPS)

**Objective:**  
On Day 59, we focus on the **NVIDIA Multi-Process Service (MPS)**—a feature that allows multiple processes to share a single GPU context to improve utilization and throughput in certain multi-process environments. By consolidating different processes into a single GPU context, MPS can reduce context switch overhead and allow more effective concurrency on capable GPUs. However, oversubscription of the GPU through MPS can lead to resource contention and reduced performance if not managed carefully.

**Key Reference:**  
- [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html)

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Why Use MPS?](#2-why-use-mps)  
3. [Setting Up MPS](#3-setting-up-mps)  
   - [a) System Requirements & Limitations](#a-system-requirements--limitations)  
   - [b) Enabling MPS Daemon](#b-enabling-mps-daemon)  
   - [c) Launching CUDA Applications under MPS](#c-launching-cuda-applications-under-mps)  
4. [Concurrency & Resource Sharing](#4-concurrency--resource-sharing)  
   - [a) Reduced Context Switch Overhead](#a-reduced-context-switch-overhead)  
   - [b) Possible Resource Contention](#b-possible-resource-contention)  
5. [Practical Example: Two Processes Sharing a GPU](#5-practical-example-two-processes-sharing-a-gpu)  
   - [a) Sample Code for MPS Testing (Process 1 & Process 2)](#a-sample-code-for-mps-testing-process-1--process-2)  
   - [b) Running Under MPS](#b-running-under-mps)  
6. [Mermaid Diagram: MPS Workflow](#6-mermaid-diagram-mps-workflow)  
7. [Common Pitfalls & Best Practices](#7-common-pitfalls--best-practices)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)

---

## 1. Overview

By default, each CUDA application spawns its own GPU context. When multiple processes run on the same GPU, context switching can degrade performance. **MPS (Multi-Process Service)** consolidates multiple processes into a single GPU context, allowing them to share GPU resources more efficiently. This approach can:
- Improve concurrency and overall GPU utilization.
- Reduce scheduling overhead if processes frequently submit kernels.

However, the GPU has finite resources. If too many processes oversubscribe it, contention can result in lower performance for all processes. Balancing the number of processes and resource usage is key to achieving performance gains under MPS.

---

## 2. Why Use MPS?

1. **Reduced Context Overheads:**  
   MPS merges multiple processes into a unified context, reducing the overhead of context switches that can degrade latency and throughput.

2. **Enhanced Concurrency:**  
   If multiple processes are running kernels that can overlap, MPS may enable better scheduling and concurrency on the GPU.

3. **Streamlined Management:**  
   From an administrative perspective, MPS can help cluster environments where multiple users share a single GPU.  

**Constraints:** MPS is typically used in HPC or server environments where multiple short-lived or medium-lived processes run concurrently and can benefit from more straightforward scheduling.

---

## 3. Setting Up MPS

### a) System Requirements & Limitations

- **Supported GPU Architecture:**  
  MPS requires modern NVIDIA GPUs (Kepler or later) that support concurrent execution.  
- **Driver & CUDA Version:**  
  Must have a compatible driver and CUDA toolkit that support MPS.

### b) Enabling MPS Daemon

1. **Stop** any existing GPU processes and unload modules if needed.  
2. **Enable** MPS by setting environment variables or using system daemons, e.g.:
   ```bash
   sudo nvidia-cuda-mps-control -d
   ```
   This starts the MPS daemon in the background.

3. **Verify** MPS is running:  
   ```bash
   echo get_server_list | nvidia-cuda-mps-control
   ```

### c) Launching CUDA Applications under MPS

1. **Set the environment variable** (depending on the OS/distribution):
   ```bash
   export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
   export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
   ```
2. **Launch** multiple processes that run CUDA code.  
3. **MPS** merges them into a single GPU context, reducing overhead.

---

## 4. Concurrency & Resource Sharing

### a) Reduced Context Switch Overhead
Under default configurations, each process has its own context. The GPU must swap contexts to handle different processes. **MPS** eliminates most context switches by unifying the processes into one context, potentially boosting concurrency.

### b) Possible Resource Contention
When multiple processes oversubscribe the GPU, all resources—registers, shared memory, SMs—are shared. If too many processes saturate the GPU, overall performance might degrade. Careful concurrency control is crucial.

---

## 5. Practical Example: Two Processes Sharing a GPU

### a) Sample Code for MPS Testing (Process 1 & Process 2)

Below is simplified code for two different processes that each run a small kernel. They run concurrently under MPS for demonstration. In a real scenario, you’d compile these as separate executables.

```cpp
// File: process1.cu
#include <cuda_runtime.h>
#include <stdio.h>
#define CUDA_CHECK(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while (0)

__global__ void dummyKernel1(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f;
    }
}

int main(){
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    dummyKernel1<<<(N+255)/256, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
    printf("Process 1 done.\n");
    return 0;
}

// File: process2.cu
#include <cuda_runtime.h>
#include <stdio.h>
#define CUDA_CHECK(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while (0)

__global__ void dummyKernel2(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main(){
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    dummyKernel2<<<(N+255)/256, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
    printf("Process 2 done.\n");
    return 0;
}
```

### b) Running Under MPS

1. **Start MPS Daemon:**
   ```bash
   sudo nvidia-cuda-mps-control -d
   ```
2. **Set environment variables** (if needed).  
3. **Run** both processes in separate terminals or via background tasks:
   ```bash
   ./process1 &
   ./process2 &
   ```
4. **Observe** that both processes run concurrently on the GPU through MPS. The GPU context is unified, reducing context switch overhead.

---

## 6. Mermaid Diagram: MPS Workflow

```mermaid
flowchart TD
    A[Start MPS Daemon (nvidia-cuda-mps-control)]
    B[Process1 (dummyKernel1) starts]
    C[Process2 (dummyKernel2) starts]
    D[MPS merges them into a single GPU context]
    E[GPU schedules kernels from both processes concurrently]
    F[Each process completes, logs out "done"]

    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
    E --> F
```

**Explanation:**  
1. MPS daemon is started.  
2. Two processes launch.  
3. MPS merges them, scheduling their kernels under a single GPU context.  
4. Each completes with “done”.

---

## 7. Common Pitfalls & Best Practices

- **Resource Oversubscription:**  
  If too many processes oversubscribe the GPU, all processes may experience slower performance.  
- **Driver/Toolkit Compatibility:**  
  MPS requires specific driver and toolkit versions. Check your environment.  
- **Debugging Complexity:**  
  With MPS, debugging multiple processes sharing one GPU context can be more complicated.  
- **Data Separation:**  
  Each process must manage its own device memory allocations. There is no direct inter-process memory sharing unless you enable Peer-to-Peer or use more advanced techniques.

---

## 8. Conclusion

**Day 59** covers **NVIDIA’s Multi-Process Service (MPS)** for more efficient GPU sharing among multiple processes. MPS consolidates GPU contexts, reducing overhead and improving concurrency. However, oversubscription can lead to resource contention. Balancing the number of processes, their concurrency levels, and GPU resources is essential for gaining the full benefits of MPS. By combining MPS with robust concurrency management (e.g., chunking, library-based routines), HPC systems can significantly improve GPU utilization.

---

## 9. Next Steps

1. **Experiment with MPS** in a multi-process environment:
   - Launch multiple CUDA processes, measure baseline performance, then enable MPS to see if concurrency improves.  
2. **Profile** with Nsight Systems or `cuda-gdb` to see concurrency patterns.  
3. **Resource Limits**:  
   - Monitor memory usage, concurrency levels, and SM utilization to ensure processes do not oversaturate the GPU.  
4. **Combine** MPS with multi-GPU strategies or advanced concurrency for HPC clusters.  
5. **Review** relevant sections in [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html) for advanced configuration options (e.g., setting per-GPU hardware limits, controlling scheduling policies).
```
