# Day 33: Intro to CUDA Graphs

CUDA Graphs provide a way to capture and reuse a sequence of CUDA operations (kernels, memory copies, etc.) as a single executable graph. By converting a sequence of operations into a graph, you can reduce launch overhead and potentially improve performance, especially for workloads with fixed execution patterns. However, mistakes during graph capture (such as failing to capture all necessary operations or capturing operations in an unintended order) can lead to unexpected results.

In this lesson, we will:
- Introduce CUDA Graphs and explain their benefits.
- Convert a simple kernel sequence into a CUDA Graph.
- Measure performance differences between traditional kernel launches and graph launches.
- Discuss common pitfalls and best practices for graph capture.

**References:**
- [CUDA C Programming Guide – Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphs)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Introduction to CUDA Graphs](#2-introduction-to-cuda-graphs)
3. [Practical Exercise: Converting a Kernel Sequence into a CUDA Graph](#3-practical-exercise-converting-a-kernel-sequence-into-a-cuda-graph)
   - [a) Standard Kernel Launch Sequence](#a-standard-kernel-launch-sequence)
   - [b) Capturing the Kernel Sequence as a CUDA Graph](#b-capturing-the-kernel-sequence-as-a-cuda-graph)
   - [c) Instantiating and Launching the CUDA Graph](#c-instantiating-and-launching-the-cuda-graph)
   - [d) Performance Measurement and Comparison](#d-performance-measurement-and-comparison)
4. [Common Debugging Pitfalls and Best Practices](#4-common-debugging-pitfalls-and-best-practices)
5. [Conceptual Diagrams](#5-conceptual-diagrams)
6. [References & Further Reading](#6-references--further-reading)
7. [Conclusion](#7-conclusion)
8. [Next Steps](#8-next-steps)

---

## 1. Overview

Traditionally, each kernel launch and memory copy in CUDA incurs overhead. CUDA Graphs allow you to capture a sequence of operations into a single executable unit, reducing the launch overhead and improving performance when the sequence is repeated multiple times. This is especially beneficial for workloads with a fixed control flow.

---

## 2. Introduction to CUDA Graphs

- **What Are CUDA Graphs?**  
  CUDA Graphs allow you to capture and then instantiate a graph that represents a sequence of CUDA operations. Once instantiated, the graph can be launched with very low overhead compared to individual kernel launches.

- **Benefits:**  
  - Reduced launch overhead.
  - Better performance for repeated executions of fixed operation sequences.
  - Simplified management of complex dependencies among kernels and memory operations.

- **Workflow Overview:**  
  1. **Begin Capture:** Start capturing CUDA operations into a stream.
  2. **Launch Operations:** Perform your normal CUDA operations (kernel launches, memory copies, etc.).
  3. **End Capture:** End the capture to produce a `cudaGraph_t`.
  4. **Instantiate Graph:** Convert the graph to an executable form (`cudaGraphExec_t`).
  5. **Launch Graph:** Run the graph multiple times with minimal overhead.

---

## 3. Practical Exercise: Converting a Kernel Sequence into a CUDA Graph

We will create a simple pipeline that performs two operations:
- **Kernel A:** Vector addition.
- **Kernel B:** Scaling of the resulting vector.

We will first implement the standard kernel launch sequence, then capture this sequence into a CUDA graph, and finally compare the performance.

### a) Standard Kernel Launch Sequence

```cpp
// standardKernelSequence.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel A: Vector Addition.
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel B: Vector Scaling.
__global__ void vectorScaleKernel(float *C, float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] *= scale;
    }
}

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    int N = 1 << 20;  // 1M elements.
    size_t size = N * sizeof(float);

    // Allocate host memory.
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors.
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy data from host to device.
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel A (Vector Addition).
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch Kernel B (Vector Scaling with a scale factor of 2.0).
    vectorScaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, 2.0f, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host.
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // (Optional) Print first 10 elements.
    printf("Standard Kernel Sequence Results (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Cleanup.
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```

*Comments:*
- This code demonstrates the traditional sequential launch of two kernels.
- Kernel A performs vector addition, and Kernel B scales the result.
- Host memory is allocated and initialized, data is copied to the device, kernels are launched sequentially, and the result is copied back.

---

### b) Capturing the Kernel Sequence as a CUDA Graph

We now convert the above kernel sequence into a CUDA graph to reduce launch overhead.

```cpp
// cudaGraphVectorPipeline.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel A: Vector Addition.
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel B: Vector Scaling.
__global__ void vectorScaleKernel(float *C, float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] *= scale;
    }
}

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    int N = 1 << 20;  // 1M elements.
    size_t size = N * sizeof(float);

    // Allocate host memory.
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors.
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy data from host to device.
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create a stream for graph capture.
    cudaStream_t captureStream;
    CUDA_CHECK(cudaStreamCreate(&captureStream));

    // Begin graph capture.
    CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));

    // Launch Kernel A (Vector Addition) in the capture stream.
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, captureStream>>>(d_A, d_B, d_C, N);

    // Launch Kernel B (Vector Scaling) in the capture stream.
    vectorScaleKernel<<<blocksPerGrid, threadsPerBlock, 0, captureStream>>>(d_C, 2.0f, N);

    // End graph capture.
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(captureStream, &graph));

    // Instantiate the graph.
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    // Launch the graph.
    CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float graphTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graphTime, start, stop));
    printf("CUDA Graph Execution Time: %f ms\n", graphTime);

    // Copy the result back to host.
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification.
    printf("Results from CUDA Graph (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Cleanup: Destroy graph, free device memory, free host memory, destroy stream and events.
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaStreamDestroy(captureStream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```

**Detailed Comments:**
- **Graph Capture:**  
  We create a stream (`captureStream`) and begin capturing operations using `cudaStreamBeginCapture()`.
- **Kernel Launches:**  
  Two kernels (vector addition and scaling) are launched in the capture stream.
- **Graph Instantiation:**  
  The captured operations are converted into a CUDA graph using `cudaStreamEndCapture()`, then instantiated with `cudaGraphInstantiate()`.
- **Graph Launch:**  
  The graph is launched using `cudaGraphLaunch()`, and CUDA events measure the execution time.
- **Verification:**  
  The result is copied back to host memory and printed for verification.
- **Cleanup:**  
  All resources (graph, device memory, stream, events) are destroyed and freed.

### Conceptual Diagram for Question 11:

```mermaid
flowchart TD
    A[Host: Allocate and Initialize Input Vectors]
    B[Allocate Device Memory for A, B, C]
    C[Copy Data from Host to Device]
    D[Create Capture Stream for Graph]
    E[Begin Graph Capture (cudaStreamBeginCapture)]
    F[Launch Kernel A (Vector Addition)]
    G[Launch Kernel B (Vector Scaling)]
    H[End Graph Capture to obtain CUDA Graph]
    I[Instantiate CUDA Graph (cudaGraphInstantiate)]
    J[Record CUDA Events (Start/Stop)]
    K[Launch Graph (cudaGraphLaunch)]
    L[Synchronize and Measure Execution Time]
    M[Copy Result from Device to Host]
    N[Verify Results]
    O[Cleanup: Destroy Graph, Free Memory, Destroy Stream/Events]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
```

*Explanation:*  
- The diagram shows the complete workflow for capturing a sequence of CUDA operations as a graph.
- It details the allocation of memory, graph capture, kernel launches, graph instantiation, and graph execution.
- Timing is measured using CUDA events, and the result is verified before cleanup.

---

## 4. Common Debugging Pitfalls and Best Practices

| **Pitfall**                                       | **Solution**                                                  |
|---------------------------------------------------|---------------------------------------------------------------|
| Not capturing all necessary operations           | Ensure that all dependent kernels and memory operations are launched within the capture stream. |
| Failing to end capture properly                    | Always call `cudaStreamEndCapture()` to obtain the graph; otherwise, the graph will be incomplete. |
| Not instantiating the graph correctly              | Use `cudaGraphInstantiate()` and check for errors.            |
| Inconsistent launch configurations between normal launches and graph capture | Maintain the same grid and block dimensions when capturing and launching the graph. |
| Forgetting to destroy the graph and free resources | Always clean up using `cudaGraphDestroy()` and `cudaGraphExecDestroy()`. |

---

## 5. References & Further Reading

1. **CUDA C Programming Guide – Graphs**  
   [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphs)
2. **CUDA C Best Practices Guide – Graphs**  
   [CUDA Best Practices: Graphs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
3. **NVIDIA CUDA Samples – Graphs**  
   [NVIDIA CUDA Graph Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html)
4. **NVIDIA NSight Systems Documentation**  
   [NVIDIA NSight Systems](https://docs.nvidia.com/nsight-systems/)
5. **"Programming Massively Parallel Processors: A Hands-on Approach" by David B. Kirk and Wen-mei W. Hwu**

---

## 6. Conclusion

In Day 33, we explored CUDA Graphs:
- We learned how to capture a sequence of CUDA operations into a graph.
- We converted a kernel sequence (vector addition followed by scaling) into a CUDA graph.
- We instantiated and launched the graph, measuring performance using CUDA events.
- We discussed common pitfalls in graph capture and instantiation.
- Detailed code examples with inline comments and conceptual diagrams illustrate the complete workflow.

---

## 7. Next Steps

- **Experiment with More Complex Graphs:**  
  Capture graphs that involve multiple streams, memory copies, and kernels.
- **Profile Graph Execution:**  
  Use NVIDIA NSight Systems to compare graph launch overhead against traditional kernel launches.
- **Integrate Graphs into Larger Applications:**  
  Convert a complete inference pipeline or image processing pipeline into a CUDA graph.
- **Optimize Graph Capture:**  
  Experiment with different capture modes and parameters to minimize overhead and maximize performance.

Happy CUDA coding, and continue exploring the power of CUDA Graphs for efficient GPU execution!
```

```
# Day 33: Intro to CUDA Graphs – Extended Explanation with Conceptual Diagrams

In **Day 33**, we introduced **CUDA Graphs**, a feature that allows you to capture and reuse a sequence of CUDA operations (kernels, memory copies, etc.) as a single executable unit. By capturing a workflow into a graph, we can reduce launch overhead and potentially improve performance—especially in repetitive scenarios. Below, we provide an extended explanation of the CUDA Graph workflow along with more detailed **conceptual diagrams**.

---

## Table of Contents
1. [Overview](#1-overview)
2. [What Are CUDA Graphs?](#2-what-are-cuda-graphs)
3. [Conceptual Diagrams](#3-conceptual-diagrams)
   - [a) Diagram: Traditional Sequential Launch vs. Graph Launch](#a-diagram-traditional-sequential-launch-vs-graph-launch)
   - [b) Diagram: Detailed CUDA Graph Workflow](#b-diagram-detailed-cuda-graph-workflow)
4. [Practical Example Recap](#4-practical-example-recap)
5. [Common Pitfalls & Best Practices](#5-common-pitfalls--best-practices)
6. [References & Further Reading](#6-references--further-reading)
7. [Conclusion](#7-conclusion)
8. [Next Steps](#8-next-steps)

---

## 1. Overview

Traditionally, each CUDA operation (kernel launch, memory copy, etc.) is invoked individually, incurring some overhead per launch. **CUDA Graphs** allow you to capture a series of these operations into a single graph, which can then be **instantiated** and **launched** with significantly lower overhead—particularly when the sequence of operations is repeated many times.

---

## 2. What Are CUDA Graphs?

### Core Ideas
1. **Capture Phase**: Operations within a stream (kernels, copies, etc.) are captured into a `cudaGraph_t`.
2. **Instantiation**: The captured graph is turned into an executable form `cudaGraphExec_t` via `cudaGraphInstantiate()`.
3. **Launch**: The executable graph is launched (with minimal overhead) via `cudaGraphLaunch()`.

### Use Cases
- **Repetitive Workloads**: Running the same sequence of kernels repeatedly.
- **Complex Pipelines**: Where multiple kernels, memory copies, and streams form a static dependency graph.
- **Performance Optimization**: Reducing per-launch overhead can yield significant speedups when launching many kernels.

---

## 3. Conceptual Diagrams

Below are two conceptual diagrams that illustrate how **CUDA Graphs** compare to the traditional sequential launch method, and a detailed breakdown of how a typical CUDA Graph workflow looks under the hood.

### a) Diagram: Traditional Sequential Launch vs. Graph Launch

```mermaid
flowchart LR
    subgraph Traditional Sequential Launch
    A[Host: Kernel 1 Launch] --> B[Overhead]
    B --> C[Host: Kernel 2 Launch] --> D[Overhead]
    D --> E[Host: Kernel 3 Launch] --> F[Overhead]
    end

    subgraph CUDA Graph Launch
    G[Host: Create Graph & Capture Ops (Once)]
    G --> H[Graph Instantiation]
    H --> I[Single Graph Launch => All Kernels]
end
```

**Explanation:**
- **Traditional Sequential Launch**: Each kernel launch is issued by the host, incurring overhead multiple times.
- **CUDA Graph Launch**: All kernels and operations are captured once and then launched in a single graph invocation, reducing overhead significantly.

---

### b) Diagram: Detailed CUDA Graph Workflow

Below is a more **detailed** breakdown of how the CUDA Graph workflow is executed from an application perspective:

```mermaid
flowchart TD
    A[Host: Allocate Host & Device Memory]
    B[Host: Create or Reuse Stream (CaptureStream)]
    C[Begin Graph Capture: cudaStreamBeginCapture]
    D[Launch Ops in CaptureStream (Kernels, Mem Copies)]
    E[End Graph Capture: cudaStreamEndCapture => cudaGraph_t]
    F[Instantiate Graph: cudaGraphInstantiate => cudaGraphExec_t]
    G[Launch Graph: cudaGraphLaunch => Runs All Captured Ops]
    H[Host: Copy Results Back, Synchronize, Verify]
    I[Optionally Repeat Launch => Minimal Overhead]
    J[Cleanup: cudaGraphDestroy, Free Memory]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
```

**Explanation:**
1. **Capture**: The host issues operations in a specific stream while capture is enabled.
2. **Graph Creation**: Once capture ends, a `cudaGraph_t` object is produced.
3. **Instantiation**: The graph is turned into an executable graph (`cudaGraphExec_t`).
4. **Launch**: The host launches the graph, which executes the captured operations.
5. **Verification & Reuse**: Results are copied back to the host, and if needed, the graph can be launched multiple times with minimal overhead.
6. **Cleanup**: Resources are properly destroyed and freed.

---

## 4. Practical Example Recap

In the *practical example* from Day 33 (shown in the previous code snippets), we:

1. Created a **capture stream**.
2. Began capture using `cudaStreamBeginCapture()`.
3. Launched two kernels (vectorAddKernel and vectorScaleKernel) in the capture stream.
4. Ended capture to obtain a `cudaGraph_t`.
5. **Instantiated** the graph into a `cudaGraphExec_t`.
6. Measured **performance** by launching the graph and timing it with CUDA events.

By comparing the **traditional sequential** approach vs. **graph-based** approach, you can observe the differences in overhead and total runtime.

---

## 5. Common Debugging Pitfalls & Best Practices

| **Pitfall**                                                  | **Solution**                                                                                               |
|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **Not capturing all required operations**                    | Ensure that every operation (kernel launch, memory copy) that should be part of the graph is launched in the capture stream. |
| **Forgetting to end capture**                                | Always call `cudaStreamEndCapture()` once your sequence of operations is complete.                         |
| **Incorrect graph instantiation**                            | Use `cudaGraphInstantiate()` and handle errors carefully.                                                 |
| **Not reusing the graph**                                    | The primary performance benefit comes from reusing the instantiated graph multiple times.                  |
| **Resource leaks**                                           | Destroy the graph (`cudaGraphDestroy()`) and the instantiated graph (`cudaGraphExecDestroy()`) once done.  |

---

## 6. References & Further Reading

1. **CUDA C Programming Guide – Graphs**  
   [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphs)
2. **CUDA C Best Practices Guide – Graphs**  
   [CUDA Best Practices: Graphs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
3. **NVIDIA CUDA Samples – Graphs**  
   [NVIDIA CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html)
4. **NVIDIA NSight Systems**  
   [NVIDIA NSight Systems](https://docs.nvidia.com/nsight-systems/)
5. **Programming Massively Parallel Processors: A Hands-on Approach** by David B. Kirk and Wen-mei W. Hwu

---

## 7. Conclusion

In **Day 33**—expanded with detailed **conceptual diagrams**—we have learned:

1. The basics of **CUDA Graphs**: capturing, instantiating, and launching.
2. How to **convert a sequence of kernels** and memory operations into a graph.
3. The **benefits of reduced launch overhead** when reusing the same sequence repeatedly.
4. The typical **workflow** of capturing operations in a stream and converting them into an executable graph.
5. **Common pitfalls** such as incomplete capture, forgetting to end capture, or failing to properly instantiate the graph.

---

## 8. Next Steps

- **Experiment with Larger Graphs**: Capture multiple kernels in multiple streams, including memory copies, to build complex workflows.
- **Profile with Nsight**: Use NVIDIA Nsight Systems to visualize how graphs reduce overhead and concurrency.
- **Integrate into Real Applications**: Replace repetitive kernel sequences in your production code with CUDA Graphs to benchmark performance gains.
- **Explore Graph Updates**: Investigate advanced features like updating a graph without recreating it from scratch (e.g., `cudaGraphExecUpdate()`).

Happy CUDA coding, and enjoy the performance benefits of CUDA Graphs!
```
