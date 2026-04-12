# Day 49: Using Thrust for High-Level Operations

**Objective:**  
Explore **Thrust**, NVIDIA’s C++ template library that provides high-level operations (like transforms, sorts, reductions, scans) on host or device vectors. Thrust simplifies GPU programming by abstracting away many kernel details, letting you replace custom loops with composable algorithms. However, if you frequently move data between host and device, you can incur overhead, so structuring your code to keep data mostly on the device is best.

**Key Reference**:  
- [Thrust Library Documentation](https://thrust.github.io/)  
- [CUDA C Programming Guide – Thrust Chapter](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thrust)

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Why Thrust?](#2-why-thrust)  
3. [Core Thrust Data Structures](#3-core-thrust-data-structures)  
4. [Thrust Algorithms: Transforms, Sorts, Reductions](#4-thrust-algorithms-transforms-sorts-reductions)  
5. [Practical Example: Multiple Thrust Operations](#5-practical-example-multiple-thrust-operations)  
   - [a) Code Snippet & Explanation](#a-code-snippet--explanation)  
   - [b) Observing Device-Host Transfers](#b-observing-device-host-transfers)  
6. [Mermaid Diagrams](#6-mermaid-diagrams)  
   - [Diagram 1: High-Level Thrust Flow](#diagram-1-high-level-thrust-flow)  
   - [Diagram 2: Composition of Thrust Calls](#diagram-2-composition-of-thrust-calls)  
7. [Common Pitfalls & Best Practices](#7-common-pitfalls--best-practices)  
8. [References & Further Reading](#8-references--further-reading)  
9. [Conclusion](#9-conclusion)  
10. [Next Steps](#10-next-steps)

---

## 1. Overview

Thrust is a powerful, STL-like library for GPU (and CPU) computations. It offers:
- **Vectors**: `thrust::device_vector<T>` for storing data on the device, analogous to `std::vector` but residing in device memory.
- **Algorithms**: `thrust::transform`, `thrust::sort`, `thrust::reduce`, `thrust::inclusive_scan`, etc., all running on GPU if used with device vectors.
- **Functor**: A custom **lambda or struct** for transformations or predicate operations.

**Potential Gains**:
- Less boilerplate code than writing custom kernels.  
- Well-optimized for typical patterns (sort, reduce).  

**Potential Drawbacks**:
- If data repeatedly goes device ↔ host between operations, you lose performance.  
- For specialized or extremely advanced kernels, a custom approach might beat Thrust’s generic approach.

---

## 2. Why Thrust?

1. **High-Level Abstractions**: Instead of writing repetitive kernel loops, use a single line `thrust::transform()` or `thrust::reduce()`.  
2. **Reusability**: Compose multiple operations easily.  
3. **Portability**: Thrust can also run in a host backend if needed, though typically we target the device backend for CUDA acceleration.  
4. **Performance**: For standard patterns (sort, reduce, scan), Thrust is often near state-of-the-art, especially if you keep data on the device.

---

## 3. Core Thrust Data Structures

- **`thrust::host_vector<T>`**: Host-side container, akin to `std::vector<T>`.
- **`thrust::device_vector<T>`**: Device-side container. Operations on it run on GPU (assuming the default device backend).
- Converting or copying between them automatically triggers **host-device** transfers.

**Example**:
```cpp
thrust::device_vector<int> d_vec(10, 1); // all 1's on device
```

---

## 4. Thrust Algorithms: Transforms, Sorts, Reductions

1. **Transform**:  
   ```cpp
   thrust::transform(d_vec1.begin(), d_vec1.end(),
                     d_vec2.begin(),
                     d_out.begin(),
                     thrust::plus<int>());
   ```
   Summation of corresponding elements.

2. **Sort**:  
   ```cpp
   thrust::sort(d_vec.begin(), d_vec.end());
   ```
   Sort in ascending order. Or use `thrust::sort_by_key(...)` for key-value pairs.

3. **Reduce**:  
   ```cpp
   int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
   ```
   Sum all elements in device vector.

4. **Scan**:  
   ```cpp
   thrust::inclusive_scan(d_in.begin(), d_in.end(), d_out.begin());
   ```
   Prefix sum in one line.

---

## 5. Practical Example: Multiple Thrust Operations

### a) Code Snippet & Explanation

```cpp
/**** day49_ThrustExample.cu ****/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <iostream>

// A functor for transformation: multiply by factor
struct multiply_by_factor {
    float factor;
    multiply_by_factor(float f) : factor(f) {}
    __host__ __device__
    float operator()(const float &x) const {
        return x * factor;
    }
};

int main(){
    int N = 10;
    // 1) create host vector
    thrust::host_vector<float> h_vec(N);
    for(int i=0; i<N; i++){
        h_vec[i]=(float)(rand()%100);
    }

    // 2) transfer to device vector
    thrust::device_vector<float> d_vec = h_vec;

    // 3) transform => multiply each element by 2.5
    thrust::transform(d_vec.begin(), d_vec.end(),
                      d_vec.begin(), // in place
                      multiply_by_factor(2.5f));

    // 4) sort device vector
    thrust::sort(d_vec.begin(), d_vec.end());

    // 5) reduce to get sum
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    std::cout << "Sum after transform & sort= " << sum << std::endl;

    // copy result back to host to see sorted data
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    std::cout << "Sorted *2.5 data:\n";
    for(int i=0; i<N; i++){
        std::cout << h_vec[i] << " ";
    }
    std::cout<<"\n";

    return 0;
}
```

**Explanation**:  
1. We fill a **host_vector** randomly.  
2. Move to a **device_vector**.  
3. **Transform**: multiply by 2.5 using a custom functor.  
4. **Sort** the device vector in ascending order.  
5. **Reduce** to compute the sum.  
6. Copy result back to host to inspect sorted data.  

### b) Observing Device-Host Transfers

- The largest overhead is often the initial copy of `h_vec` → `d_vec` and the final copy `d_vec` → `h_vec`.  
- The transform, sort, and reduce all operate *entirely on the device* with **no** repeated device-host transfers in-between, which is **good** for performance.  
- If you frequently re-transfer data after each step, you degrade performance significantly.

---

## 6. Mermaid Diagrams

### Diagram 1: High-Level Thrust Flow

```mermaid
flowchart TD
    A[Host data (h_vec)] --> B[thrust::device_vector (d_vec)]
    B --> C[Thrust transform (GPU)]
    C --> D[Thrust sort (GPU)]
    D --> E[Thrust reduce (GPU) => sum]
    E --> F[Copy results to host if needed]
```

**Explanation**:  
Each arrow that crosses the boundary from host to device or device to host can represent a memory transfer. The transformations (transform/sort/reduce) all remain on the GPU side if we keep data in `d_vec`.

### Diagram 2: Composition of Thrust Calls

```mermaid
flowchart LR
    subgraph Thrust Ops on GPU
    direction TB
    Op1[transform(d_vec, factor)]
    Op2[sort(d_vec)]
    Op3[sum = reduce(d_vec)]
    end
    h_vec(HOST) --> d_vec(DEVICE) --> Op1 --> Op2 --> Op3 --> sum
    sum --> (Return to host scope)
```

**Explanation**:  
- We see the pipeline of Thrust calls on the device vector.

---

## 7. Common Pitfalls & Best Practices

1. **Excessive Host ↔ Device Transfers**  
   - Minimizing them is key. If your code uses `d_vec` for multiple operations, keep it on device until all GPU tasks complete.  
2. **Large Data**  
   - Thrust sort or reduce is well-optimized, but you still must ensure enough GPU memory is available.  
3. **Functor / Predicate**  
   - Always mark them `__host__ __device__` if you want them to run on GPU. Otherwise, you get compile errors or fallback to host.  
4. **Debugging**  
   - If code is not compiled for device debug, or if you see errors in user-defined functors, check that you used the correct decorations and included Thrust headers properly.  
5. **Performance**  
   - Thrust is often quite efficient. For specialized patterns or extremely large-scale custom kernels, you might out-perform Thrust with custom code. Always measure.

---

## 8. References & Further Reading

1. **Thrust Library**  
   [Thrust GitHub & Documentation](https://thrust.github.io/)  
2. **CUDA C Programming Guide** – Thrust Chapter  
   [NVIDIA Docs Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thrust)  
3. **NVIDIA Developer Blog** – Articles on sorting, scanning, transformations with Thrust.  
4. **“Programming Massively Parallel Processors”** by Kirk & Hwu – sections on Thrust usage as an STL-like library for parallel programming.

---

## 9. Conclusion

**Day 49** highlights **Thrust** for **High-Level Operations**:
- **Core**: `thrust::device_vector` to store data on GPU, use built-in algorithms (transform, sort, reduce, scan).  
- **Advantages**: Fewer lines of code, robust and optimized for typical parallel patterns.  
- **Caveat**: Watch out for device-host data movement if you frequently create/destroy device vectors or copy in/out after each step.

**Key Takeaway**:  
Replacing custom loops with **Thrust** transforms, sorts, and reductions can speed development and produce optimized code. Keep data primarily on the device to avoid overhead, and measure performance vs. hand-written kernels if needed.

---

## 10. Next Steps

1. **Practice**: Try rewriting older custom CUDA loops with `thrust::transform`, `thrust::sort`, or `thrust::reduce`.  
2. **Profiling**: Use Nsight Systems to confirm minimal device-host transfers.  
3. **Extend**: Implement a multi-step pipeline purely in Thrust, chaining transformations in device space.  
4. **Compare**: For specialized large-scale kernels, test performance vs. a custom approach.  
```
