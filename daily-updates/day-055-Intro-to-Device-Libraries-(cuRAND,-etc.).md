# Day 55: Intro to Device Libraries (cuRAND & More)

**Objective:**  
Learn how to use **device libraries** like **cuRAND** to generate random numbers on the GPU. We will demonstrate a **simple Monte Carlo simulation** (e.g., estimating \(\pi\) or another random-based computation) while highlighting best practices around **seeding** and **distribution** parameters. We’ll focus on cuRAND usage and pitfalls, such as using the wrong generator or reusing seeds incorrectly.

**Key References**:  
- [cuRAND Library User Guide](https://docs.nvidia.com/cuda/curand/index.html)

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Why Use cuRAND?](#2-why-use-curand)  
3. [Basic cuRAND Workflow](#3-basic-curand-workflow)  
4. [Practical Example: Monte Carlo \(\pi\) Estimation](#4-practical-example-monte-carlo-pi-estimation)  
   - [a) Code Snippet & Explanation](#a-code-snippet--explanation)  
   - [b) Observing Randomness & Seeds](#b-observing-randomness--seeds)  
5. [Common Pitfalls & Best Practices](#5-common-pitfalls--best-practices)  
6. [Mermaid Diagrams](#6-mermaid-diagrams)  
   - [Diagram 1: cuRAND Setup & Device RNG Usage](#diagram-1-curand-setup--device-rng-usage)  
   - [Diagram 2: Monte Carlo Flow for \(\pi\)](#diagram-2-monte-carlo-flow-for-\pi)  
7. [References & Further Reading](#7-references--further-reading)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)

---

## 1. Overview

Device libraries like **cuRAND** allow you to generate random numbers **directly on the GPU**, avoiding overhead from CPU-based RNG and large data transfers. cuRAND supports multiple RNG algorithms (e.g., Mersenne Twister, XORWOW, Philox), distributions (uniform, normal, lognormal), and can seed each generator for reproducibility.  

We’ll demonstrate a classic **Monte Carlo** approach for approximating \(\pi\) by random points in a unit square. Then we highlight library usage patterns, common mistakes, and performance considerations.

---

## 2. Why Use cuRAND?

1. **GPU-Accelerated RNG**:  
   - Avoid transferring random data from host.  
   - Generate large arrays of random numbers in parallel.

2. **Multiple RNG Types**:  
   - **XORWOW**, **MRG32k3a**, **Philox4x32-10**, etc. Each has different speed/quality trade-offs.

3. **Flexible Distributions**:  
   - Uniform, Normal, Lognormal, Poisson (with some additional steps), etc.

4. **Batch or Device API**:  
   - You can generate random arrays in batch from the host side or integrate RNG calls **inside** device kernels with the **`curandState`** approach.

---

## 3. Basic cuRAND Workflow

1. **Include** `#include <curand.h>` or `<curand_kernel.h>` if you do device API.  
2. **Choose** an RNG type (e.g., `CURAND_RNG_PSEUDO_DEFAULT`) or a **quasi**-random generator for certain HPC.  
3. **Create** a handle or states, set seeds.  
4. **Generate** random numbers (host function for array, or device function if each thread has a state).  
5. **Destroy** or free resources after usage.

**Approaches**:
- **Host** approach: `curandCreateGenerator(...)`, `curandGenerateUniform(...)` to fill a device array.  
- **Device** approach: Each thread has a `curandState`, calls `curand_uniform(&state)` in a kernel. Typically, you must **init** states with `curand_init(...)`.

---

## 4. Practical Example: Monte Carlo \(\pi\) Estimation

### a) Code Snippet & Explanation

```cpp
/**** day55_cuRAND_MonteCarloPi.cu ****/
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Each thread samples points in the unit square [0,1]x[0,1], 
// checks if inside the unit circle => accumulate hits

__global__ void monteCarloPiKernel(unsigned long long *counts, int totalPoints, unsigned long long seed) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // each thread can handle multiple points or just one. For simplicity, 1 point per thread
    if(idx< totalPoints){
        // Setup RNG state
        curandState_t state;
        curand_init(seed, idx, 0, &state); 
        // Generate x,y in [0,1)
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        // Check if in circle (x^2 + y^2 <1)
        unsigned long long localCount=0ULL;
        if(x*x + y*y <= 1.0f){
            localCount=1ULL;
        }

        // Write result
        counts[idx] = localCount;
    }
}

int main(){
    int nPoints=1<<20; // 1 million points
    // We'll spawn 1 thread per point
    dim3 block(256);
    dim3 grid( (nPoints+block.x-1)/block.x );

    // We'll store 1 result per thread => array of size nPoints
    size_t size = nPoints*sizeof(unsigned long long);
    unsigned long long *d_counts;
    cudaMalloc(&d_counts, size);

    // Launch kernel
    monteCarloPiKernel<<<grid,block>>>(d_counts,nPoints,1234ULL);
    cudaDeviceSynchronize();

    // Copy back
    unsigned long long *h_counts=(unsigned long long*)malloc(size);
    cudaMemcpy(h_counts, d_counts, size, cudaMemcpyDeviceToHost);

    // Sum
    unsigned long long sum=0ULL;
    for(int i=0;i<nPoints;i++){
        sum += h_counts[i];
    }

    // Pi approx
    double piEst= (4.0*(double)sum)/(double)nPoints;
    printf("Estimated Pi= %f\n", piEst);

    cudaFree(d_counts);
    free(h_counts);
    return 0;
}
```

**Explanation**:
1. We define a kernel where each thread calls **`curand_init`** with a `seed` + thread index => different sequences.  
2. Generate **(x,y)** in [0,1) uniform.  
3. Check if inside circle => store 1 or 0 in `counts[idx]`.  
4. On host, sum results => \(\pi \approx 4 * (inCircle / totalPoints)\).  
5. We keep it straightforward. For bigger HPC use, each thread might sample multiple points for efficiency, or we might reduce partial sums in a device approach.

### b) Observing Randomness & Seeds

- If you pass the same `seed` + same `idx`, you get identical sequences. So each thread uses a unique sequence because `idx` is unique.  
- If you want guaranteed reproducibility but different sequences per run, you might do a host RNG for the top-level seed. Or if you want full reproducibility across multiple runs, keep the same seed.

---

## 5. Common Pitfalls & Best Practices

1. **Misusing Seeds**  
   - If all threads use the same `(seed, idx=0)`, you get identical sequences => no real randomness. Must vary the seed or offset.  
2. **Distribution Parameters**  
   - For normal distributions, check if you want `mean=0, stdDev=1` or other params. Using wrong param leads to incorrect results.  
3. **Performance**  
   - Generating many random numbers can be compute-heavy. Typically, using a pseudo-random generator with good speed (e.g., XORWOW, Philox) is enough.  
4. **Batch Generation**  
   - If you just need a large array of random floats, consider using **`curandCreateGenerator`** and `curandGenerateUniform` from the **host**. Let it fill a device array more efficiently than per-thread calls.  
5. **Large Data**  
   - Watch out for memory usage if storing large arrays of random data. Possibly compute on-the-fly in a kernel.

---

## 6. Mermaid Diagrams

### Diagram 1: cuRAND Setup & Device RNG Usage

```mermaid
flowchart TD
    A[Device Kernel: each thread] --> B[curand_init(seed, idx, offset, &state)]
    B --> C[x= curand_uniform(&state)]
    C --> D[y= curand_uniform(&state)]
    D --> E[Compute result => e.g., inside circle?]
    E --> F[Write out result in global mem]
```

**Explanation**:  
The thread-based approach for random generation within device kernels. Each thread seeds a local RNG state.

### Diagram 2: Monte Carlo Flow for \(\pi\)

```mermaid
flowchart LR
    subgraph GPU
    direction TB
    RNG[Device RNG: x,y in [0,1)]
    Check[Check if x^2 + y^2 <1]
    Accum[Write 1 or 0 to global mem]
    end

    Host[Host: sum results => pi=4*(inCircle/ totalPoints)]
    RNG --> Check --> Accum
    Accum --> Host
```

**Explanation**:  
We see how random points are generated, tested, and the result is aggregated on the host.

---

## 7. References & Further Reading

1. **cuRAND Library User Guide**  
   [Docs Link](https://docs.nvidia.com/cuda/curand/index.html)  
2. **NVIDIA Developer Blog** – articles on advanced random generation or combining cuRAND with HPC libraries.  
3. **“Programming Massively Parallel Processors” by Kirk & Hwu** – covers random number usage in GPU-based Monte Carlo methods.

---

## 8. Conclusion

**Day 55**: Introduction to **Device Libraries** with **cuRAND**:

- We illustrated how to generate random numbers **on the GPU** via **`curandState`** in device kernels or using the host generator approach.  
- We built a simple **Monte Carlo** example for approximating \(\pi\).  
- We emphasized correct seeding to ensure each thread has a unique sequence or offset.  
- We pointed out distribution parameter pitfalls (like using normal vs. uniform incorrectly) and performance considerations (like batch generation vs. device kernel calls).

**Key Takeaway**:  
Using **cuRAND** effectively requires mindful handling of seeds, distribution type, and location of generation (host side or device side). By generating random data directly on the GPU, you can avoid major host-device transfer overhead, enabling large-scale Monte Carlo or HPC stochastic simulations with minimal friction.

---

## 9. Next Steps

1. **Try** different RNG types (e.g., XORWOW vs. Philox) in your code, measure performance differences.  
2. **Explore** advanced distributions: normal, lognormal, etc. for HPC or ML workloads.  
3. **Batch** generation on the host side if you only need an array of random floats in device memory. Compare speed vs. per-thread approach.  
4. **Integrate** with advanced HPC pipelines, e.g., combining cuRAND with cuBLAS or cuFFT-based computations for full GPU-based Monte Carlo, PDEs, or statistical simulations.  
5. **Profile** large Monte Carlo kernels with Nsight Systems to see if RNG or memory fetch is your main overhead.  
```
