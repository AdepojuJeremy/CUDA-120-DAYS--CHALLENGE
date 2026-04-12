# Day 40: Capstone Project #2 – Sparse Matrix-Vector Multiplication

**Objective**  
In this second capstone project, we tackle **Sparse Matrix-Vector Multiplication (SpMV)** on large sparse data sets using CUDA. Sparse matrices arise in numerous applications (e.g., scientific computing, graph analytics, machine learning). Storing and multiplying large sparse matrices efficiently on the GPU requires specialized data structures and algorithms to avoid excessive memory usage and wasted computation on zero elements.

**Key References**:  
- [CUDA C Programming Guide – Sparse Matrix Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
- [NVIDIA Developer Blog on SpMV](https://developer.nvidia.com/blog/)  
- [“Programming Massively Parallel Processors” by Kirk & Hwu – Sparse Computations Chapter]  

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Why Sparse Matrix-Vector Multiplication?](#2-why-sparse-matrix-vector-multiplication)  
3. [Sparse Data Formats](#3-sparse-data-formats)  
   - [a) Compressed Sparse Row (CSR)](#a-compressed-sparse-row-csr)  
   - [b) ELL or Other Formats](#b-ell-or-other-formats)  
4. [Practical Implementation: CSR-Based SpMV](#4-practical-implementation-csr-based-spmv)  
   - [a) Host Data Preparation](#a-host-data-preparation)  
   - [b) Device Kernel for CSR SpMV](#b-device-kernel-for-csr-spmv)  
   - [c) Putting It All Together](#c-putting-it-all-together)  
5. [Conceptual Diagrams](#5-conceptual-diagrams)  
6. [Common Pitfalls & Best Practices](#6-common-pitfalls--best-practices)  
7. [References & Further Reading](#7-references--further-reading)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)

---

## 1. Overview

**Sparse Matrix-Vector Multiplication** (SpMV) multiplies an N×N matrix \( A \), which is mostly zeros, by a vector \( x \). The result is \( y = A \times x \). Because many entries of \( A \) are zero, we store only non-zero entries plus some indexing structure. This drastically reduces memory usage and speeds up multiplication for large but sparse data.

**Key Goals**:
- Represent the matrix in a **compact** format (CSR, ELL, etc.).
- Launch a CUDA kernel that iterates only over **non-zero** elements.
- Use **coalesced memory access** if possible, and handle concurrency for large data sets.

---

## 2. Why Sparse Matrix-Vector Multiplication?

1. **Real-World Applications**:
   - Scientific simulations with large, sparse systems (finite element methods).
   - Graph algorithms with adjacency matrices that are mostly zero.
   - Machine learning models with high-dimensional, sparse feature vectors.
2. **Performance Gains**:
   - Avoid reading/writing zeros reduces memory overhead.
   - GPU parallelism can accelerate repeated SpMV steps significantly (like iterative solvers).

---

## 3. Sparse Data Formats

### a) Compressed Sparse Row (CSR)

**CSR** stores:
- **rowOffsets**: An array of length (numRows+1). rowOffsets[i] indicates the start index in the “columns” and “values” arrays for row i.  
- **columns**: The column indices of each non-zero element.  
- **values**: The corresponding non-zero values.

**Pros**:  
- Intuitive row-based format.  
- Good for row-based parallelization.  

### b) ELL or Other Formats

**ELL** packs each row’s non-zeros into a fixed-length array. Works well if the distribution of non-zeros per row is fairly uniform.  
**Others**: e.g., COO (Coordinate), HYB (ELL+COO hybrid). Each has different pros/cons depending on matrix structure.

---

## 4. Practical Implementation: CSR-Based SpMV

### a) Host Data Preparation

**Example**: Suppose we have an N×N matrix and a vector x. We gather the non-zero entries row by row into:

- `rowOffsets[i]`: The cumulative count of non-zero elements up to row i.  
- `columns[j]`: Column index of the j-th non-zero.  
- `values[j]`: Value of the j-th non-zero.  

```cpp
// day40_csrSpmv.cu (partial snippet for data creation)
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>

void createCSR(int N, const float *denseMatrix,
               std::vector<int> &rowOffsets,
               std::vector<int> &columns,
               std::vector<float> &values) {
    rowOffsets.resize(N+1, 0);
    // Count non-zeros row by row
    int nnz = 0;
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            float val = denseMatrix[i*N + j];
            if(val != 0.0f){
                columns.push_back(j);
                values.push_back(val);
                nnz++;
            }
        }
        rowOffsets[i+1] = nnz;
    }
}
```

**Note**: For demonstration, we convert from a dense input to CSR. In real usage, the matrix might already be given in CSR or read from a file.

### b) Device Kernel for CSR SpMV

**Kernel**: Each row i is processed by one thread (or a group of threads), computing:

\[
y[i] = \sum_{j = rowOffsets[i]}^{ rowOffsets[i+1]-1 } \text{values}[j] * x[\text{columns}[j]]
\]

**Implementation**:

```cpp
__global__ void csrSpmvKernel(const int *rowOffsets,
                              const int *columns,
                              const float *values,
                              const float *x,
                              float *y,
                              int numRows) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < numRows) {
        int rowStart = rowOffsets[row];
        int rowEnd   = rowOffsets[row+1];
        float sum    = 0.0f;
        for(int jj = rowStart; jj < rowEnd; jj++){
            int col = columns[jj];
            float val = values[jj];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}
```

**Comments**:
- Each thread is responsible for one row. The loop runs over that row’s non-zero elements.
- This simple approach might lead to load imbalance if row lengths vary, but it’s straightforward.

### c) Putting It All Together

```cpp
// day40_csrSpmv.cu (main)
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "day40_csrSpmv_utils.h" // or inline

// Kernel from snippet above
__global__ void csrSpmvKernel(...);

int main() {
    int N = 5;  // a small example or large for actual usage

    // 1) Create or load a dense matrix (for demonstration)
    std::vector<float> denseMat(N*N, 0.0f);
    // Fill with random, some zeros
    // ...

    // 2) Convert to CSR
    std::vector<int> rowOffsets, columns;
    std::vector<float> values;
    createCSR(N, denseMat.data(), rowOffsets, columns, values);
    int nnz = values.size();

    // 3) Allocate device memory for CSR
    int *d_rowOffsets, *d_columns;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_rowOffsets, (N+1)*sizeof(int));
    cudaMalloc(&d_columns, nnz*sizeof(int));
    cudaMalloc(&d_values, nnz*sizeof(float));

    // 4) Copy rowOffsets, columns, values to device
    cudaMemcpy(d_rowOffsets, rowOffsets.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, columns.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nnz*sizeof(float), cudaMemcpyHostToDevice);

    // 5) Create x & y
    std::vector<float> h_x(N, 1.0f); // x vector
    std::vector<float> h_y(N, 0.0f);
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, N*sizeof(float));

    // 6) Launch SpMV kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    csrSpmvKernel<<<blocksPerGrid, threadsPerBlock>>>(d_rowOffsets, d_columns,
                                                      d_values, d_x, d_y, N);
    cudaDeviceSynchronize();

    // 7) Copy y back
    cudaMemcpy(h_y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print partial result
    std::cout << "y[0] = " << h_y[0] << " y[N-1] = " << h_y[N-1] << std::endl;

    // Cleanup
    cudaFree(d_rowOffsets);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
```

**Testing**:  
- For large N with a suitably sparse matrix, measure performance. Possibly compare this CSR approach to a naive dense approach to see the speed improvements if the matrix is truly sparse.

---

## 5. Conceptual Diagrams

### Diagram 1: CSR Structure

```mermaid
flowchart LR
    subgraph CSR
    direction TB
    A[RowOffsets: [0, 2, 5, 7, ...]] 
    B[Columns: [colIndex1, colIndex2, ...]] 
    C[Values: [val1, val2, ...]]
    end
A --> B
B --> C
```

*Explanation*:  
- `rowOffsets[i]` points to the start of row i’s data in `Columns` & `Values`.

### Diagram 2: SpMV in CSR

```mermaid
flowchart TD
    A[For row i] --> B[Get rowStart = rowOffsets[i], rowEnd = rowOffsets[i+1]]
    B --> C[sum=0.0]
    C --> D[For j in rowStart..rowEnd-1]
    D --> E[ col=columns[j], val=values[j] ]
    E --> F[ sum += val*x[col] ]
    F --> G[ y[i] = sum ]
```

---

## 6. Common Pitfalls & Best Practices

1. **Incorrect Offsets**  
   - Off-by-one errors in building `rowOffsets` can cause catastrophic indexing.  
2. **Non-coalesced Accesses**  
   - If columns array leads to scattered fetches from x, performance can degrade.  
   - Some formats reorder columns for better coalescing.  
3. **Load Balancing**  
   - If some rows are extremely long while others are short, a row-per-thread approach might be unbalanced.  
   - Solutions: row-per-block or warp-based approaches for balanced load.  
4. **Large CSR Overheads**  
   - If matrix is extremely sparse, consider ELL or HYB.  
5. **Double-check**  
   - Mismatched rowOffsets can lead to out-of-bounds access. Validate carefully.

---

## 7. References & Further Reading

1. **CUDA C Programming Guide – Sparse Matrix**  
   [Documentation Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#sparse-matrix)  
2. **NVIDIA Developer Blog – SpMV**  
   Articles about optimizing SpMV on GPUs.  
3. **“Programming Massively Parallel Processors” by David Kirk & Wen-mei W. Hwu**  
   Detailed coverage on advanced sparse computations.  

---

## 8. Conclusion

**Day 40** covers a **Capstone Project** implementing **Sparse Matrix-Vector Multiplication** (SpMV). Key highlights:

- **Use CSR** or other sparse formats to store only non-zero entries.  
- **Implement** a CSR-based kernel that loops over row ranges.  
- **Parallelize** each row or segment of non-zero elements.  
- **Measure** performance vs. a naive dense approach; see significant gains if matrix is truly sparse.  
- **Pitfalls**: watch out for indexing mistakes, unbalanced row lengths, and memory coalescing challenges.

**Takeaway**: SpMV is a crucial building block for HPC and large-scale linear algebra. By leveraging GPU concurrency and careful sparse data structures, we can effectively handle large, sparse data sets.

---

## 9. Next Steps

1. **Further Optimization**: Explore warp-based or block-based approaches for load balancing (e.g., a warp handles multiple rows).  
2. **Hybrid Formats**: Combine ELL and COO in a HYB format if row lengths vary widely.  
3. **Use Tools**: NVIDIA’s cuSPARSE library provides advanced, optimized SpMV implementations you can compare with your own.  
4. **Expand**: If you handle iterative solvers, keep re-using your SpMV approach and see how to further optimize or cache data.

Happy CUDA coding, and enjoy your final **capstone** in advanced GPU-based sparse computations!
```
