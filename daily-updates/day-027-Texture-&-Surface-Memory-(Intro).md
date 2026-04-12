# Day 27: Texture & Surface Memory (Intro)

In this lesson, we introduce **Texture Memory** in CUDA, a special read-only memory space optimized for 2D spatial locality and interpolation. Texture memory is particularly useful for image processing tasks such as sampling, filtering, and edge detection. We will cover the basics of texture memory, how to bind a CUDA array to a texture, and how to sample a small 2D texture in a kernel. We will also compare texture fetch performance with global memory fetch, and discuss common pitfalls such as missing texture binding/unbinding steps.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [What is Texture Memory?](#2-what-is-texture-memory)  
3. [Setting Up Texture Memory in CUDA](#3-setting-up-texture-memory-in-cuda)  
4. [Practical Exercise: Sampling a Small 2D Texture](#4-practical-exercise-sampling-a-small-2d-texture)  
    - [a) Kernel Code: Texture Sampling vs. Global Memory Fetch](#a-kernel-code-texture-sampling-vs-global-memory-fetch)  
    - [b) Host Code: Binding and Unbinding Texture Memory](#b-host-code-binding-and-unbinding-texture-memory)  
5. [Common Debugging Pitfalls and Best Practices](#5-common-debugging-pitfalls-and-best-practices)  
6. [Conceptual Diagrams](#6-conceptual-diagrams)  
7. [References & Further Reading](#7-references--further-reading)  
8. [Conclusion](#8-conclusion)  
9. [Next Steps](#9-next-steps)  

---

## 1. Overview

Texture memory is a **read-only** memory space on the GPU that is optimized for 2D spatial access patterns. It provides benefits such as:
- **Hardware caching** for faster access.
- **Built-in addressing modes** (e.g., clamping, wrapping) useful for image processing.
- **Interpolation support** for filtering operations.

Unlike global memory, texture fetches are optimized for **spatial locality**; when threads access nearby data elements, the texture cache can dramatically improve performance. However, to use texture memory properly, you must correctly bind the data to a texture object and unbind it when finished.

---

## 2. What is Texture Memory?

**Texture Memory** in CUDA is:
- **Read-only** during kernel execution.
- Cached on the GPU for faster access.
- Ideal for accessing image-like data with 2D spatial locality.

It is typically used by:
- **Image processing algorithms** (e.g., filtering, edge detection).
- **3D graphics applications.**

**Important:**  
Texture memory requires proper binding of CUDA arrays (or linear memory) to a texture object before the kernel can sample from it.

---

## 3. Setting Up Texture Memory in CUDA

To use texture memory:
1. **Allocate a CUDA Array:**  
   Use `cudaMallocArray()` to allocate a 2D array on the device.
2. **Copy Data to the CUDA Array:**  
   Use `cudaMemcpy2DToArray()` to copy image data to the CUDA array.
3. **Create a Texture Object:**  
   Set up a `cudaResourceDesc` and `cudaTextureDesc`, then call `cudaCreateTextureObject()`.
4. **Use the Texture Object in a Kernel:**  
   Sample the texture in your kernel using functions like `tex2D()`.
5. **Destroy the Texture Object:**  
   After kernel execution, release the texture object with `cudaDestroyTextureObject()`.

---

## 4. Practical Exercise: Sampling a Small 2D Texture

In this exercise, we implement a simple kernel that compares sampling from a 2D texture with a direct global memory fetch. We will use a small 2D image for demonstration.

### a) Kernel Code: Texture Sampling vs. Global Memory Fetch

```cpp
// textureSampleKernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Declare a texture object globally. Note that in CUDA 5.0 and later,
// we use texture objects rather than texture references.
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Kernel that samples data using texture memory and compares it with a global memory fetch.
__global__ void textureVsGlobalKernel(const float *globalData, float *outputTex, float *outputGlobal, int width, int height) {
    // Calculate pixel coordinates.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds.
    if (x < width && y < height) {
        // Calculate normalized texture coordinates.
        // Note: tex2D() expects coordinates in float format.
        float u = x + 0.5f;
        float v = y + 0.5f;
        
        // Fetch the value from the texture.
        float texVal = tex2D(texRef, u, v);
        
        // Compute the index for global memory.
        int idx = y * width + x;
        // Fetch the value directly from global memory.
        float globalVal = globalData[idx];
        
        // Write both values to output arrays for comparison.
        outputTex[idx] = texVal;
        outputGlobal[idx] = globalVal;
    }
}
```

*Detailed Comments:*
- The kernel computes 2D coordinates (x, y) for each pixel.
- It calculates normalized coordinates to sample the texture with `tex2D()`.
- It also reads the same value directly from a global memory array.
- The outputs are stored in separate arrays (`outputTex` and `outputGlobal`) for later comparison.

---

### b) Host Code: Binding and Unbinding Texture Memory

```cpp
// textureSampleHost.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel declaration.
__global__ void textureVsGlobalKernel(const float *globalData, float *outputTex, float *outputGlobal, int width, int height);

// Texture reference is declared globally (for texture objects we can also use cudaTextureObject_t, but here we use the legacy texture reference for simplicity).
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

int main() {
    // Image dimensions.
    int width = 512, height = 512;
    size_t size = width * height * sizeof(float);

    // Allocate host memory for image and output arrays.
    float *h_image = (float*)malloc(size);
    float *h_outputTex = (float*)malloc(size);
    float *h_outputGlobal = (float*)malloc(size);
    if (!h_image || !h_outputTex || !h_outputGlobal) {
        printf("Host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize image with random values.
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        h_image[i] = (float)(rand() % 256) / 255.0f;
    }

    // Allocate CUDA array for the 2D texture.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Copy image data from host to CUDA array.
    cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Set texture parameters (address mode, filter mode, etc.).
    texRef.addressMode[0] = cudaAddressModeClamp;  // Clamp coordinates
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;         // No filtering
    texRef.normalized = false;                       // Use unnormalized texture coordinates

    // Bind the CUDA array to the texture reference.
    cudaBindTextureToArray(texRef, cuArray, channelDesc);

    // Allocate device memory for global data and output arrays.
    float *d_image, *d_outputTex, *d_outputGlobal;
    cudaMalloc((void**)&d_image, size);
    cudaMalloc((void**)&d_outputTex, size);
    cudaMalloc((void**)&d_outputGlobal, size);

    // Copy the same image data to device global memory.
    cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);

    // Define kernel launch parameters.
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel.
    textureVsGlobalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_outputTex, d_outputGlobal, width, height);
    cudaDeviceSynchronize();

    // Copy results from device to host.
    cudaMemcpy(h_outputTex, d_outputTex, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputGlobal, d_outputGlobal, size, cudaMemcpyDeviceToHost);

    // Compare outputs (for demonstration, we print the first 10 elements).
    printf("First 10 values from texture fetch:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_outputTex[i]);
    }
    printf("\n");

    printf("First 10 values from global memory fetch:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_outputGlobal[i]);
    }
    printf("\n");

    // Unbind the texture.
    cudaUnbindTexture(texRef);

    // Free device memory and CUDA array.
    cudaFree(d_image);
    cudaFree(d_outputTex);
    cudaFree(d_outputGlobal);
    cudaFreeArray(cuArray);

    // Free host memory.
    free(h_image);
    free(h_outputTex);
    free(h_outputGlobal);

    return 0;
}
```

*Detailed Comments:*
- **Host Memory Allocation:**  
  Allocate memory for the image and output arrays on the host.
- **CUDA Array Allocation:**  
  Use `cudaMallocArray()` to allocate a 2D CUDA array for the texture.
- **Copy Data to CUDA Array:**  
  Use `cudaMemcpy2DToArray()` to copy the image data into the CUDA array.
- **Texture Parameter Setup:**  
  Set texture address mode, filter mode, and normalization parameters.
- **Binding the Texture:**  
  Bind the CUDA array to the texture reference with `cudaBindTextureToArray()`.
- **Device Memory Allocation for Global Data:**  
  Allocate a separate device array to hold the same image data for comparison.
- **Kernel Launch:**  
  Launch the kernel to sample the texture and fetch data from global memory.
- **Result Comparison:**  
  Copy the outputs back to the host and print them for comparison.
- **Unbinding and Cleanup:**  
  Unbind the texture using `cudaUnbindTexture()`, free device memory, and free host memory.

---

## 5. Common Debugging Pitfalls and Best Practices

| **Pitfall**                               | **Solution**                                             |
|-------------------------------------------|----------------------------------------------------------|
| Missing texture binding                   | Always bind the CUDA array to the texture reference before launching the kernel. |
| Forgetting to unbind the texture          | Unbind texture with `cudaUnbindTexture()` after kernel execution to avoid unintended side effects in subsequent operations. |
| Incorrect texture parameters              | Ensure correct address modes, filter mode, and normalization settings for your use case. |
| Mismatched memory types                   | Do not mix texture fetches with non-texture global memory accesses without proper validation. |
| Not checking for CUDA errors              | Use `cudaGetErrorString()` to log any errors after CUDA API calls. |

---

## 6. Conceptual Diagrams

### Diagram 1: Texture Memory Workflow
```mermaid
flowchart TD
    A[Host: Allocate Image Data (Pageable/Pinned Memory)]
    B[Host: Allocate CUDA Array via cudaMallocArray()]
    C[Host: Copy Image Data to CUDA Array (cudaMemcpy2DToArray)]
    D[Set Texture Parameters (addressMode, filterMode, normalized)]
    E[Bind CUDA Array to Texture Reference (cudaBindTextureToArray)]
    F[Kernel: Sample Data using tex2D()]
    G[Kernel: Also fetch data from Global Memory for Comparison]
    H[Host: Unbind Texture (cudaUnbindTexture)]
    I[Host: Copy Kernel Output from Device to Host]
    J[Host: Compare and Verify Results]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> I
    I --> J
    J --> H
```

*Explanation:*  
- The diagram illustrates the workflow from allocating image data, copying it to a CUDA array, binding it to a texture, sampling it in the kernel, and finally unbinding and copying results back to the host.

### Diagram 2: Kernel Execution Flow for Texture Sampling

```mermaid
flowchart TD
    A[Kernel Launch]
    B[Each Thread Computes (x, y) Coordinates]
    C[Compute Normalized Texture Coordinates]
    D[Fetch Value using tex2D(texRef, u, v)]
    E[Fetch Value from Global Memory]
    F[Store Both Values in Output Arrays]
    
    A --> B
    B --> C
    C --> D
    C --> E
    D & E --> F
```

*Explanation:*  
- This diagram details the steps each thread takes within the kernel.
- Threads compute their coordinates, fetch values from both texture memory and global memory, and store the results for later comparison.

---

## 7. References & Further Reading

1. **CUDA C Programming Guide – Texture Memory**  
   [CUDA Texture Memory Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)  
   Comprehensive guide to texture memory in CUDA.
2. **CUDA C Best Practices Guide – Texture Memory**  
   [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#texture-memory)  
   Best practices for using texture memory effectively.
3. **NVIDIA CUDA Samples – Texture Memory**  
   [NVIDIA CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html)  
   Example codes provided by NVIDIA for texture memory usage.
4. **NVIDIA Developer Blog – Texture Memory**  
   [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)  
   Articles discussing optimization and usage of texture memory.
5. **"Programming Massively Parallel Processors: A Hands-on Approach" by David B. Kirk and Wen-mei W. Hwu**  
   A comprehensive resource covering CUDA memory hierarchies including texture memory.

---

## 8. Conclusion

In Day 27, you have learned:
- **The basics of texture memory** and its benefits for 2D spatial data access.
- **How to allocate a CUDA array and bind it to a texture reference.**
- **How to sample a texture in a CUDA kernel** using `tex2D()`, and compare it to global memory fetch.
- **The importance of correct texture binding and unbinding** to avoid bugs.
- **Detailed code examples** with extensive comments and conceptual diagrams to reinforce understanding.

---

## 9. Next Steps

- **Experiment:**  
  Extend this project to implement advanced image processing algorithms (e.g., Sobel edge detection, Gaussian blur) using texture memory.
- **Profile:**  
  Use NVIDIA NSight Compute to compare performance differences between texture fetches and global memory fetches.
- **Optimize:**  
  Experiment with different texture parameters (filter mode, address mode, normalized coordinates) to find the optimal configuration for your application.
- **Integrate:**  
  Combine texture memory with other optimization techniques (such as shared memory and asynchronous transfers) in larger projects.

```
