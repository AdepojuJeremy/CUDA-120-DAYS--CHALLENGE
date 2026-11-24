# Day 29: Texture Memory (Practical)

In this lesson, we implement an image processing kernel using CUDA texture memory. We will create a simple grayscale conversion kernel that reads an input image from texture memory and produces a grayscale output. Texture memory is optimized for spatially coherent (2D) data and provides built-in caching and addressing modes, which are particularly beneficial for image processing applications.

> **Key Pitfall:**  
> Incorrect normalized coordinates can lead to invalid texture fetches. Ensure that if you use normalized coordinates, you supply values in the range [0, 1]. In this example, we use unnormalized coordinates.


---

## Table of Contents

1. [Overview](#1-overview)  
2. [Understanding Texture Memory](#2-understanding-texture-memory)  
3. [Practical Exercise: Grayscale Image Conversion Using Textures](#3-practical-exercise-grayscale-image-conversion-using-textures)  
   3.1. [Kernel Code: Grayscale Conversion](#31-kernel-code-grayscale-conversion)  
   3.2. [Host Code: Setting Up Texture Memory and Launching Kernel](#32-host-code-setting-up-texture-memory-and-launching-kernel)  
4. [Common Debugging Pitfalls](#4-common-debugging-pitfalls)  
5. [Conceptual Diagrams](#5-conceptual-diagrams)  
6. [References & Further Reading](#6-references--further-reading)  
7. [Conclusion](#7-conclusion)  
8. [Next Steps](#8-next-steps)  

---

## 1. Overview

Texture memory is a **read-only memory space** in CUDA that is optimized for 2D data access patterns. It offers:  
- **Hardware caching** for faster fetches.  
- **Built-in addressing modes** (e.g., clamp, wrap) that simplify image boundary handling.  
- **Interpolation capabilities** (e.g., linear filtering) if needed.  

In this project, we implement a kernel that converts a color image to grayscale using texture memory. The pipeline includes:  
- Allocating a CUDA array to hold the image.  
- Copying image data from the host to the CUDA array.  
- Binding the CUDA array to a texture reference.  
- Launching a kernel that uses `tex2D()` to sample the image.  
- Unbinding the texture and copying the output back to the host.

---

## 2. Understanding Texture Memory

Texture memory in CUDA is ideal for applications where data is accessed in a spatially coherent manner. It is cached, which reduces latency when multiple threads access nearby data locations. When using texture memory:  
- **Texture coordinates:**  
  Can be normalized (range [0, 1]) or unnormalized (actual pixel indices).  
  In our example, we use unnormalized coordinates by setting `normalized = false`.  
- **Binding:**  
  You bind a CUDA array to a texture reference using `cudaBindTextureToArray()`.  
- **Fetching:**  
  The kernel accesses the texture using `tex2D()`, which returns the value at the specified coordinates.

---

## 3. Practical Exercise: Grayscale Image Conversion Using Textures

We will implement a grayscale conversion kernel. For simplicity, assume the input image is stored as a single-channel (grayscale) image, and the kernel simply reads the pixel value from texture memory and writes it to an output array (or performs minimal processing).

### 3.1. Kernel Code: Grayscale Conversion

```cpp
// grayscaleKernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Declare a texture reference for 2D texture sampling.
// We use 'cudaTextureType2D' with 'cudaReadModeElementType' to fetch float elements.
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Kernel for grayscale conversion using texture memory.
// Each thread reads a pixel from texture memory and writes it to the output array.
__global__ void grayscaleKernel(float *output, int width, int height) {
    // Calculate pixel coordinates (x, y).
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure coordinates are within the image boundaries.
    if (x < width && y < height) {
        // When using unnormalized coordinates, tex2D expects pixel indices plus an offset of 0.5.
        float pixel = tex2D(texRef, x + 0.5f, y + 0.5f);
        // For a grayscale image, simply copy the pixel value.
        output[y * width + x] = pixel;
    }
}
```

**Detailed Comments:**  
- **Texture Declaration:**  
  The texture reference `texRef` is declared as a global variable.  
- **Kernel Function:**  
  Each thread calculates its (x, y) coordinate. A boundary check ensures that only valid pixels are processed.  
- **Texture Sampling:**  
  `tex2D()` is used to sample the image from texture memory. We add 0.5f to the coordinates for proper center sampling in unnormalized coordinates.  
- **Output:**  
  The fetched pixel value is stored in a linear output array.

### 3.2. Host Code: Setting Up Texture Memory and Launching Kernel

```cpp
// grayscalePipeline.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel declaration.
__global__ void grayscaleKernel(float *output, int width, int height);

// Declare texture reference (global).
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

#define CUDA_CHECK(call) {                                      \
    cudaError_t err = call;                                     \
    if(err != cudaSuccess) {                                    \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err));                        \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

int main() {
    // Image dimensions.
    int width = 1024, height = 768;
    size_t imgSize = width * height * sizeof(float);

    // Allocate host memory for the image.
    // For demonstration, we simulate a grayscale image.
    float *h_image = (float*)malloc(imgSize);
    if (!h_image) {
        printf("Failed to allocate host memory for image.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the image with random grayscale values.
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        h_image[i] = (float)(rand() % 256) / 255.0f;
    }

    // Allocate host memory for output image.
    float *h_output = (float*)malloc(imgSize);
    if (!h_output) {
        printf("Failed to allocate host memory for output.\n");
        free(h_image);
        exit(EXIT_FAILURE);
    }

    // Allocate a CUDA array for the image.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Copy image data from host to the CUDA array.
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float),
                                   width * sizeof(float), height, cudaMemcpyHostToDevice));

    // Set texture parameters.
    texRef.addressMode[0] = cudaAddressModeClamp;   // Clamp x coordinates.
    texRef.addressMode[1] = cudaAddressModeClamp;   // Clamp y coordinates.
    texRef.filterMode = cudaFilterModePoint;        // Use point sampling (no interpolation).
    texRef.normalized = false;                      // Use unnormalized coordinates.

    // Bind the CUDA array to the texture reference.
    CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray, channelDesc));

    // Allocate device memory for the output image.
    float *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, imgSize));

    // Define kernel launch parameters.
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the grayscale conversion kernel.
    grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the output image from device to host.
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost));

    // Verify output: Print the first 10 pixel values.
    printf("First 10 pixel values of the processed image:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Cleanup: Unbind texture and free all resources.
    CUDA_CHECK(cudaUnbindTexture(texRef));
    CUDA_CHECK(cudaFreeArray(cuArray));
    CUDA_CHECK(cudaFree(d_output));
    free(h_image);
    free(h_output);

    return 0;
}
```

**Detailed Comments:**  
- **Image Preparation:**  
  The host simulates a grayscale image by allocating and initializing an array of pixel values.  
- **CUDA Array Allocation:**  
  A CUDA array is allocated with `cudaMallocArray()` using a channel descriptor suitable for floats.  
- **Data Transfer:**  
  The host image is copied to the CUDA array using `cudaMemcpy2DToArray()`.  
- **Texture Binding:**  
  The texture parameters are set (clamping, point sampling, unnormalized coordinates) and the CUDA array is bound to the texture reference.  
- **Kernel Launch:**  
  The grayscale conversion kernel is launched with a 2D grid of threads, which samples the texture using `tex2D()`.  
- **Output Verification:**  
  The processed image is copied back to host memory and the first 10 pixel values are printed.  
- **Cleanup:**  
  The texture is unbound and all allocated resources (CUDA array, device memory, host memory) are freed.

---

## 4. Common Debugging Pitfalls

| Pitfall                                      | Solution                                                                                  |
|----------------------------------------------|-------------------------------------------------------------------------------------------|
| Using normalized coordinates when not intended | Set `texRef.normalized = false` if you want to use pixel indices directly.                |
| Incorrect texture parameter settings (e.g., filter mode) | Ensure you choose the correct filter mode (point vs. linear) and address modes (clamp, wrap). |
| Failing to bind or unbind the texture correctly | Always bind the CUDA array to the texture reference before kernel launch and unbind it after kernel execution using `cudaUnbindTexture()`. |
| Not synchronizing after kernel execution     | Use `cudaDeviceSynchronize()` to ensure kernel execution is complete before copying results back. |
| Memory copy errors due to incorrect pitch or dimensions | Double-check the parameters provided to `cudaMemcpy2DToArray()`.                         |

---

## 5. Conceptual Diagrams

**Diagram 1: Texture Memory Workflow for Grayscale Conversion**
```mermaid
flowchart TD
    A[Host: Allocate and Initialize Grayscale Image]
    B[Host: Allocate CUDA Array (cudaMallocArray)]
    C[Copy Image Data to CUDA Array (cudaMemcpy2DToArray)]
    D[Set Texture Parameters (Clamp, Point Sampling, Unnormalized)]
    E[Bind CUDA Array to Texture Reference (cudaBindTextureToArray)]
    F[Kernel Launch: Each Thread Computes Pixel Coordinates]
    G[Kernel: Sample Pixel using tex2D() from Texture Memory]
    H[Kernel: Write Pixel Value to Output Array]
    I[Host: Copy Output from Device to Host]
    J[Unbind Texture (cudaUnbindTexture) and Cleanup]
    
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
- The diagram details the entire workflow: from preparing image data, copying it to a CUDA array, binding it to a texture, launching the kernel for grayscale conversion, and finally unbinding the texture and cleaning up.

**Diagram 2: Kernel Execution Flow for Grayscale Conversion**
```mermaid
flowchart TD
    A[Kernel Launch]
    B[Each Thread Computes (x, y) Coordinates]
    C[Convert (x, y) to Texture Coordinates (add 0.5)]
    D[Sample Pixel Value using tex2D(texRef, x+0.5, y+0.5)]
    E[Store Sampled Pixel in Output Array]
    
    A --> B
    B --> C
    C --> D
    D --> E
```

**Explanation:**  
- This diagram shows how each thread in the kernel computes its coordinates, samples the texture using `tex2D()`, and writes the output.

---

## 6. References & Further Reading

1. **CUDA C Programming Guide – Texture Memory**  
   [CUDA Texture Memory Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)  

2. **CUDA C Best Practices Guide – Texture Memory**  
   [CUDA Best Practices: Texture Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#texture-memory)  

3. **NVIDIA CUDA Samples – Texture Memory**  
   [NVIDIA CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html#texture-memory-samples)  

4. **NVIDIA NSight Compute Documentation**  
   [NVIDIA NSight Compute](https://docs.nvidia.com/nsight-compute/)  

5. **"Programming Massively Parallel Processors: A Hands-on Approach" by David B. Kirk and Wen-mei W. Hwu**  
   *(Note: This is a book, not a direct linkable resource.)*  

6. **NVIDIA Developer Blog**  
   [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)  

---

## 7. Conclusion

In Day 29, we have:  
- Introduced texture memory and its benefits for 2D image processing.  
- Implemented a grayscale conversion kernel that samples an image using texture memory.  
- Set up the host code to allocate a CUDA array, bind it to a texture, and launch the kernel.  
- Highlighted common pitfalls such as incorrect normalized coordinates and missing texture binding/unbinding.  
- Provided extensive code examples with detailed inline comments and conceptual diagrams.  
- This project serves as a foundation for more advanced image processing tasks using CUDA texture memory.

---

## 8. Next Steps

- **Extend the Pipeline:**  
  Implement additional image processing filters (e.g., Sobel edge detection, Gaussian blur) using texture memory.  
- **Integrate with Surface Memory:**  
  Experiment with using surface memory for read-write operations alongside texture memory.  
- **Profile and Optimize:**  
  Use NVIDIA NSight Compute to profile texture fetch performance and optimize parameters (filter mode, address mode).  
- **Real-World Application:**  
  Integrate this approach into a real-time image processing application.  
- Happy CUDA coding, and continue exploring the power of texture memory for high-performance image processing!

```
