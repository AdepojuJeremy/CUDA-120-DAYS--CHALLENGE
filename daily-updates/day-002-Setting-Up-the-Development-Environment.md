# Day 02: Setting Up the Development Environment

## Table of Contents
1. [Overview](#1-overview)  
2. [Why a Proper Environment Matters](#2-why-a-proper-environment-matters)  
3. [Core Components of a CUDA Dev Setup](#3-core-components-of-a-cuda-dev-setup)  
4. [Organizing Your Project](#4-organizing-your-project)  
5. [Compiler & Build System Essentials](#5-compiler--build-system-essentials)  
6. [Debugging Approaches](#6-debugging-approaches)  
7. [Profiling & Performance Tools](#7-profiling--performance-tools)  
8. [Practical Exercise: Multi-File CUDA Project](#8-practical-exercise-multi-file-cuda-project)  
9. [Common Pitfalls](#9-common-pitfalls)  
10. [References & Further Reading](#10-references--further-reading)  
11. [Conclusion](#11-conclusion)  
12. [Next Steps](#12-next-steps)  

## 1. Overview
On **Day 1**, you installed the CUDA toolkit and wrote a simple "Hello GPU" kernel. Now, let's **build a robust environment** for larger, more complex projects. By the end of **Day 2**, you'll be able to:

- Set up an organized folder structure for multi-file CUDA applications.  
- Understand how to configure compilation flags and environment variables for different OS platforms.  
- Use debugging and profiling tools efficiently.  
- Avoid common mistakes that occur when scaling up from a single `.cu` file to multi-kernel, multi-file projects.

Mastering your development environment ensures you won't be derailed by messy builds or cryptic runtime errors as your CUDA journey progresses.

## 2. Why a Proper Environment Matters
1. **Scalability**: As soon as you add multiple kernels or external libraries (cuBLAS, cuFFT, etc.), a solid environment helps maintain order.  
2. **Maintainability**: Isolating host vs. device code, using consistent naming conventions, and adopting a build system (e.g., Make or CMake) all reduce technical debt.  
3. **Performance Insights**: Quick access to profiling tools (Nsight Systems, Nsight Compute) and debug builds fosters iterative optimization.  
4. **Cross-Platform Consistency**: With a reliable setup, you can develop on Windows, Linux, or macOS while minimizing environment-related quirks.

## 3. Core Components of a CUDA Dev Setup

1. **CUDA Toolkit**  
   - Includes `nvcc`, header files, libraries (e.g., `libcudart`, `libcublas`).  
   - Make sure it's on your `PATH` (Windows) or in your `LD_LIBRARY_PATH` (Linux/macOS) if needed.
  
2. **NVIDIA Driver**  
   - Must match or exceed your CUDA Toolkit version; mismatches lead to compilation or runtime errors.  
   - Check with `nvidia-smi` (Linux) or in Windows' Device Manager → Display Adapters → NVIDIA driver version.

3. **Host Compiler (e.g., gcc, clang, MSVC)**  
   - `nvcc` invokes a host compiler for CPU portions of your CUDA code.  
   - Keep your host compiler updated, and ensure it's compatible with your CUDA version (check [NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#supported-compilers)).

4. **IDE or Editor**  
   - Many developers use **Visual Studio** (Windows) or **Nsight Eclipse Edition** (Linux).  
   - Others prefer **VS Code**, **CLion**, or a text editor (Vim, Emacs) with a custom build workflow.

## 4. Organizing Your Project

A clear folder structure prevents confusion once you start splitting logic into multiple files:
```
my-cuda-project/
├── src/
│   ├── main.cu          # Host code & kernel invocations
│   ├── kernels.cu       # Device kernels (can be multiple .cu files)
│   └── utils.cu         # Additional device or host utilities
├── include/
│   ├── kernels.h        # Declarations for kernels
│   └── utils.h
├── Makefile (or CMakeLists.txt)
├── docs/                # Optional: documentation, design notes
└── README.md
```

**Tips**:
- Keep separate `.h` files for device/host function prototypes.  
- If you plan to use advanced features like **separate compilation** or **device linking**, structure accordingly (e.g., `-dc` and `-dlink` flags in Makefiles).

## 5. Compiler & Build System Essentials

### 5.1 Basic `nvcc` Usage
- **`-o <output>`**: Set output file name (e.g., `-o my_app`).  
- **`-arch=sm_XX`**: Compile for a specific GPU architecture, like `sm_75` (Turing) or `sm_86` (Ampere).  
- **`-G`**: Enable debug info for device code (disables certain optimizations).  
- **`-lineinfo`**: Include source line info for better profiling and debugging.  
- **`-Xcompiler "<flags>"`**: Pass additional flags to the host compiler (e.g., `-Wall`, `-O2`).

### 5.2 Build Systems
1. **Makefiles**  
   - Quick, widely used. A simple example can compile all `.cu` files in `src/`.
2. **CMake**  
   - More robust for cross-platform builds.  
   - Add lines like `enable_language(CUDA)` and `project(MyProject LANGUAGES CXX CUDA)` in your `CMakeLists.txt`.
3. **Visual Studio or Nsight Eclipse**  
   - Create a CUDA project using the provided wizards; IDE auto-manages compilation and linking.

### 5.3 Separate Compilation & Device Linking
For very large projects with multiple CUDA files, you might use:
- **`-dc`**: Compile device code but don't link.  
- **`-dlink`**: Perform device linking at a later stage to combine multiple compiled `.o` or `.obj` files into a single executable.

```bash
nvcc -dc kernels.cu -o kernels.o
nvcc -dc utils.cu -o utils.o
nvcc -dlink kernels.o utils.o -o dlink.o
nvcc main.cu kernels.o utils.o dlink.o -o final_app
```

## 6. Debugging Approaches

1. **cuda-gdb**  
   - GDB-based debugger for CUDA. Set device breakpoints, inspect thread-local variables, and step through kernel instructions.  
   - Usage: `cuda-gdb ./my_app`

2. **Nsight Eclipse (Linux) / Nsight Visual Studio (Windows)**  
   - Integrates source-level debugging, breakpoints, variable inspection, and GPU kernel stepping within an IDE.

3. **`printf` in Kernels**  
   - Quick for small-scale debugging. But watch out for performance overhead if you're printing a lot.

4. **`cuda-memcheck`**  
   - Checks for out-of-bounds access, misaligned memory usage, and other GPU memory errors.  
   - Usage: `cuda-memcheck ./my_app`

## 7. Profiling & Performance Tools

1. **Nsight Systems**  
   - A timeline-based profiler. It shows how CPU functions and GPU kernels overlap, helps identify concurrency or synchronization bottlenecks.  

2. **Nsight Compute**  
   - Offers deep kernel-level metrics: occupancy, instruction throughput, warp divergence, memory transactions.  
   - Vital for performance tuning.

3. **CLI Tools (Legacy)**  
   - `nvprof` and `nvvp` (Visual Profiler) are older, now mostly replaced by Nsight. Some legacy workflows might still rely on them.

**Tip**: Start with Nsight Systems to see the "bigger picture." Then switch to Nsight Compute for kernel-deep optimizations.

## 8. Practical Exercise: Multi-File CUDA Project

### 8.1 Create a Simple Project Structure
```bash
mkdir -p my-cuda-project/src
mkdir -p my-cuda-project/include
cd my-cuda-project
```

### 8.2 Write a Kernel (`src/kernels.cu`)
```cpp
#include <stdio.h>
#include "kernels.h"

// A dummy kernel that prints block and thread indices
__global__ void dummyKernel() {
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}
```

### 8.3 Header File (`include/kernels.h`)
```cpp
#ifndef KERNELS_H
#define KERNELS_H

__global__ void dummyKernel();

#endif
```

### 8.4 Main File (`src/main.cu`)
```cpp
#include <cuda_runtime.h>
#include "kernels.h"

int main() {
    dummyKernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 8.5 Sample Makefile
```makefile
PROJECT = cuda_app
NVCC = nvcc
SRCS = src/main.cu src/kernels.cu
INCLUDES = -I./include
ARCH = -arch=sm_75
FLAGS = -Xcompiler "-Wall -O2"

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o $(PROJECT) $(SRCS)

clean:
	rm -f $(PROJECT) *.o
```

### 8.6 Build & Run
```bash
make
./cuda_app
```
**Output**:
```
Block 0, Thread 0
Block 0, Thread 1
Block 0, Thread 2
Block 0, Thread 3
Block 1, Thread 0
...
```

## 9. Common Pitfalls

1. **Forgetting Debug vs. Release Flags**  
   - Use `-G` for debugging, remove it (or use `-O2`) for release/performance builds.
2. **Driver/Toolkit Mismatch**  
   - Ensure driver is not too old for your toolkit (check [NVIDIA docs](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)).
3. **Environment Variables on Linux/macOS**  
   - If the CUDA toolkit isn't on your `PATH` or `LD_LIBRARY_PATH`, you'll see "command not found" or linking errors.
4. **Excessive `printf`**  
   - Printing inside kernels is helpful but drastically slows down your app in large loops. Use sparingly.
5. **Uninitialized Device Memory**  
   - Always check your pointers are allocated (`cudaMalloc`) and/or copied from host memory before use.

## 10. References & Further Reading

1. **[CUDA C Programming Guide – Chapters 2 & 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)**  
   Deeper dive into compilation, linking, and device code structure.
2. **[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)**  
   System-wide profiling and concurrency visualization.
3. **[Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)**  
   Kernel-level performance metrics and optimization insights.
4. **[CMake for CUDA](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)**  
   Official docs on enabling CUDA in CMake-based workflows.
5. **[cuda-memcheck Documentation](https://docs.nvidia.com/cuda/cuda-memcheck/index.html)**  
   Detecting GPU memory errors.

## 11. Conclusion
You now have the **tools and knowledge** to structure a multi-file CUDA project, compile it with the right flags, and debug or profile your kernels. Investing time to set up a **clean, scalable environment** early on will pay off as you write more complex kernels, integrate libraries, and strive for maximum performance.

**Key Points**:
- Keep your project organized with separate files for kernels, headers, and host logic.  
- Use Makefiles or CMake for consistent, repeatable builds.  
- Familiarize yourself with cuda-gdb and Nsight tools to tackle debugging and performance tuning head-on.

## 12. Next Steps
**In Day 3**, we'll dive into **thread hierarchy** (threads, blocks, warps) and how to effectively map data to those threads. Make sure your environment is stable so you can focus on the GPU's execution model and fine-tuning your parallel approach.

*__End of Day 2__: Your development environment is now fully primed for more advanced CUDA topics!*
