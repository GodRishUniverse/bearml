# BearML: a `mini-pytorch` in pure C++/CUDA

Deep Learning Framework: Implementing Neural Networks in C++

## Why?

I want to implement a project in C++ and CUDA (AMD HIP/ROCm may also be implemented - **optional right now**) properly and this would enable me to do that.

C++ is very fast as compared to Python 3.

Inspired by [llm.c](https://github.com/karpathy/llm.c) from Andrej Karpathy and the [PyTorch](https://github.com/pytorch/pytorch) repository

## Quick Start

### Prerequisites
**Only tested on Fedora Linux at the moment**
- C++20 or higher [Check NVCC compatibility]
- CMake 3.30+
- CUDA Toolkit 13.0 [will have to make optional]

### Building

```bash
git clone https://github.com/GodRishUniverse/BearML.git;
cd BearML;
mkdir build && cd build;
cmake ..;
make;
<edit your cmake for building the files>
```

Please edit the CMake files according to your liking.


## Implementation

The implementation is currently trying to implement broadcasting tensors, GEMM, CUDA support (**Refactoring in Progress**).

A simple reverse-mode autodifferentiation pipeline is set up. Need to work on applying that for the activation functions and verify on the Tensor operations.

The Autodiff works on  `double` and `Tensor` - cuda support needs to be added and checked

## What has been done

* Simple (double) based automatic differentiation for: addition, subtraction, multiplication and division
* A simple reverse-mode autodifferentiation pipeline is set up. Need to work on ap
* added a sign function
* added double and Tensor comparison masks
* element wise divide is done
* SGD is done
* Can train a basic neural network on the CPU


### NOTE and manual patch-work

CUDA compilation (manual patch applied)
`/usr/local/cuda-13/targets/x86_64-linux/include/crt`
as suggested by "https://www.linuxquestions.org/questions/slackware-14/help-compiling-ffmpeg-8-with-cuda-12-9-using-an-alternative-glibc-4175754496-print/"
changed math libraries to
```c
#if defined(__GLIBC__) && (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 42)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double rsqrt(double x) noexcept (true);
#else
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double rsqrt(double x);
#endif

#if defined(__GLIBC__) && (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 42)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float rsqrtf(float x) noexcept (true);
#else
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float rsqrtf(float x);
#endif
```

`__restrict__` keyword usage: https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/

# What we need to do

## CUDA Support & Core Infrastructure
* **Refactoring for CUDA Support (Major Overhaul)** — **IN PROGRESS (Priority)** 
    * Write host code and kernels — **Status: Kernels in progress**
      * kernels need to be fixed for when the datasize is larger than the number of threads (this wont compute it) AS THREAD_IDX WILL NEVER REACH N
    * Call kernels in the `Tensor` class when devices match (CUDA). **DOING alongside CUDA support**
    * Implement **Lazy Copy** operation.
    * **Memory Management:** Address CPU/GPU memory usage. (Decision needed: Single memory space vs. syncing to avoid expensive copy operations).
* **Refactoring for Slice Support** - thinking about it 
    * All ops will have to become slice aware and a `::contiguous` static function will be needed 
    * This will allow easy implementation of convolution operations (i think) - but also nice way to have slicing support
  
* **Dependency Management:** Fix `#includes` for the repo to remove cyclical dependencies and repeated includes. — **Priority After Kernels**
* **Templatify**
  * **Templatify Tensor:** Implement template specialization for `Tensor` class to support different data types. **IMPORTANT (Priority)**
  * **Templatify Autodiff:** Implement template specialization for `Autodiff` class to support different data types.
  * **Templatify Kernel:** Implement template specialization for `Kernel` class to support different data types.
    * Provide template specification for types
* Change data types in Tensor class to use `int8, int16, int32, int64, float16, float32, float64, bfloat8, bfloat16, bfloat32, bfloat64` (Need to check types)
  * Allow Mixed Precision
* Add OpenMP support for CPU side
* **Hardware Support:** Potential support for AMD HIP/ROCm. — **Maybe**

## Math Engine & Tensor Operations
* **Matrix Multiplication (GEMM):** Use [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) integration.
    * Device-aware execution: Check device string to call CUDA kernel or standard Matmul.
    * Use CUDA for GEMM and Matmul.**Naive kernel is made**
* **Vector Operations:** Rectify `Transpose` for vector operations (column vs. row).
    * Apply corresponding modifications to multiplication in `autogradient.cpp`.
* **Padding:** Implement tensor padding support.
* **Caching:** Implement tensor caching to reduce memory usage.
* **Convolution:** Implement tensor convolution support.
* **Legacy/Completed Items:**
    * ~~Implement broadcasting (Required for tensor multiplication).~~
    * ~~**Problem:** Print code bug due to shape changes/data access.~~
    * ~~**Plan:** Make `broadcast` a private function so the broadcasted tensor vanishes after computation.~~
    * ~~Implement Tensor multiplication (GEMM and not Tensor Product).~~
    * ~~[Multi-dim Transpose](https://www.iaeng.org/publication/WCE2010/WCE2010_pp1838-1841.pdf)~~
    * ~~Use Kahman Summation.~~
    * ~~Address Batched Matmul (e.g., `[2,3,4] * [2,4,5] -> [2,3,5]`).~~

## Autogradient/Autodiff & Neural Network Module
* **Autodiff Engine:** Implement for activation functions (Unary and Binary ops).
* **Sequential Class:** Create a wrapper to stack layers.
* **Loss Functions:**
    * L1 Loss (MAE) — **Done**
    * L2 Loss (MSE) - **Done**
    * Log Loss/BCELossWithLogits - **Done**
    * Cross Entropy Loss
    * Softmax Loss
    * Cross Entropy Loss with Softmax
    * Binary Cross Entropy Loss with Sigmoid
* **Optimizers (Post-Loss Implementation):**
    * **SGD:** Currently implemented but slow. Add Momentum.
    * **Adam / AdamW:** Implement Eps, Betas, and Regularization.

## Data & Pipeline
* [ ] **Dataloading Pipeline:** * Support for shuffling and batching. **Will make a `Rust` based project for this to also learn**
    * Modular design for different types (Images, CSV, etc.).
* **Model Persistence:** Saving and loading pipelines.
* **Integration:** Integrate Caffe2 if needed. [Basic info about Caffe](https://builtin.com/machine-learning/caffe#:~:text=Is%20Caffe%20Still%20Used%3F,processing%2C%20computer%20vision%20and%20multimedia.)

## Testing
* [ ] Unit Tests using `gtest`

# Documentation
* [ ] Use Claude/LLM to make documentation for the repository - there are comments everywhere so should be trivial task for a LLM - although I would have to read it thoroughly

## Optimizations
* **CPU Optimizations:**
    * Vectorization using AVX/AVX2 intrinsics (e.g., `axpy` for SGD).
    * Matrix Multiplication Tiling (Cache blocking).
    * OpenMP parallelization (specifically for Matmul).
* **Graph & Performance:**
    * Lazy evaluation of the computational graph.
    * Operator Fusing (e.g., ReLU + Linear).
    * Strassen's algorithm for large matrices.
* **Advanced Features:**
    * Mixed precision training — **High Importance**
    * Dropout.
    * He initialization (to supplement current Xavier default).
* **Cleanup:** Removing Eigen operations **(Long-term)**.

## Lazy Tensor / ML Compiler: Caveats & Approaches

Caveats worth internalizing from existing tensor libraries so they're built in from the start rather than rediscovered the hard way:

* **Implicit broadcasting complicates everything downstream.** NumPy-style implicit broadcast rules are convenient in eager code but make shape inference, fusion, and buffer-liveness analysis harder to reason about — a size-1 dim could be intentional or a bug, and every fused kernel has to re-derive the broadcast pattern. This is exactly why StableHLO makes broadcasts an explicit op (`broadcast_in_dim`) instead of implicit: once building the lazy graph, consider making broadcast a real node in the IR (metadata-only) rather than something inferred inside `elementwise_binary` the way `Tensor.h` does today.
* **In-place mutation vs. a functional graph.** The eager `Tensor` has `+=`/`-=` that mutate storage. The moment a `LazyTensor` op can mutate a buffer another node already captured as input, memoization and buffer-reuse both become unsound — a cached result silently changes underneath a consumer that already ran. Frameworks that support both eager in-place ops and lazy/traced execution (PyTorch's `aten::add_` vs. functionalized ops before `torch.compile`) end up needing a whole "functionalization" pass just to undo this. Cheaper to just not expose in-place ops on `LazyTensor` at all.
* **Dynamic shapes vs. compiled specialization.** XLA (and JAX on top of it) compiles per concrete shape signature — a shape change is effectively a different program, and if shape isn't cached/keyed on, that's a recompilation storm. Even before reaching StableHLO, decide whether `LazyTensorNode` metadata assumes shapes are always static at graph-construction time (simplest, matches what's already there) or needs to support shape polymorphism later — retrofitting the latter is painful.
* **Graph breaks from host-side control flow.** Any place a lazy value gets read back into a plain C++ `if`/loop condition forces a partial materialize and fragments the graph — this is the single most common way people accidentally lose all the fusion benefit of laziness (a constant issue in both the early PyTorch/XLA bridge and `torch.compile` graph breaks). Worth being deliberate later about which NN layers can stay fully lazy through a forward pass vs. where a break is intentional.
* **Fusion isn't just "glue ops together" — it can change numerics.** A naive fusion pass that merges elementwise ops as written can be fine, but fusing something like a softmax normalization or a log-sum-exp naively (rather than using a numerically-stable fused formulation) can reintroduce the overflow/underflow bugs the eager code already avoids. Any oracle-diff test comparing fused vs. eager output should use a tolerance, not exact equality, once fusion is in play.
* **Row/col-major layout assumptions break across a fusion boundary.** `Tensor<T>` already caches `strides_col_major` for this reason at the eager level — the same care applies to the lazy graph: two ops with incompatible layouts can't be silently fused into one kernel without either a transpose node in between or a layout-assignment pass (a real pass in XLA, not incidental).

Given the eventual StableHLO path, the actual payoff of going through it rather than hand-rolling everything is that most of the above — layout assignment, CSE, dynamic-shape handling, numerically-stable fused kernels — are already solved in XLA's passes; the job is learning the IR and emission mapping well enough to hand off to them, not reimplementing every pass by hand.

## Compatibility and Containerization

* Provide a Dockerfile for building the project.[Although it should work across different platforms using `g++` and `cmake`]
* Add compatibility for `clang` and `msvc` compilers.

<!-- commented because this is the old todo - above is formatted 
## What do we need to complete

* Refactoring for cuda support - overhaul (2nd big refactor) -> - I am working on this first! **Important**
  * Write host code and kernels **kernels in progress** 
    * -> then call in the Tensor class when devices are the same and are both cuda
  * Lazy copy operation 
  * cpu and gpu memory both used -> need to think if we want to only use one (cause copy operation is expensive) -> we could use syncing though - **idk about this need to chek**

Afterwards:
  * fix the `#includes` for the repo so that cyclical dependencies and repeated includes are removed **Important**
  * Support for AMD HIP/ROCm **maybe**
  * A model saving and loading pipelines 
  * Integrate Caffe2 IF NEEDED: [Basic info about Caffe](https://builtin.com/machine-learning/caffe#:~:text=Is%20Caffe%20Still%20Used%3F,processing%2C%20computer%20vision%20and%20multimedia.)
  
  * ~~Implement broadcasting properly as it is required for tensor multiplication - there is another bug in the print code because the tensor shape changes but not the data so it accesses beyond what is allocated - **PROBLEM**~~
    * ~~Potential solution is to basically set a boolean to see if it is a broadcasted tensor or nott - PLAN IS TO MAKE BORADCAST A PRIVATE FUNCTION SO THE BROADCASTED TENSOR VANISHES AFTER COMPUTATION is APPLIED~~
  
  * Just use [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) ~~Implement Tensor multiplication (GEMM and not Tensor Product - both are infact different)~~
    * ~~[Multi dim Transpose](https://www.iaeng.org/publication/WCE2010/WCE2010_pp1838-1841.pdf)~~
    * ~~Use Kahman Summation~~
    * Check what device has been set - using a string to call a CUDA kernel or normal matmul for GEMM
      * **TODO**: Use CUDA for GEMM and Matmul - use
    * ~~The problem is not matmul or vector-mat mul or vector-vector mul - rather batched mat mul - what I now understand is how batched matmul works and I give an example here~~
      ~~`When we multiply [2,3,4] with [2,4,5] -> [2,3,5] because the first is a 2-batch of [3,4] matrices and the next one is the 2-batch of [4,5] matrices`~~
      ~~This is one reason why when doing batched mul only the last 2 indices can differ but the rest HAVE TO BE THE SAME.~~
  
  * Implement Autodiff for activation functions -> will involve using unary and binary ops so is a matter of implementing Autodiff for operations
  
  * Make Sequential Class to stack Classes in Neural Network
  
  * Implement loss functions
    * L1 loss (MAE) - done
    * L2 loss (MSE) 
    * log loss
  
  * Implement SGD and Adam (AdamW) **(After loss functions are made)**
    * SGD is done but is slow
  * Implement SGD and Adam **(After loss functions are made)**
    * Eps and Betas
    * Regularization
  
  * Rectify Transpose for vector operations as well -> column transpose or row transpose
    * Same needs to modified in multiplication in `autogradient.cpp`
  
  * A dataloading pipeline (with shuffle and batching) -> needs to modular for different data types like (images, csv, etc.)
  
  * removing Eigen operations (im lazy so I dont wanna remove it sadly :( )
  
  * adding padding
  
  * Optimizations (finally! cause we can now train a very basic neural network) -> we will come back to other loss functions in a bit as that is a mechanical process of adding operations to the autodifferentiation engine and the Tensor class
    * Momentum in SGD
    * Some CPU Optimizations
      * Vectorization and using AVX/AVX2 intrinsics for element-wise ops
        * Like an operation like `axpy` to speed `SGD` operation
      * Tiling the matrix multiplication (block/cache blocking)
      * OpenMP parallelization -> Not sure how we will do so right now -> only matmul makes sense
    * Lazy evaluation of the computational graph
    * Cache friendly tiling
    * Operator Fusing (Like ReLU and Linear)
    * Strassen's algorithm for large matrix mul.
    * Dropout
    * Mixed precision training (This is very important!!!)
    * He initialization as well -> right now Xavier is the default
-->

## Roadblocks I faced

So one of the first roadblocks that I faced is that (I have spent months on this - not completely but relatively speaking) implementing GEMM on the CPU without a prebaked library is hard.
- also need to think about subgradients and cases of discontinuities (I think PyTo


### Current Hurdle

How to make the SGD faster -> using momentum? Cause it is slow for lower learning rates
Refactoring the entire code base (cause Tensor class will change) -> cuda kernel and movement of data need to be consideredz`




### Open to Contributions
This repository is open to contribute to. Please make an issue before submitting

To clone please do the following: (for submodules)
```bash
git clone --recursive https://github.com/GodRishUniverse/BearML.git
```

### Citations [will formalize]

>
> [1] [Thank you u/brandonpelfrey](https://www.reddit.com/r/algorithms/comments/1naehk1/comment/ndpkcqr/)
>
> [2] [max() derivative formula](https://math.stackexchange.com/questions/368432/derivative-of-max-function)
>
>


### OLD

* Integrate Caffe2 IF NEEDED: [Basic info about Caffe](https://builtin.com/machine-learning/caffe#:~:text=Is%20Caffe%20Still%20Used%3F,processing%2C%20computer%20vision%20and%20multimedia.)

* ~~Implement broadcasting properly as it is required for tensor multiplication - there is another bug in the print code because the tensor shape changes but not the data so it accesses beyond what is allocated - **PROBLEM**~~
  * ~~Potential solution is to basically set a boolean to see if it is a broadcasted tensor or nott - PLAN IS TO MAKE BORADCAST A PRIVATE FUNCTION SO THE BROADCASTED TENSOR VANISHES AFTER COMPUTATION is APPLIED~~

* Just use [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) ~~Implement Tensor multiplication (GEMM and not Tensor Product - both are infact different)~~
  * ~~[Multi dim Transpose](https://www.iaeng.org/publication/WCE2010/WCE2010_pp1838-1841.pdf)~~
  * ~~Use Kahman Summation~~
  * Check what device has been set - using a string to call a CUDA kernel or normal matmul for GEMM
    * **TODO**: Use CUDA for GEMM and Matmul - use
  * ~~The problem is not matmul or vector-mat mul or vector-vector mul - rather batched mat mul - what I now understand is how batched matmul works and I give an example here~~
    ~~`When we multiply [2,3,4] with [2,4,5] -> [2,3,5] because the first is a 2-batch of [3,4] matrices and the next one is the 2-batch of [4,5] matrices`~~
    ~~This is one reason why when doing batched mul only the last 2 indices can differ but the rest HAVE TO BE THE SAME.~~
