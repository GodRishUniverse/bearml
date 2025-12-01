# SimpleNet: a `mini-pytorch` in pure C++/CUDA

Deep Learning Framework: Implementing Neural Networks in C++

## Why?

I want to implement a project in C++ and CUDA properly and this would enable me to do that.

C++ is very fast as compared to Python 3.

Inspired by [llm.c](https://github.com/karpathy/llm.c) from Andrej Karpathy and the [PyTorch](https://github.com/pytorch/pytorch) repository

## Quick Start

### Prerequisites
**Only tested on Fedora Linux at the moment**
- C++20 or higher
- CMake 3.20+
- CUDA Toolkit 13.0 [will have to make optional]

### Building

```bash
git clone https://github.com/GodRishUniverse/SimpleNet.git
cd SimpleNet
mkdir build && cd build
cmake ..
make
```

Please edit the CMake files according to your liking.


## Implementation

The implementation is currently trying to implement broadcasting tensors, GEMM, CUDA support (**Refactoring in Progress**).

A simple reverse-mode autodifferentiation pipeline is set up. Need to work on applying that for the activation functions and verify on the Tensor operations.

The Autodiff works on the

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


## What do we need to complete

* Refactoring for cuda support - overhaul (2nd big refactor) -> - I am working on this first! **Important**
  * Write host code and kernels **kernels in progress** 
    * -> then call in the Tensor class when devices are the same and are both cuda
  * Lazy copy operation 
  * cpu and gpu memory both used -> need to think if we want to only use one (cause copy operation is expensive) -> we could use syncing though - **idk about this need to chek**

Afterwards:
  
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
    * Same needs to modified in multiplication in `autogradient.h`
  
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

## Roadblocks I faced

So one of the first roadblocks that I faced is that (I have spent months on this - not completely but relatively speaking) implementing GEMM on the CPU without a prebaked library is hard.
- also need to think about subgradients and cases of discontinuities (I think PyTo


### Current Hurdle

How to make the SGD faster -> using momentum? Cause it is slow for lower learning rates
Refactoring the entire code base (cause Tensor class will change) -> cuda kernel and movement of data need to be consideredz`




### Open to Contributions
This repository is open to contribute to. Please make an issue before submitting

### Citations [will formalize]

> [1] [Thank you Reddit!](https://www.reddit.com/r/algorithms/comments/1naehk1/comment/ndpkcqr/)
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
