# SimpleNet

Implementing Neural Networks in C++

## Why?

I want to implement a project in C++ and CUDA properly and this would enable me to do that.

C++ is very fast as compared to Python 3.

Inspired by [llm.c](https://github.com/karpathy/llm.c) from Andrej Karpathy and the [PyTorch](https://github.com/pytorch/pytorch) repository

## Implementation

The implementation is currently trying to implement broadcasting tensors, GEMM, tensor inversion.

A simple reverse-mode autodifferentiation pipeline is set up. Need to work on applying that for the activation functions and verify on the Tensor operations.

The Autodiff works on the

## What has been done

* Simple (double) based automatic differentiation for: addition, subtraction, multiplication and division
* Tensor library with adddition, subtraction, equivalence check

## What do we need to complete

* Integrate Caffe2 : [Basic info about Caffe](https://builtin.com/machine-learning/caffe#:~:text=Is%20Caffe%20Still%20Used%3F,processing%2C%20computer%20vision%20and%20multimedia.)
* Implement broadcasting properly as it is required for tensor multiplication - there is another bug in the print code because the tensor shape changes but not the data so it accesses beyond what is allocated - **PROBLEM**
  * Potential solution is to basically set a boolean to see if it is a broadcasted tensor or nott - PLAN IS TO MAKE BORADCAST A PRIVATE FUNCTION SO THE BROADCASTED TENSOR VANISHES AFTER COMPUTATION is APPLIED
* Implement Tensor multiplication (GEMM and not Tensor Product - both are infact different)
  * Use Kahman Summation
* Implement Autodiff for activation functions
  * Also make template specifications for them
* Check what device has been set - using a string to call a CUDA kernel or normal matmul for GEMM
  * **TODO**: Use CUDA for GEMM and Matmul
