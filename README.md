# SimpleNet

Deep Learning Framework: Implementing Neural Networks in C++

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
* Tensor library with adddition, subtraction, equivalence check, broadcasting is applied to both addition and subtraction
* Batched Matrix Multiplication for the CPU was implemented
* Broadcasting of tensors for multiplication and addition and subtraction was implemented
* flattening operations were extended to match how Pytorch flatten works with keepdims and start and end dimension specifications

## What do we need to complete

* Integrate Caffe2 : [Basic info about Caffe](https://builtin.com/machine-learning/caffe#:~:text=Is%20Caffe%20Still%20Used%3F,processing%2C%20computer%20vision%20and%20multimedia.)
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
* Implement Autodiff for activation functions
  * **FIX AUTODIFF Backward functions for each operation to check for Tensors - IMPORTANT**
  * Also make template specifications for them
* Implement reduce_to_shape(Tensor grad_out, target_shape)
  * We use the flatten and summation/aggregation to do so - PROBLEM: sumation will change shapes so need to figure out how to do it efficiently

* Refactor each op’s backward_fn to compute raw grads in out_shape, then call reduce_to_shape into input.grad.
* Matmul backward: unbatched first, then batched + broadcasted batches (reduce back).


## Roadblocks I faced

So one of the first roadblocks that I faced is that (I have spent months on this - not completely but relatively speaking) implementing GEMM on the CPU without a prebaked library is hard.

Implementing batched multiplication wasn't as straight forward as expected as the computation for each batch coordinate was also needed to be done according to the broadcasting done - brodcasted shape and then using the strides with it

Another thing that I started with was using the identity matrix for when we do the backward pass for the Tensors, however, The Jacobian computation requires using a whole Tensor of Ones (1s) rather than an identity matrix. Moreover, using the  identity matrix is also flawed because the identity matrix only exists for square matrices and in our case we do not use square matrices - we also have rectangular matrices to consider when we change shapes.


## Acknowledgements
I would like to thank Dr. Usman Alim at the University of Calgary to help me rethink my backward pass (using jacobians)

I would like to thank Anthropic for their Claude models as I used the "learning" style to clear my thoughts and help me understand better (especially batched mat mul and reduce operations - my understanding had very large gaps that it helped me fill in)
