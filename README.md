# SimpleNet: a `mini-pytorch` in pure C++/CUDA

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
* refactored code
* transpose
* reduce_shape - > NEED TO TEST if it works or not

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
* Implement reduce_to_shape(Tensor grad_out, target_shape) -> **CURRENT APPROACH NEEDS TESTING**
  * We use the flatten and summation/aggregation to do so - ~~PROBLEM: sumation will change shapes so need to figure out how to do it efficiently~~


* **IMPORTANT** Got some feedback from an LLM that said i should rethink backwardop inversion to reductionop as it has too less information
  *  rethink on whether the BroadCastOp is correctly inverted to ReductionOp  or not in computeBroadcastShape

* Refactor each op’s backward_fn to compute raw grads in out_shape, then call reduce_to_shape into input.grad
  * **IMPORTANT**: we would need to BASICALLY DECLARE NEW GRAD OPS so that we can pass the order of reduce ops back to the Node class in computational graph

* Rectify Transpose for vector operations as well -> column transpose or row transpose
  * Same needs to modified in multiplication in `autogradient.h`

## Roadblocks I faced

So one of the first roadblocks that I faced is that (I have spent months on this - not completely but relatively speaking) implementing GEMM on the CPU without a prebaked library is hard.

Implementing batched multiplication wasn't as straight forward as expected as the computation for each batch coordinate was also needed to be done according to the broadcasting done - brodcasted shape and then using the strides with it

Another thing that I started with was using the identity matrix for when we do the backward pass for the Tensors, however, The Jacobian computation requires using a whole Tensor of Ones (1s) rather than an identity matrix. Moreover, using the  identity matrix is also flawed because the identity matrix only exists for square matrices and in our case we do not use square matrices - we also have rectangular matrices to consider when we change shapes.

Another roadblock was Eigen -> it used column major format rather than row major -> **so I will remove Eigen down the line and have changed it to not use Column Major Format now**: [StackOverFlow Link that verifies this](https://stackoverflow.com/questions/61140594/why-eigen-use-column-major-by-default-instead-of-row-major)

Transpose is still a roadblock... when I say transpose I mean n-dim Transpose and not matrix transpose (that is already implemented) as I'm not sure how that works (example would be something like [2,3,4,5,6,7] transposed at dim 0 and dim 3 to get [5,3,4,2,6,7]) -> Im assuming offsets will change and rest should be the same. Need to verify...


### Current Hurdle
Another roadblock in front of me is the reduction and transpose operations to be applied in the backward passes to take into account the cases for broadcasting operations. I have implemented tranpose, but I need to think about how to segregate ops (addition, subtraction, multiplication) for gradients in Tensor class as we need to pass the shapes to autogradient to do the reduce method to get grads backward.
