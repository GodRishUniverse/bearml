#include "tensor/Tensor.h"

using ll = long long;

__global__ void addTensors(Tensor* A, Tensor* B, Tensor* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // need to think about how broadcasting addition may be done on the gpu
}
