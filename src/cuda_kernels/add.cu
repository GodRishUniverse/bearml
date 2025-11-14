#include "tensor/Tensor.h"

using ll = long long;

__global__ void addTensors(Tensor* A, Tensor* B, Tensor* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
