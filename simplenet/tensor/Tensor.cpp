#include "Tensor.h"

namespace simplenet{

    // flatten
    Tensor Tensor::flatten(int start_dim, int end_dim, bool keepdims) {
        Tensor result = *this; // copy the tensor
        result.setShape(Tensor::flatten_(start_dim, end_dim, keepdims));
        result.computeStrides();
        return result;
    }
}
