#include "Tensor.h"

namespace simplenet{

    size_t Tensor::print_precision = 14;

    // flatten
    Tensor Tensor::flatten(int start_dim, int end_dim, bool keepdims) {
        Tensor result = *this; // copy the tensor
        std::vector<int> newShape = Tensor::flatten_(start_dim, end_dim, keepdims);
        result.setShape(newShape);
        result.computeStrides();
        return result;
    }
}
