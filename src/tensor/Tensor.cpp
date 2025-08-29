#include "Tensor.h"

namespace simplenet{


    // flatten
    Tensor Tensor::flatten(int start_dim, int end_dim, bool keepdims) {
        Tensor result(Tensor::flatten_(start_dim, end_dim, keepdims)); // we do this so that the strides are correctly calculate for the new tensor
        for (ll i = 0; i < this->sizeOfTensor(); i++){
            result.data[i] = this->data[i];
        }
        return result;
    }
}
