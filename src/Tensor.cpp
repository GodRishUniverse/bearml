#include "Tensor.h"
#include <iostream> 

namespace simplenet{

    // flatten
    template<>
    Tensor Tensor::flatten<Tensor>(){
        // We already have a 1D tensor - we have to copy and return
        std::vector<int> temp;
        ll total = 1;
        for (ll i = 0; i < this->shape.size(); i++){
            total *= this->shape[i];
        }
        temp.push_back(total);
        Tensor result(temp);
        for (ll i = 0; i < this->sizeOfTensor(); i++){
            result.data[i] = this->data[i];
        }
        return result;
    }

    //flatten - in place
    template<>
    void Tensor::flatten<void>(){
        int flattened_shape {1};
        for (size_t s = 0; s < this->shape.size(); s++){
            flattened_shape *= this->shape[s];
        }

        this->shape =std::vector<int> {flattened_shape};
        computeStrides();
    }
    
}
