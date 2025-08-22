#include "Tensor.h"

namespace simplenet{

    // flatten
    template<>
    Tensor Tensor::flatten<Tensor>(int start_dim, int end_dim){

        // default value of end_dim is -1
        if (end_dim == -1){
            end_dim = this->shape.size()-1;
        }

        if (start_dim <0 || start_dim>=this->shape.size() || start_dim>end_dim ||  end_dim>=this->shape.size() || end_dim  < 0){
            throw std::invalid_argument("Start Dim and End Dims not the appropriate ranges-> CHECK");
        }

        std::vector<int> temp;
        ll total = 1;
        for (ll i = 0; i < this->shape.size(); i++){
            if (i<start_dim || i>end_dim){
                temp.push_back(this->shape[i]);
            } else {
                total *= this->shape[i];
                if (i == end_dim) temp.push_back(total);
            }
        }
        Tensor result(temp); // we do this so that the strides are correctly calculate for the new tensor
        for (ll i = 0; i < this->sizeOfTensor(); i++){
            result.data[i] = this->data[i];
        }
        return result;

    }

    //flatten - in place
    template<>
    void Tensor::flatten<void>(int start_dim, int end_dim){
        // default value of end_dim is -1
        if (end_dim == -1){
            end_dim = this->shape.size()-1;
        }

        if (start_dim <0 || start_dim>=this->shape.size() || start_dim>end_dim ||  end_dim>=this->shape.size() || end_dim  < 0){
            throw std::invalid_argument("Start Dim and End Dims not the appropriate ranges-> CHECK");
        }

        std::vector<int> temp;
        ll total = 1;
        for (ll i = 0; i < this->shape.size(); i++){
            if (i<start_dim || i>end_dim){
                temp.push_back(this->shape[i]);
            } else {
                total *= this->shape[i];
                if (i == end_dim) temp.push_back(total);
            }
        }

        this->shape =temp;
        computeStrides();
    }
}
