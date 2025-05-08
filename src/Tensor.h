#pragma once
#include <vector>
#include <iostream>

using ll = long long;

// TODO : implementation order - permute, reshape, flatten, unflatten, unsqueeze, squeeze, GEMM, unflatten


#ifndef TENSOR_H
#define TENSOR_H
namespace simplenet{
    class Tensor {
        private:
            std::vector<int> shape;
            std::vector<int> strides; // will be used in permute and in GEMM

            double * data;

            double * getTensorDataFlat() const { return this->data; };

            void computeStrides() {
                strides.resize(shape.size());
                size_t s = 1;
                for (int d = shape.size()-1; d >= 0; --d) {
                  strides[d] = s;
                  s *= shape[d];
                }
            }

        bool negOrZeroInSizeCheck(std::vector<int> sizePassedDown) const {
            for (const int & i : sizePassedDown){
                if (i < 1){
                    return true;
                }
            }
            return false;
        };

        bool sizeCheck(std::vector<int> sizePassedDown) const {
            if (negOrZeroInSizeCheck(sizePassedDown)){
                return false;
            }
            
            ll total = 1; // cause size may be huge
            for (const int & i : sizePassedDown){
                total*= i;
            }
        
            return (total == this->sizeOfTensor());
        }

        public:
            // constructor when size and data are provided
            Tensor(std::vector<int> sizePassed) {
                if (negOrZeroInSizeCheck(sizePassed)){
                    throw std::invalid_argument("Size cannot have a negative or zero");
                }
                shape = sizePassed;
                computeStrides(); // compute strides

                ll total = 1; // cause size may be huge
                for (const int & i : shape){
                    total*= i;
                }
        
                // std::cout << "Total Size: " << total<< std::endl; 
        
                data = new double[total]; 
                for (ll start = 0; start<total; start++){
                    data[start] = 0.0;
                }
            };
        
            /**
             * @brief Get the value at the given index
             * @param index The index to get the value at. Must be of size equal to the shape of the tensor.
             * @throw std::invalid_argument if the index size does not match the shape of the tensor.
             * @return The value at the given index
             */
            double get(std::vector<int> index) const {
                
                if (index.size() != shape.size()){
                    throw std::invalid_argument("Invalid index size");
                }

                if (sizeCheck(index) == false){
                    throw std::invalid_argument("Invalid index shape");
                }
        
                size_t off = 0;
                for (size_t d = 0; d < shape.size(); ++d)
                    off += index[d] * this->strides[d];
                return data[off];
            }

            void printShape() const {
                std::cout << "Shape: [";
                for (ll i = 0; i < this->shape.size(); i++){
                    std::cout << this->shape[i];
                    if (i != this->shape.size()-1){
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
            }
        
        
            void set(double val, std::vector<int> index) const {
                if (index.size() != shape.size()){
                    throw std::invalid_argument("Invalid index size");
                }

                if (sizeCheck(index) == false){
                    throw std::invalid_argument("Invalid index shape");
                }
        
                size_t off = 0;
                for (size_t d = 0; d < shape.size(); ++d)
                    off += index[d] * strides[d];
            
                data[off] = val;
            }
            // helper function
            ll sizeOfTensor() const {
                ll total = 1; // cause size may be huge
                for (const int & i : shape){
                    total*= i;
                }
                return total;
            }
        
            // copy constructor
            Tensor(const Tensor& other) {
                this->shape = other.shape;
                this->strides = other.strides;
                this->data = new double[other.sizeOfTensor()];
                std::copy(other.data, other.data + other.sizeOfTensor(), this->data);
            }
        
            // copy assignment operator
            Tensor& operator=(const Tensor& other) {
                if (this != &other) {
                    delete[] data;
                    this->shape = other.shape;
                    this->strides = other.strides;
                    this->data = new double[other.sizeOfTensor()];
                    std::copy(other.data, other.data + other.sizeOfTensor(), this->data);
                }
                return *this;
            }
        
            // move constructor
            Tensor(Tensor&& other) {
                this->shape = other.shape;
                this->data = other.data;
                this->strides = other.strides;
                other.data = nullptr;
            }
        
            // move assignment operator
            Tensor& operator=(Tensor&& other) {
                if (this != &other) {
                    delete[] data;
                    this->shape = other.shape;
                    this->data = other.data;
                    this->strides = other.strides;
                    other.data = nullptr;
                }
                return *this;
            }
        
            // destructor
            ~Tensor(){
                delete[] data;
            }
        
        
            // friend function - add tensors    
            friend Tensor operator+(const Tensor &a, const Tensor &b) {
                if (a.shape != b.shape){
                    throw std::invalid_argument("Tensors must have the same shape");
                }
                Tensor result(a.shape);
                for (ll i = 0; i < a.sizeOfTensor(); i++){
                    result.data[i] = a.data[i] + b.data[i];
                }
                return result;
            }

            Tensor operator+=(const Tensor &other) {
                if (this->shape != other.shape){
                    throw std::invalid_argument("Tensors must have the same shape");
                }
                for (ll i = 0; i < other.sizeOfTensor(); i++){
                    other.data[i] += other.data[i];
                }
                return *this;
            }


            std::vector<int> getShape() const {
                return this->shape;
            }

            std::vector<int> getStrides() const {
                return this->strides;
            }


            // TODO: friend function - print the tensor to be added
            friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
                int total_elements = 1;
                for (int dim : tensor.shape) {
                    total_elements *= dim;
                }

                os << "Tensor with shape: [";
                for (size_t i = 0; i < tensor.shape.size(); ++i) {
                    os << tensor.shape[i];
                    if (i != tensor.shape.size() - 1) os << ", ";
                }
                os << "]\n";

                os << "Tensor data: \n";
                for (int i = 0; i < total_elements; ++i) {
                    os << tensor.data[i] << " ";
                }
                os << std::endl;

                return os;
            }

            
        
            // friend function - subtract tensors
            friend Tensor operator-(const Tensor &a, const Tensor &b) {
                if (a.shape != b.shape){
                    throw std::invalid_argument("Tensors must have the same shape");
                }
                Tensor result(a.shape);
                for (ll i = 0; i < a.sizeOfTensor(); i++){
                    result.data[i] = a.data[i] - b.data[i];
                }
                return result;
            }

            Tensor operator-=(const Tensor &other) {
                if (this->shape != other.shape){
                    throw std::invalid_argument("Tensors must have the same shape");
                }
                for (ll i = 0; i < other.sizeOfTensor(); i++){
                    other.data[i] -= other.data[i];
                }
                return *this;
            }

            // Hadamard product
            Tensor operator*(const Tensor &other) {
                if (this->shape != other.shape){
                    throw std::invalid_argument("Tensors must have the same shape");
                }

                
                Tensor result(this->shape);
                for (ll i = 0; i < this->sizeOfTensor(); i++){
                    result.data[i] = this->data[i] * other.data[i];
                }
                return result;
            }
        
            // TODO: friend function - multiply tensors - dot product
            

            // Permute  
            Tensor permute(std::vector<int> new_order){
                std::vector<int > temp ;
               for (int i = 0; i < new_order.size(); i++){
                    if (new_order[i] < 0 || new_order[i] >= this->shape.size()){
                        throw std::invalid_argument("Invalid permute order");
                    }
               }
               // permuting the shape
               for (int i = 0; i < new_order.size(); i++){
                    temp.push_back(this->shape[new_order[i]]);
               }
               Tensor result(temp);
               // the numbers position will be the same - only the way they are represented will be different
               for (ll i = 0; i < this->sizeOfTensor(); i++){
                    result.data[i] = this->data[i];
               }
               return result;
            }


            // reshape 
            void reshape(std::vector<int> new_shape){
                // check multipliability
                if (sizeCheck(new_shape) == false){
                    throw std::invalid_argument("Invalid shape - needs to be multipliable to original shape");
                }
                this->shape = new_shape;
                computeStrides();
            }


            // flatten
            Tensor flatten(){
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

        
            // unsqueeze
            void unsqueeze(int dim){
                std::vector<int> temp = this->shape;
                temp.insert(temp.begin() + dim, 1);
                this->shape = temp;
                computeStrides();
            }
        
            // squeeze
            // default dim = 0
            void squeeze(int dim = 0 ){
                std::vector<int> temp = this->shape;
                int tempDim = temp[dim];
                if (tempDim == 1){
                    throw std::invalid_argument("Cannot squeeze dimension with size 1");
                }else{
                    if (dim == 0){
                        temp[dim+1]*=tempDim;
                    }else{
                        temp[dim-1]*=tempDim;
                    }
                    temp.erase(temp.begin() + dim);   
                    this->shape = temp; 
                    computeStrides();
                }                
            }
            
        
            // TODO:concat
           
        
            
            void stack(std::initializer_list<Tensor> tensors){
                if (tensors.size() == 0) {
                    throw std::invalid_argument("At least one tensor is required for stacking.");
                }

                for (const Tensor& tensor : tensors){
                    if (tensor.shape != this->shape){
                        throw std::invalid_argument("Tensors must have the same shape");
                    }
                }

                ll total = (tensors.size()+1) * this->sizeOfTensor();
                double * temp = new double[total];
                ll stride = this->sizeOfTensor();

                this->shape.insert(this->shape.begin(), tensors.size()+1);
                computeStrides();
                
                
                std::copy(this->data, this->data + stride, temp + 0);
                ll start = stride;
                for (const Tensor& tensor : tensors){
                    double * tempData = tensor.getTensorDataFlat();
                    
                    std::copy(tempData, tempData + stride, temp + start); // copy the data
                    start += stride;
                }
                
                
                if (this->data != nullptr) {
                    delete[] this->data;
                }
                this->data = temp;
            }
        
        
        };
        
}
#endif
