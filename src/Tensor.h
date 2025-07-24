#pragma once
#include <vector>
#include <iostream>
#include <string>

using ll = long long;

// TODO :implementation needed - division (inversion - should work for constants and matrix inversion ), unflatten, GEMM
// TODO: add constants [numbers] to tensors by adding below - WILL BE USED IN GRAD OPERATIONS
// TODO: element-wise multiply 
// TODO: element-wise divide
// TODO: overload not equal and equal to

#ifndef TENSOR_H
#define TENSOR_H
namespace simplenet{
    class Tensor {
        private:
        
            std::vector<int> shape;
            std::vector<int> strides; // will be used in permute and in GEMM

            double * data;

            double * getTensorDataFlat() const { return this->data; };

            // default constructor - added for edge cases - private ONLY
            Tensor() {};
                

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

            void computeStrides() {
                strides.resize(shape.size());
                size_t s = 1;
                for (int d = shape.size()-1; d >= 0; --d) {
                strides[d] = s;
                s *= shape[d];
                }
            }
            

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
                    std::string err = "Expected size = ";
                    std::string expected_size {"("};
                    std::string size_passed{"("};
                    for (size_t s = 0; s<shape.size(); s++){
                        expected_size+= std::to_string(shape[s]);
                        if (s-1 != shape.size()-1){
                            expected_size+= ", ";
                        }
                    }
                    expected_size+=")";

                    err+=expected_size;

                    err+= " Actual passed in = ";

                    for (size_t s = 0; s<index.size(); s++){
                        size_passed+= std::to_string(index[s]);
                        if (s-1 != index.size()-1){
                            size_passed+= ", ";
                        }
                    }
                    size_passed+=")";

                    err+=size_passed;


                    err = "Invalid index size: " + err;
                    
                    throw std::invalid_argument(err);
                }


                // check negatives and size on each index
                bool flag = false;

                std::string err = "Expected size = ";
                std::string expected_size {"("};
                std::string size_passed{"("};
                for (size_t s = 0; s<shape.size(); s++){
                    expected_size+= std::to_string(shape[s]);
                    if (s-1 != shape.size()-1){
                        expected_size+= ", ";
                    }

                    size_passed+= std::to_string(index[s]);
                    if (s-1 != index.size()-1){
                        size_passed+= ", ";
                    }

                    if (index[s] > shape[s] || index[s]<0){
                        flag = true;
                    }
                }
                expected_size+=")";

                err+=expected_size;

                err+= " Actual passed in = ";
                size_passed+=")";

                err+=size_passed;

                err= "Invalid index shape: " + err;

                if (flag){
                    throw std::invalid_argument(err);
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

            // Helper functions
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


            // TODO: Broadcasting
            static std::vector<int> computeBroadcastShape(
                const std::vector<int>& A, const std::vector<int>& B) 
            {
                size_t n = std::max(A.size(), B.size());
                std::vector<int> a(A), b(B);
                a.insert(a.begin(), n - A.size(), 1);
                b.insert(b.begin(), n - B.size(), 1);

                std::vector<int> out(n);
                for (size_t i = 0; i < n; ++i) {
                    if (a[i] == b[i] || a[i] == 1 || b[i] == 1) {
                        out[i] = std::max(a[i], b[i]);
                    } else {
                        throw std::invalid_argument("Shapes not broadcastable");
                    }
                }
                return out;
            }

            static std::vector<int> computeBroadcastStrides(
                const std::vector<int>&   origShape,
                const std::vector<int>&    origStrides,
                const std::vector<int>&   targetShape)
            {
                size_t n = targetShape.size();
                // pad on the left
                std::vector<int>  s = origShape;
                std::vector<int>  st = origStrides;
                s.insert(s.begin(), n - s.size(), 1);
                st.insert(st.begin(), n - st.size(), 0);

                std::vector<int> out(n);
                for (size_t i = 0; i < n; ++i) {
                    out[i] = (s[i] == targetShape[i] ? st[i] : 0);
                }
                return out;
            }

            static Tensor makeBroadcastView(const Tensor &t, const std::vector<int>& newShape) {
                Tensor v;            // default-constructed
                v.data    = t.data;  // points at the same buffer
                v.shape   = newShape;
                v.strides = computeBroadcastStrides(t.shape, t.strides, newShape);
                return v;
            }

                    
        
        
            // friend function - add tensors    
            friend Tensor operator+(const Tensor &A, const Tensor &B) {
                if (A.shape == B.shape) {
                    Tensor C(A.shape);
                    for (ll i = 0, N = A.sizeOfTensor(); i < N; ++i)
                        C.data[i] = A.data[i] + B.data[i];
                    return C;
                }
            
                // broadcast path
                auto outShape = computeBroadcastShape(A.shape, B.shape);
                Tensor  C(outShape);
            
                // make “broadcasted views”
                Tensor aView = makeBroadcastView(A, outShape);
                Tensor bView = makeBroadcastView(B, outShape);
            
                // now do a single flat loop
                ll N = C.sizeOfTensor();
                for (ll idx = 0; idx < N; ++idx) {
                    // decode idx → coordinates and accumulate offsets
                    ll tmp = idx, offA = 0, offB = 0;
                    for (size_t d = 0; d < outShape.size(); ++d) {
                        int coord = tmp / C.strides[d];
                        tmp %= C.strides[d];
                        offA += coord * aView.strides[d];
                        offB += coord * bView.strides[d];
                    }
                    C.data[idx] = A.data[offA] + B.data[offB];
                }
                return C;
            }

            Tensor operator+=(const Tensor &other) {
                // broadcast or exact-shape
                if (shape != other.shape) {
                    auto outShape = computeBroadcastShape(shape, other.shape);
                    // we require that *this already has exactly outShape:
                    // otherwise you'd need to reallocate or error.
                    if (shape != outShape)
                        throw std::invalid_argument("LHS must match broadcasted shape");
                    Tensor oView = makeBroadcastView(other, outShape);

                    ll N = sizeOfTensor();
                    for (ll idx = 0; idx < N; ++idx) {
                        // same flat-to-multi decode as above
                        ll tmp = idx, offO = 0;
                        for (size_t d = 0; d < outShape.size(); ++d) {
                            int coord = tmp / strides[d];
                            tmp %= strides[d];
                            offO += coord * oView.strides[d];
                        }
                        data[idx] += other.data[offO];
                    }
                } else {
                    for (ll i = 0, N = sizeOfTensor(); i < N; ++i)
                        data[i] += other.data[i];
                }
                return *this;
            }

            // element wise add
            Tensor operator+=(const double& b) {
                for (ll i = 0, N = sizeOfTensor(); i < N; ++i)
                    data[i] += b;
                return *this;
            }

            // element wise add
            friend Tensor operator+(const Tensor &A, const double& b) {         
                Tensor C(A.shape);
                for (ll i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = A.data[i] + b;
                return C;
            }

        
            // friend function - subtract tensors
            friend Tensor operator-(const Tensor &A, const Tensor &B) {
                if (A.shape == B.shape) {
                    Tensor C(A.shape);
                    for (ll i = 0, N = A.sizeOfTensor(); i < N; ++i)
                        C.data[i] = A.data[i] - B.data[i];
                    return C;
                }
            
                // broadcast path
                auto outShape = computeBroadcastShape(A.shape, B.shape);
                Tensor  C(outShape);
            
                // make “broadcasted views”
                Tensor aView = makeBroadcastView(A, outShape);
                Tensor bView = makeBroadcastView(B, outShape);
            
                // now do a single flat loop
                ll N = C.sizeOfTensor();
                for (ll idx = 0; idx < N; ++idx) {
                    // decode idx → coordinates and accumulate offsets
                    ll tmp = idx, offA = 0, offB = 0;
                    for (size_t d = 0; d < outShape.size(); ++d) {
                        int coord = tmp / C.strides[d];
                        tmp %= C.strides[d];
                        offA += coord * aView.strides[d];
                        offB += coord * bView.strides[d];
                    }
                    C.data[idx] = A.data[offA] - B.data[offB];
                }
                return C;
            }

            Tensor operator-=(const Tensor &other) {
                    // broadcast or exact-shape
                    if (shape != other.shape) {
                    auto outShape = computeBroadcastShape(shape, other.shape);
                    // we require that *this already has exactly outShape:
                    // otherwise you'd need to reallocate or error.
                    if (shape != outShape)
                        throw std::invalid_argument("LHS must match broadcasted shape");
                    Tensor oView = makeBroadcastView(other, outShape);

                    ll N = sizeOfTensor();
                    for (ll idx = 0; idx < N; ++idx) {
                        // same flat-to-multi decode as above
                        ll tmp = idx, offO = 0;
                        for (size_t d = 0; d < outShape.size(); ++d) {
                            int coord = tmp / strides[d];
                            tmp %= strides[d];
                            offO += coord * oView.strides[d];
                        }
                        data[idx] -= other.data[offO];
                    }
                } else {
                    for (ll i = 0, N = sizeOfTensor(); i < N; ++i)
                        data[i] -= other.data[i];
                }
                return *this;
            }

             // element wise subtract
            Tensor operator-=(const double& b) {
                for (ll i = 0, N = sizeOfTensor(); i < N; ++i)
                    data[i] -= b;
                return *this;
            }

              // element wise subtract
            friend Tensor operator-(const Tensor &A, const double& b) {         
                Tensor C(A.shape);
                for (ll i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = A.data[i] - b;
                return C;
            }

            // Hadamard product - //TODO: cchange operator as may clash with the friend function which is going to be used for dot product
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
            friend Tensor operator*(const Tensor &a, const Tensor &b) {
                // TODO: need to implement broadcasting first before we can do this
                // basically we need to make sure that the shapes are compatible
                // especially: Tensor a(a1, ... , an) Tensor b(b1, ... ,bn, bm) - we want at least an = bn and if a or b has less size then broadcasting will be needed
                // special cases where tensor is just a number

            }

            // element wise multiply
            friend Tensor operator*(const Tensor &A, const double& b) {         
                Tensor C(A.shape);
                for (ll i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = A.data[i]*b;
                return C;
            }
            

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

            template <typename T>
            T flatten(); // has an inplace and Tensor return type specialization in the cpp file
        
            // unsqueeze
            // Finalized
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
                //debugging
                // std::cout << "INSIDE: " << std::endl;
                // for (size_t s = 0; s <temp.size(); s++){
                //     std::cout << temp[s] <<std::endl;
                // } 

                if (temp.size() == 1){
                    throw std::invalid_argument("Cannot squeeze dimension with size 1");
                }else{
                    int tempDim = temp[dim];
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
            void concat(std::initializer_list<Tensor> tensors){

            }
           
        
            
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
