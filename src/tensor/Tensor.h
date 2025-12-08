#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>

// TODO: refactor Tensor class
#include "devices/device_type.h"
#include "devices/device_allocator.h"


#include "Eigen/Dense" // IMPORTING eigen for BLAS functions
#include "Eigen/src/Core/Matrix.h"

#include "utils/shape_utils.h"
#include "utils/debug_utils.h"
#include "utils/linalg_utils.h"
#include "utils/device_utils.h"

// #include <cmath>
#include <iomanip>

using ll = long long; // can also use int_fast64_t
using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// TODO :implementation needed - division (inversion - should work for constants and matrix inversion ), unflatten, GEMM
// TODO: element-wise divide
// TODO: allow float values as well (32-bit precision) - template specialize equal to
// TODO: do max and min operations -> comparing doubles and same size Tensors



#ifndef TENSOR_H
#define TENSOR_H
namespace simplenet{

    // forward declaration
    namespace linear_algebra {
       Tensor batchedMatMul(const Tensor& a, const Tensor& b); // Forward declare the friend function
       Tensor reduce(const Tensor& a, std::vector<int>& afterShape); // forward declare for the friend reduce
       Tensor hadamard(const Tensor &a, const Tensor &other);

       // Tensor and Tensor
       Tensor mask_of_greater_than_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_greater_than(const Tensor& first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_less_than_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_less_than(const Tensor& first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val);

       // Double and Tensor
       Tensor mask_of_greater_than_equal_to(double first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_greater_than(double first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_less_than_equal_to(double first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_less_than(double first, const Tensor& other,  double first_val, double second_val);
       Tensor mask_of_equal_to(double first, const Tensor& other,  double first_val, double second_val);

       // Tensor and Double
       Tensor mask_of_greater_than_equal_to(const Tensor& first, double other,  double first_val, double second_val);
       Tensor mask_of_greater_than(const Tensor& first, double other,  double first_val, double second_val);
       Tensor mask_of_less_than_equal_to(const Tensor& first, double other,  double first_val, double second_val);
       Tensor mask_of_less_than(const Tensor& first, double other,  double first_val, double second_val);
       Tensor mask_of_equal_to(const Tensor& first, double other,  double first_val, double second_val);

       Tensor sign(const Tensor& a);
    }

    class Tensor {

        // ==============================PRIVATE========================================
        private:
            static constexpr double MIN_DIFF = 1e-12;
            std::vector<int> shape;
            std::vector<int> strides; // will be used in permute and in GEMM
            double * data; // need to change to Tensor<T> where T can be custom data types like int8, float16, float32, float64 (double)
            bool owns_data;  // NEEDED To not cause the double destructor deletion of the broadcasting methods
            Device device;
            std::unique_ptr<DeviceAllocator> allocator_; // the device allocator we will be using for moving -> we only want a unique allocator


            // default constructor - added for edge cases - private ONLY -> cpu only allocation
            Tensor() : data(nullptr), owns_data(false), device(Device(DeviceType::CPU, -1)) , allocator_(nullptr){};

            static Tensor makeBroadcastView(const Tensor &t, const std::vector<int>& newShape) {
                Tensor v;            // default-constructed
                v.device = t.device; // same device (broadcasting)
                v.data    = t.data;
                v.shape   = newShape;
                v.strides = utils::computeBroadcastStrides(t.shape, t.strides, newShape); // no need to change this
                v.owns_data = false; // we do not want the broadcasted tensors to own the data that it points - so we do not double delete
                v.allocator_ = nullptr; // we do not have an allocator for the views
                return v;
            }

            // TODO - refactor
            static bool isScalar(const Tensor& t) {
                if (t.shape.empty()) return true;
                for (int dim : t.shape) {
                    if (dim != 1) return false;
                }
                return true;
            }

            // TODO - refactor
            static double getScalarValue(const Tensor& t){
                return t.data[0];
            }

            std::vector<int> flatten_(int start_dim =0, int end_dim = -1, bool keepdims=false)
                // has a Tensor return type specialization in the cpp file
            {

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
                    if (keepdims){
                        if (i>=start_dim && i<=end_dim){
                            total *= this->shape[i];
                            temp.push_back((i ==end_dim) ? total: 1);
                        } else{
                            temp.push_back(this->shape[i]);
                        }
                    } else{
                        if (i<start_dim || i>end_dim){
                            temp.push_back(this->shape[i]);
                        } else {
                            total *= this->shape[i];
                            if (i == end_dim) temp.push_back(total);
                        }
                    }

                }

                return temp;
            }

            // ==============================PRIVATE========================================

        public:

            // constructor when size and data are provided
            Tensor(std::vector<int> sizePassed, const Device& device = Device::cpu()) :shape(sizePassed),  owns_data(true), device(device){ // we own the data here
                // shape cannot have 0 or negatives in it
                if (utils::negOrZeroInSizeCheck(this->shape)){
                    throw std::invalid_argument("Size cannot have a negative or zero");
                }

                // check if we even have a cuda capable device
                if (device.type== DeviceType::CUDA){
                    int cuda_device_count = 0;
                    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
                    if (cuda_device_count == 0) {
                        throw std::invalid_argument("No CUDA-capable devices available");
                    }
                }
                computeStrides(); // compute strides

                this->allocator_.reset(get_allocator(this->device));

                size_t sizeTensor = sizeOfTensor();
                size_t bytes = sizeTensor*sizeof(double);
                this->data = static_cast<double*>(allocator_->allocate(bytes));

                // initialize to zero
                if (this->device.is_cpu()) {
                    std::fill_n(this->data, sizeTensor, 0.0);
                } else {
                    // Zero initialize on GPU
                    CUDA_CHECK(cudaMemset(this->data, 0.0, bytes));
                }

            };


            // copy constructor
            Tensor(const Tensor& other) : owns_data(true), shape(other.shape), device(other.device), strides(other.strides){ // we own the data when we copy;
                this->allocator_.reset(get_allocator(device));
                size_t bytes = sizeOfTensor() * sizeof(double);
                this->data =static_cast<double*>(allocator_->allocate(bytes));
                if (this->device == other.device) {
                    // cpu to cpu or gpu to gpu
                    allocator_->copy_device_to_device(this->data, other.data, bytes);
                } else {
                    if (this->device.is_cpu()) {
                        // from device to cpu
                        other.allocator_->copy_to_host(this->data, other.data, bytes);
                    } else {
                        // from cpu to device
                        this->allocator_->copy_to_device(this->data, other.data, bytes);
                    }
                }
                // std::copy(other.data, other.data + other.sizeOfTensor(), this->data);
            }

            // copy assignment operator
            Tensor& operator=(const Tensor& other) {
                if (this != &other) {
                    if (this->owns_data && this->data && this->allocator_) {
                        allocator_->deallocate(this->data);
                    }
                    this->shape = other.shape;
                    this->device = other.device;
                    this->strides = other.strides;
                    this->owns_data = true;  // Copy always owns its data

                    this->allocator_.reset(get_allocator(device));
                    size_t bytes = sizeOfTensor() * sizeof(double);
                    this->data =static_cast<double*>(allocator_->allocate(bytes));
                    if (this->device == other.device) {
                        // cpu to cpu or gpu to gpu
                        allocator_->copy_device_to_device(this->data, other.data, bytes);
                    } else {
                        if (this->device.is_cpu()) {
                            // from device to cpu
                            other.allocator_->copy_to_host(this->data, other.data, bytes);
                        } else {
                            // from cpu to device
                            this->allocator_->copy_to_device(this->data, other.data, bytes);
                        }
                    }
                }
                return *this;
            }

            // move constructor
            Tensor(Tensor&& other) noexcept : owns_data(other.owns_data), shape(std::move(other.shape)), strides(std::move(other.strides)), device(other.device), data(other.data), allocator_(std::move(other.allocator_)){
                // this->shape = other.shape;
                // this->data = other.data;
                // this->device = other.device;
                // this->strides = other.strides;
                other.data = nullptr;
                other.owns_data = false;
            }

            // move assignment operator
            Tensor& operator=(Tensor&& other) noexcept {
                if (this != &other) {

                    if (this->owns_data && this->data && this->allocator_) {
                        allocator_->deallocate(this->data);
                    }

                    this->shape = std::move(other.shape);
                    this->data = other.data;
                    this->device = other.device;
                    this->strides = std::move(other.strides);
                    this->allocator_ = std::move(other.allocator_);
                    this->owns_data = other.owns_data;
                    other.data = nullptr;
                    other.owns_data = false;
                }
                return *this;
            }

            // destructor
            ~Tensor(){
                // standard cleanup
                if (this->owns_data && this->data && this->allocator_) {
                    allocator_->deallocate(this->data);
                }
            }

            // copy to
            Tensor to(const Device& targetDevice) const{
                if (device == targetDevice) {
                    return *this; // Return copy on same device
                }
                Tensor result(shape, targetDevice);
                size_t bytes = sizeOfTensor() * sizeof(double);
                if (targetDevice.is_cpu()) {
                    // GPU -> CPU
                    allocator_->copy_to_host(result.data, data, bytes);
                } else {
                    // CPU -> GPU
                    allocator_->copy_to_device(result.data, data, bytes);
                }
                return result;
            }


            // inplace to
            void to_(const Device& targetDevice) {
                if (device == targetDevice) return;

                size_t bytes = sizeOfTensor() * sizeof(double);
                auto new_allocator = std::unique_ptr<DeviceAllocator>(get_allocator(targetDevice));
                double* new_data = static_cast<double*>(new_allocator->allocate(bytes));

                // transfer data
                if (targetDevice.is_cpu()) {
                    allocator_->copy_to_host(new_data, data, bytes);
                } else {
                    new_allocator->copy_to_device(new_data, data, bytes);
                }

                if (owns_data) {
                    allocator_->deallocate(data);
                }

                data = new_data;
                device = targetDevice;
                allocator_ = std::move(new_allocator); // new unique pointer setting
                owns_data = true;
            }

            Device getDevice() const {
                return this->device;
            }


            double get(std::vector<int> index) const {
                if (!this->device.is_cpu()) {
                    throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use get()");
                }
                if (index.size() != shape.size()){
                    throw std::invalid_argument("Invalid index size: \nPassed:" + utils::debugShapes(index)+"\nExpected:" +utils::debugShapes(this->shape)+"\n");
                }

                if (utils::isIndexValid(index, this->shape) == false){
                    throw std::invalid_argument("Invalid index shape: \nPassed" + utils::debugShapes(index)+"\nExpected:" + utils::debugShapes(this->shape)+"\n");
                }

                size_t off = 0;
                for (size_t d = 0; d < shape.size(); ++d)
                    off += index[d] * this->strides[d];
                return data[off];
            }


            void set(double val, std::vector<int> index) const {

                if (!this->device.is_cpu()) {
                    throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use set()");
                }

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

            // TODO: refactor
            // TODO: add test to check offset values
            void set_with_offset(ll offset, int row, int col, double val){
                // assumes offset is passed correctly at the moment
                if (!this->device.is_cpu()) {
                    throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use set()");
                }

                data[offset+row*(this->shape[this->shape.size()-1])+col] = val;
            }

            // helper function
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

            // helper function
            size_t sizeOfTensor() const {
                size_t total = 1; // cause size may be huge
                for (const int & i : shape){
                    total*= i;
                }
                return total;
            }

            // helper functions
            std::vector<int> getShape() const {
                return this->shape;
            }

            // helper function
            std::vector<int> getStrides() const {
                return this->strides;
            }

            void computeStrides() {
                strides.resize(shape.size());
                size_t s = 1;
                for (int d = shape.size()-1; d >= 0; --d) {
                    strides[d] = s;
                    s *= shape[d];
                }
            }


            // TODO:CHANGE to not print extra stuff after the boradcast is applied as stride is set to 0 after broadcast is done
            friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {

                Tensor cpu_tensor_view = tensor.to(Device::cpu());


                size_t total_elements = 1;
                for (int dim : tensor.shape) {
                    total_elements *= dim;
                }

                os << "Tensor with shape: [";
                for (size_t i = 0; i < tensor.shape.size(); ++i) {
                    os << tensor.shape[i];
                    if (i != tensor.shape.size() - 1) os << ", ";
                }
                os << "]\n";

                os << "Tensor on device: ";
                os << tensor.device.to_string() << "\n";

                os << "Tensor data: \n";
                for (size_t i = 0; i < total_elements; ++i) {
                    os << std::setprecision(14) << cpu_tensor_view.data[i] << " ";
                }
                os << "\n";

                os << "OWNERSHIP: " << ((tensor.owns_data) ? "TRUE" : "FALSE") << std::endl;

                return os;
            }

            // TODO: REFACTOR ALL OPERATIONS for CUDA support as well
            // ================================ OPERATIONS ========================================================================

            // --------------------------------ADDITION----------------------------------------------------------------------------
            // friend function - add tensors
            friend Tensor operator+(const Tensor &A, const Tensor &B) {
                utils::errorCheckSameDevice(A, B); // will throw an error if devices don't match

                if (A.shape == B.shape) {
                    Tensor C(A.shape);
                    for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                        C.data[i] = A.data[i] + B.data[i];
                    return C;
                }

                // broadcast path
                auto outShape = utils::computeBroadcastShape(A.shape, B.shape);
                Tensor  C(outShape);

                // make “broadcasted views” - we copy as we do not want the original shapes and strides of A and B to change
                Tensor aView = makeBroadcastView(A, outShape);
                Tensor bView = makeBroadcastView(B, outShape);

                // now do a single flat loop
                size_t N = C.sizeOfTensor();
                for (ll idx = 0; idx < N; ++idx) {
                    // decode idx → coordinates and accumulate offsets
                    ll tmp = idx, offA = 0, offB = 0;
                    std::vector<int> coords(outShape.size());
                    // Decode coordinates from rightmost to leftmost
                    for (int d = outShape.size() - 1; d >= 0; --d) {
                        coords[d] = tmp % outShape[d];
                        tmp /= outShape[d];
                    }

                    // Now compute offsets using broadcast strides
                    for (size_t d = 0; d < outShape.size(); ++d) {
                        offA += coords[d] * aView.strides[d];
                        offB += coords[d] * bView.strides[d];
                    }
                    C.data[idx] = A.data[offA] + B.data[offB];
                }
                return C;
            }

            Tensor& operator+=(const Tensor &other) {
                utils::errorCheckSameDevice(*this, other); // will throw an error if devices don't match

                // broadcast or exact-shape
                if (shape != other.shape) {

                    auto outShape = utils::computeBroadcastShape(shape, other.shape);
                    // we require that *this already has exactly outShape:
                    // otherwise you'd need to reallocate or error.
                    if (shape != outShape)
                        throw std::invalid_argument("LHS must match broadcasted shape");
                    Tensor oView = makeBroadcastView(other, outShape);


                    size_t N = sizeOfTensor();
                    for (size_t idx = 0; idx < N; ++idx) {
                        // same flat-to-multi decode as above

                        std::vector<int> coords(outShape.size(), 0);
                        ll tmp = idx, offO = 0;

                        for (int d = outShape.size() - 1; d >= 0; --d) {
                            coords[d] = tmp % outShape[d];
                            tmp /= outShape[d];
                        }

                        // Now compute offsets using broadcast strides
                        for (size_t d = 0; d < outShape.size(); ++d) {
                            offO += coords[d] * oView.strides[d];
                        }

                        data[idx] += oView.data[offO];
                    }
                } else {
                    for (size_t i = 0, N = sizeOfTensor(); i < N; ++i)
                        data[i] += other.data[i];
                }
                return *this;
            }

            // element wise add
            Tensor& operator+=(const double& b) {
                for (size_t i = 0, N = sizeOfTensor(); i < N; ++i)
                    data[i] += b;
                return *this;
            }

            // element wise add
            friend Tensor operator+(const Tensor &A, const double& b) {
                Tensor C(A.shape);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = A.data[i] + b;
                return C;
            }

            friend Tensor operator+(const double& b, const Tensor &A) {
                return A+b; // same as above
            }


            // --------------------------------SUBTRACTION-------------------------------------------------------------------------



            // friend function - subtract tensors
            friend Tensor operator-(const Tensor &A, const Tensor &B) {
                if (A.shape == B.shape) {
                    Tensor C(A.shape);
                    for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                        C.data[i] = A.data[i] - B.data[i];
                    return C;
                }

                // broadcast path
                auto outShape = utils::computeBroadcastShape(A.shape, B.shape);
                Tensor  C(outShape);

                // make “broadcasted views”
                Tensor aView = makeBroadcastView(A, outShape);
                Tensor bView = makeBroadcastView(B, outShape);

                // now do a single flat loop
                size_t N = C.sizeOfTensor();
                for (ll idx = 0; idx < N; ++idx) {
                    // decode idx → coordinates and accumulate offsets
                    ll tmp = idx, offA = 0, offB = 0;
                    std::vector<int> coords(outShape.size());
                    // Decode coordinates from rightmost to leftmost
                    for (int d = outShape.size() - 1; d >= 0; --d) {
                        coords[d] = tmp % outShape[d];
                        tmp /= outShape[d];
                    }

                    // Now compute offsets using broadcast strides
                    for (size_t d = 0; d < outShape.size(); ++d) {
                        offA += coords[d] * aView.strides[d];
                        offB += coords[d] * bView.strides[d];
                    }
                    C.data[idx] = A.data[offA] - B.data[offB];
                }
                return C;
            }

            Tensor& operator-=(const Tensor &other) {
                    // broadcast or exact-shape
                    if (shape != other.shape) {
                    auto outShape = utils::computeBroadcastShape(shape, other.shape);
                    // we require that *this already has exactly outShape:
                    // otherwise you'd need to reallocate or error.
                    if (shape != outShape)
                        throw std::invalid_argument("LHS must match broadcasted shape");
                    Tensor oView = makeBroadcastView(other, outShape);

                    size_t N = sizeOfTensor();
                    for (size_t idx = 0; idx < N; ++idx) {
                        // same flat-to-multi decode as above

                        std::vector<int> coords(outShape.size(), 0);
                        ll tmp = idx, offO = 0;

                        for (int d = outShape.size() - 1; d >= 0; --d) {
                            coords[d] = tmp % outShape[d];
                            tmp /= outShape[d];
                        }

                        // Now compute offsets using broadcast strides
                        for (size_t d = 0; d < outShape.size(); ++d) {
                            offO += coords[d] * oView.strides[d];
                        }

                        data[idx] -= oView.data[offO];
                    }
                } else {
                    for (size_t i = 0, N = sizeOfTensor(); i < N; ++i)
                        data[i] -= other.data[i];
                }
                return *this;
            }

             // element wise subtract
            Tensor& operator-=(const double& b) {
                for (size_t i = 0, N = sizeOfTensor(); i < N; ++i)
                    data[i] -= b;
                return *this;
            }

              // element wise subtract
            friend Tensor operator-(const Tensor &A, const double& b) {
                Tensor C(A.shape);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = A.data[i] - b;
                return C;
            }

            // element wise subtract
            friend Tensor operator-(const double& b, const Tensor &A) {
                Tensor C(A.shape);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = b- A.data[i]; // operation is switched as above
                return C;
            }

            // --------------------------------MULTIPLICATION----------------------------------------------------------------------


            // Hadamard product
            friend Tensor linear_algebra::hadamard(const Tensor &a, const Tensor &other);


            // THIS is where we will be doing the multiplication when the dimensions exceed the normal 2 of a matrix
            friend Tensor linear_algebra::batchedMatMul(const Tensor& a, const Tensor& b);

            friend Tensor linear_algebra::reduce(const Tensor& a, std::vector<int>& afterShape);


            // element wise multiply
            friend Tensor operator*(const Tensor &A, const double& b) {
                Tensor C(A.shape);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = A.data[i]*b;
                return C;
            }
            // for when the operations are reversed
            friend Tensor operator*(const double& b, const Tensor &A) {
                return A*b; // order of multiplication doesn't matter here -> does it?
            }



            friend Tensor operator*(const Tensor &a, const Tensor &b) {

                // 5 cases that need to be checked for this
                //                  case 0: A and B are scalars in the form of Tensors [TAKEN INSIDE CASE 1 and 2]
                // case 1: A is scalar
                // case 2: B is scalar
                // case 3: A is a vector and B is a vector
                // case 4: A is matrix and B is a vector
                // case 5: matrix multiplication - batched and unbatched (GEMM operations)

                // CASE 1
                if (isScalar(a)){
                    return b*getScalarValue(a); // b is not a scalar - we call our friend function (created above for element wise mult) here
                }

                // CASE 2
                if (isScalar(b)){
                    return a*getScalarValue(b); // a is not a scalar - we call our friend function (created above for element wise mult) here
                }

                std::vector<int> a_shape = a.getShape();
                std::vector<int> b_shape = b.getShape();

                // CASE 3: vector-vector product - dot product
                if (a_shape.size() == 1 && b_shape.size()==1){
                    if (a_shape[0] != b_shape[0]) {
                        // TODO: put debugPrints
                        throw std::invalid_argument("Vector dimensions must match for dot product");
                    }
                    // matches - calling Eigen
                    Eigen::Map<const Eigen::VectorXd> vec_a(a.data, a_shape[0]);
                    Eigen::Map<const Eigen::VectorXd> vec_b(b.data, b_shape[0]);

                    Tensor output({1});
                    output.data[0] = vec_a.dot(vec_b);
                    return output;
                }

                // CASE 4: matrix n by m multiplied with m by 1 vector
                if (a_shape.size() == 2 && b_shape.size() == 1) {
                    if (a_shape[1] != b_shape[0]) {
                        // TODO: put debugPrints
                        throw std::invalid_argument("Matrix columns must match vector size");
                    }
                    // matches - calling Eigen
                    Eigen::Map<const MatrixRowMajor> mat_a(a.data, a_shape[0], a_shape[1]);
                    Eigen::Map<const Eigen::VectorXd> vec_b(b.data, b_shape[0]);

                    Tensor result({a_shape[0]});
                    Eigen::Map<Eigen::VectorXd> result_vec(result.data, a_shape[0]);
                    result_vec = mat_a * vec_b;

                    return result;
                }

                // CASE 5: unbatched matmul
                if (a_shape.size() == 2 && b_shape.size() == 2) {
                    if (a_shape[1] != b_shape[0]) {
                        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
                    }

                    Eigen::Map<const MatrixRowMajor> mat_a(a.data, a_shape[0], a_shape[1]);
                    Eigen::Map<const MatrixRowMajor> mat_b(b.data, b_shape[0], b_shape[1]);

                    Tensor result({a_shape[0], b_shape[1]});
                    Eigen::Map<MatrixRowMajor> result_mat(result.data, a_shape[0], b_shape[1]);
                    result_mat = (mat_a * mat_b);

                    return result;
                }

                // CASE 5: batched batched matmul
                if (a_shape.size() >= 2 && b_shape.size() >= 2) {
                    return linear_algebra::batchedMatMul(a, b);
                }

                // SHOULD NEVER REACH HERE
                throw std::invalid_argument("SHOULD NOT REACH HERE - Unsupported tensor shapes for multiplication");
            }

            // In-place matrix multiplication - we call our function made above
            Tensor& operator*=(const Tensor& B) {
                 *this = *this * B;
                 return *this;
            }
            // calls the element wise mul
            Tensor& operator*=(const double& B) {
                 *this = *this * B;
                 return *this;
            }

            // -------------------------------DIVISION--------------------------------------------------------------------------

            // element wise divide
            friend Tensor operator/(const Tensor &A, const double& b) {
                Tensor C(A.shape);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = A.data[i]/b;
                return C;
            }

            // element wise divide
            friend Tensor operator/(const double& b, const Tensor &A) {
                Tensor C(A.shape);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = b/A.data[i];
                return C;
            }


            friend Tensor operator/(const Tensor &a, const Tensor &b) {
                if (b.shape.size() ==1 || a.shape.size() ==1){
                    return a/b;
                }

                if (b.shape != a.shape){
                    throw std::runtime_error("Shapes should match for element-wise divide");
                }

                // shapes match
                Tensor C(a.shape);
                for (size_t i = 0, N = a.sizeOfTensor(); i < N; ++i)
                    C.data[i] = a.data[i]/b.data[i];
                return C;

                // TODO: Another case exists but that is when matrix b is invertible and then it just becomes matrix mul
            }



            // --------------------------------EQUALITY--------------------------------------------------------------------------


            // will change as float precision is added
            friend bool operator==(const Tensor &a, const Tensor &b){
                if (a.getShape() == b.getShape() && a.getStrides() == b.getStrides()){
                    // NOTE: std::abs is better for doubles
                    for (size_t i = 0; i<a.sizeOfTensor(); i++){
                        if ((std::abs(a.data[i]-b.data[i])) >= Tensor::MIN_DIFF){ // check if the error is greater than 10^-15
                            return false;
                        }
                    }
                    return true;
                }
                return false;
            }


            friend bool operator!=(const Tensor &a, const Tensor &b){
                return !(a==b);
            }


            // -----------------------------------------------Operations helpful in mask generations------------------
            friend Tensor linear_algebra::mask_of_greater_than_equal_to(const Tensor& first, const Tensor& other, double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_greater_than(const Tensor& first, const Tensor& other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_less_than_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_less_than(const Tensor& first, const Tensor& other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val);



            // Double and Tensor
            friend Tensor linear_algebra::mask_of_greater_than_equal_to(double first, const Tensor& other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_greater_than(double first, const Tensor& other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_less_than_equal_to(double first, const Tensor& other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_less_than(double first, const Tensor& other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_equal_to(double first, const Tensor& other,  double first_val, double second_val);

            // Tensor and Double
            friend Tensor linear_algebra::mask_of_greater_than_equal_to(const Tensor& first, double other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_greater_than(const Tensor& first, double other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_less_than_equal_to(const Tensor& first, double other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_less_than(const Tensor& first, double other,  double first_val, double second_val);
            friend Tensor linear_algebra::mask_of_equal_to(const Tensor& first, double other,  double first_val, double second_val);

            // sign matrix
            friend Tensor linear_algebra::sign(const Tensor& a);


            //----------------------------------------Exponential------------------------------------------------------
            static Tensor exp(Tensor& t){
                // std::cout <<"EXPONENTIATED" <<std::endl;
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::exp(t.data[i]);
                }
                return a;
            }


            //----------------------------------------Max and Min------------------------------------------------------
            static Tensor max(Tensor& t, double val){
                // std::cout <<"MAX" <<std::endl;
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::max({t.data[i], val});
                }
                return a;
            }

            static Tensor max(double val, Tensor& t){
                return Tensor::max(t,val);
            }

            static Tensor max(Tensor& t, Tensor& s){
                // std::cout <<"MAX" <<std::endl;
                if (t.getShape() != s.getShape()){
                    throw std::invalid_argument("Shapes dont match for max operation");
                }
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::max({t.data[i], s.data[i]});
                }
                return a;
            }

            static Tensor min(Tensor& t, double val){
                // std::cout <<"MIN" <<std::endl;
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::min({t.data[i], val});
                }
                return a;
            }

            static Tensor min(double val, Tensor& t){
                return Tensor::min(t,val);
            }

            static Tensor min(Tensor& t, Tensor& s){
                // std::cout <<"MIN" <<std::endl;
                if (t.getShape() != s.getShape()){
                    throw std::invalid_argument("Shapes dont match for min operation");
                }
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::min({t.data[i], s.data[i]});
                }
                return a;
            }

            //----------------------------------------Absolute value------------------------------------------------------
            static Tensor abs(Tensor &t ){
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::abs(t.data[i]);
                }
                return a;
            }

            //----------------------------------------Mean ------------------------------------------------------

            // To be changed
            static Tensor mean(Tensor &t ){
                Tensor returnMean({1});
                size_t sizeTensor = t.sizeOfTensor();
                for (size_t i =0; i<sizeTensor; i++){
                    returnMean.data[0] += t.data[i];
                }
                returnMean.data[0]/= static_cast<double>(sizeTensor);
                return returnMean;
            }


            // TODO: refactor
            // Tensor(bool owns_data) : data(nullptr), owns_data(owns_data) {};
            void fill(double v){
                for (size_t i =0;i<sizeOfTensor(); i++){
                    this->data[i] = v;
                }
            }

            // TODO: refactor
            static bool has_nonzero_gradient(Tensor& t){
                // we only have to check if there is at least one of the numbers that is non-zero
                for (size_t i =0;i<t.sizeOfTensor(); i++){
                    if (std::abs(t.data[i]) > 1e-12) {
                        return true;
                    }
                }
                return false;
            }

            // TODO: refactor
            Tensor flatten(int start_dim =0, int end_dim = -1, bool keepdims=false);

            // TODO: refactor
            // flatten - inplace
            void flatten_inplace(int start_dim =0, int end_dim = -1, bool keepdims=false){
                this->shape = Tensor::flatten_(start_dim, end_dim, keepdims);
                computeStrides();
            }

            // TODO: refactor
            // in place summation across a dimension - works like torch.sum()
            Tensor sum(int dim, bool keepdims = false){
                if (dim<0 || dim>=shape.size()){
                    throw std::invalid_argument("DIM not in the correct range!");
                }

                // edge cases to consider - when we only have a vector then sum will give a scalar
                if (shape.size()==1 && dim ==0){
                    // no need to check keep dims as if keepdims is false then it will be a scalar anyways
                    Tensor new_t({1});
                    for (size_t i =0; i < sizeOfTensor(); i++){
                        new_t.data[0]+=data[i];
                    }
                    return new_t;
                }

                std::vector<int> newShape = shape;
                int oldDim = newShape[dim];
                newShape[dim] = 1; // we will change the shape afterwards

                ll offset_new_shape{1};
                for (int d = dim+1; d <newShape.size(); d++){
                    offset_new_shape*=newShape[d];
                }

                ll offset_old{offset_new_shape*oldDim};

                Tensor new_t(newShape);
                double* flat_data = new_t.data;

                ll dest_idx = 0;
                for (size_t v =0; v<sizeOfTensor(); v+=offset_old){
                    for (size_t s = 0; s<offset_new_shape;s++){
                        double val {};
                        for (int idx = 0; idx<oldDim; idx++){
                            val+= data[v+idx*offset_new_shape+s];
                        }
                        flat_data[dest_idx]= val;
                        dest_idx++;
                    }
                }
                if (keepdims) return new_t;
                new_t.flatten_inplace((dim<shape.size()-1)? dim : (dim-1),  (dim<shape.size()-1) ? dim+1 : -1, keepdims); // PROBLEM FOUND HERE in case when the dim passed in dim= shape.size()-1
                return new_t;
            }


            // TODO: refactor
            // linspace function  to edit the current tensor
            Tensor& linspace(double start, double end){
                size_t long_size = this->sizeOfTensor();
                double size = static_cast<double>(long_size);
                double step = (end-start)/(size);
                for (size_t i =0; i < long_size ; i++){
                    this->data[i] = start;
                    start+=step;
                }
                return *this;
            }




            // TODO: refactor
            Tensor transpose(){
                // 3 cases:
                // Case 1: 1 as the shape return the same thing - scalar - If transposing the same dimension or a single element tensor, return a copy
                if (sizeOfTensor() <= 1 ) {
                    return *this; // copy op
                }

                // Case 2: Vectors
                // TODO: ROW Tranpose or COLUMN Transpose


                // Case 3: Transpose the last 2 dims
                std::vector<int> new_shape = this->shape;
                std::reverse(new_shape.begin()+(new_shape.size()-2), new_shape.end()); // reverse the shape
                Tensor n (new_shape); // new matrix with reversed shape (transposed)
                ll offset = new_shape[new_shape.size()-1]*new_shape[new_shape.size()-2];
                for (size_t s = 0; s<n.sizeOfTensor(); s+=offset){
                    for (int r = 0 ; r<new_shape[this->shape.size()-2]; r++){
                        for (int c = 0; c < new_shape[this->shape.size()-1]; c++){
                            n.data[s+r*new_shape[new_shape.size()-1]+c] = this->data[s+c*this->shape[this->shape.size()-1]+r]; // transpose last two dims
                        }
                    }
                }
                return n; // transposed
            }


            // Permute - Not the same as TRANSPOSE
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
                for (size_t i = 0; i < this->sizeOfTensor(); i++){
                        result.data[i] = this->data[i];
                }
                return result;
            }


            // reshape
            void reshape(std::vector<int> new_shape){
                // check multipliability
                if (utils::isSizeValid(new_shape, this->sizeOfTensor()) == false){
                    throw std::invalid_argument("Invalid shape - needs to be multipliable to original shape");
                }
                this->shape = new_shape;
                computeStrides();
            }


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

                size_t total = (tensors.size()+1) * this->sizeOfTensor();
                double * temp = new double[total];
                size_t stride = this->sizeOfTensor();

                this->shape.insert(this->shape.begin(), tensors.size()+1);
                computeStrides();


                std::copy(this->data, this->data + stride, temp + 0);
                ll start = stride;
                for (const Tensor& tensor : tensors){
                    double * tempData = tensor.data;

                    std::copy(tempData, tempData + stride, temp + start); // copy the data
                    start += stride;
                }


                if (this->data != nullptr) {
                    delete[] this->data;
                }
                this->data = temp;
            }

            // TODO
            // static Tensor identity_matrix(std::vector<int>& sizePassed, int n) {
            //     Tensor t (sizePassed);
            //     // fill the diagonals with one
            //     if (sizePassed.size() ==1){
            //         // vector basically - does also work for scalars
            //         if (sizePassed[0] != n){
            //             std::invalid_argument("Identity vector is just 1s and size should match what is entered");
            //         }
            //         for (int r =0; r < sizePassed[0]; r++){
            //             t.set(1.0, {r});
            //         }
            //     } else if (sizePassed.size() ==2){
            //         if (sizePassed[sizePassed.size()-1] != n || sizePassed[sizePassed.size()-2]!=n){
            //             std::invalid_argument("Identity matrices are only defined for square matrices");
            //         }
            //         for (int r =0; r < sizePassed[0]; r++){
            //             t.set(1.0, {r,r}); // diagonals only
            //         }
            //     } else{
            //         // batched matrix - set individually to identity matrix
            //         //TODO:
            //     }

            //     return t;
            // }

            // // acts as an alias for the identity matrix
            // static Tensor eye(std::vector<int>& sizePassed, int n){
            //     return identity_matrix(sizePassed, n);
            // }


        };
}
#endif
