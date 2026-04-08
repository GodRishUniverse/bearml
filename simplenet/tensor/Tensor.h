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

#include "reductions/reduction_ops.h"

// #include <cmath>
#include <iomanip>


// cuda imports
#include "cuda/includes/cuda_helper.h"
#include "cuda/includes/kernel_links.h"

using ll = long long; // can also use int_fast64_t
using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// TODO :implementation needed - division (inversion - should work for constants and matrix inversion ), unflatten
// TODO: element-wise divide
// TODO: allow float values as well (32-bit precision) - template specialize equal to
// TODO: fix inplace operators for CUDA

#ifndef TENSOR_H
#define TENSOR_H
namespace simplenet{

    // forward declaration
    namespace linear_algebra {
       Tensor batchedMatMul(const Tensor& a, const Tensor& b); // Forward declare the friend function
       Tensor reduce(const Tensor& a, std::vector<int>& afterShape, reductions::ReductionOps op); // forward declare for the friend reduce
       Tensor hadamard(const Tensor &a, const Tensor &other);

       Tensor compare(const Tensor& a, const Tensor& b,CompareOp op, double true_val, double false_val);

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

       Tensor inverse(const Tensor& a);
    }

    namespace neural_network {

        Tensor padding(const simplenet::Tensor& input, int pad_amount, Padding_Op_Code padding_mode);
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

            static bool isScalar(const Tensor& t) {
                if (t.shape.empty()) return true;

                // for (int dim : t.shape) {
                //     if (dim != 1) return false;
                // }
                // return true;

                // the below code is the same as above but now uses stl
                return std::all_of(t.shape.begin(), t.shape.end(), [](int dim) { return dim == 1; });
            }

            static double getScalarValue(const Tensor& t){
                if (!isScalar(t)) {
                    throw std::runtime_error("Cannot get scalar value from non-scalar tensor");
                }

                double result;
                if (t.device.is_cpu()) {
                    result = t.data[0];
                } else {
                    // we copy to the host (cpu) to get the scalar value
                    t.allocator_->copy_to_host(&result, t.data, sizeof(double));
                }
                return result;

                // return t.data[0];
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

            // utility for permute
            void setShape(std::vector<int> &newShape){
                this->shape = newShape;
            }

            // ==============================PRIVATE========================================

        public:

            // constructor when size and data are provided -> copies on same device as I want to ensure the programmer has explicit knowledge of where the tensor is and should use .to before doing "cross-devices" copies
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
                    // check if assigning to wrong CUDA device
                    if (device.device_id >= cuda_device_count) {
                           throw std::invalid_argument("Invalid CUDA device ID: " +
                                                      std::to_string(device.device_id));
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
                    CUDA_CHECK(cudaMemset(this->data, 0, bytes));
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

            // Reduced the repetitive code - CPU side element-wise binary — handles both contiguous and broadcast paths
            template<typename Func>
            static Tensor elementwise_binary_cpu(const Tensor& A, const Tensor& B, Func fn) {
                if (A.shape == B.shape) {
                    Tensor C(A.shape, A.device);
                    for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                        C.data[i] = fn(A.data[i], B.data[i]);
                    return C;
                }
                // broadcast path
                auto outShape = utils::computeBroadcastShape(A.shape, B.shape);
                Tensor aView = makeBroadcastView(A, outShape);
                Tensor bView = makeBroadcastView(B, outShape);
                Tensor C(outShape, A.device);
                size_t N = C.sizeOfTensor();
                for (ll idx = 0; idx < static_cast<ll>(N); ++idx) {
                    ll tmp = idx, offA = 0, offB = 0;
                    for (int d = static_cast<int>(outShape.size()) - 1; d >= 0; --d) {
                        int coord = tmp % outShape[d];
                        tmp /= outShape[d];
                        offA += coord * aView.strides[d];
                        offB += coord * bView.strides[d];
                    }
                    C.data[idx] = fn(A.data[offA], B.data[offB]);
                }
                return C;
            }

            // Tensor-Tensor operator: uses device types to choose the path - uses OpCode for choosing the operator
            static Tensor elementwise_binary(const Tensor& A, const Tensor& B, OP_Code op) {
                utils::errorCheckSameDevice(A, B);
                if (A.device.type == DeviceType::CUDA) {
                    if (A.shape == B.shape) {
                        Tensor C(A.shape, A.device);
                        cuda::launch_elementwise_contiguous<double>(A.data, B.data, C.data, C.getShape(), op);
                        return C;
                    }
                    auto outShape = utils::computeBroadcastShape(A.shape, B.shape);

                    Tensor aView = makeBroadcastView(A, outShape);
                    Tensor bView = makeBroadcastView(B, outShape);

                    // aView.to_(A.device);
                    // bView.to_(A.device);

                    Tensor C(outShape, A.device);

                    cuda::launch_elementwise_broadcast<double>(aView.data, bView.data, C.data,
                        aView.getStrides(), bView.getStrides(), C.getShape(), op);
                    return C;
                }
                // CPU path with appropriate lambda
                switch(op) {
                    case OP_Code::OP_ADD: return elementwise_binary_cpu(A, B, std::plus<double>{});
                    case OP_Code::OP_SUB: return elementwise_binary_cpu(A, B, std::minus<double>{});
                    case OP_Code::OP_MUL: return elementwise_binary_cpu(A, B, std::multiplies<double>{});
                    case OP_Code::OP_DIV: return elementwise_binary_cpu(A, B, std::divides<double>{});
                    case OP_Code::OP_MAX: return elementwise_binary_cpu(A, B, [](double a, double b){ return std::max(a,b); });
                    case OP_Code::OP_MIN: return elementwise_binary_cpu(A, B, [](double a, double b){ return std::min(a,b); });
                    default:
                        throw std::invalid_argument("OP Code does not exist - in elementwise_binary.");
                }
            }

            // Tensor-scalar operator: handles Tensor op scalar and scalar op Tensor
            static Tensor elementwise_scalar(const Tensor& A, double b, OP_Code op, LHS_RHS_Code side) {
                Tensor C(A.shape, A.device);
                if (A.device.type == DeviceType::CUDA) {
                    cuda::launch_elementwise_contiguous_with_constant<double>(A.data, b, C.data, A.shape, op, side);
                    return C;
                }
                size_t N = A.sizeOfTensor();
                if (side == LHS_RHS_Code::OP_RHS) {
                    // A op b
                    switch(op) {
                        case OP_Code::OP_ADD:
                            for (size_t i=0;i<N;++i) C.data[i]=A.data[i]+b;
                            break;
                        case OP_Code::OP_SUB:
                            for (size_t i=0;i<N;++i) C.data[i]=A.data[i]-b;
                            break;
                        case OP_Code::OP_MUL:
                            for (size_t i=0;i<N;++i) C.data[i]=A.data[i]*b;
                            break;
                        case OP_Code::OP_DIV:
                            for (size_t i=0;i<N;++i) C.data[i]=A.data[i]/b;
                            break;
                        default:
                            throw std::invalid_argument("OP Code not supported for scalar op.");
                    }
                } else {
                    // b op A
                    switch(op) {
                        case OP_Code::OP_ADD:
                            for (size_t i=0;i<N;++i) C.data[i]=b+A.data[i];
                            break;
                        case OP_Code::OP_SUB:
                            for (size_t i=0;i<N;++i) C.data[i]=b-A.data[i];
                            break;
                        case OP_Code::OP_MUL:
                            for (size_t i=0;i<N;++i) C.data[i]=b*A.data[i];
                            break;
                        case OP_Code::OP_DIV:
                            for (size_t i=0;i<N;++i) C.data[i]=b/A.data[i];
                            break;
                        default:
                            throw std::invalid_argument("OP Code not supported for scalar op.");
                    }
                }
                return C;
            }

            // in-place Tensor-Tensor operator -> used only for + and -
            Tensor& inplace_tensor_binary(const Tensor& other, OP_Code op) {
                utils::errorCheckSameDevice(*this, other);
                if (shape != other.shape) {
                    auto outShape = utils::computeBroadcastShape(shape, other.shape);
                    // we require that *this already has exactly outShape:
                    // otherwise you'd need to reallocate or error.
                    if (shape != outShape)
                        throw std::invalid_argument("LHS must match broadcasted shape for in-place op");

                    // CUDA
                    if (device.type == DeviceType::CUDA) {

                        Tensor oView = makeBroadcastView(other, outShape);

                        cuda::launch_elementwise_broadcast<double>(data, other.data, data,
                            getStrides(), oView.getStrides(), outShape, op);

                        return *this;
                    }

                    // CPU broadcast in-place
                    Tensor oView = makeBroadcastView(other, outShape);
                    size_t N = sizeOfTensor();
                    for (size_t idx = 0; idx < N; ++idx) {
                        ll tmp = idx, offO = 0;
                        for (int d = static_cast<int>(outShape.size()) - 1; d >= 0; --d) {
                            int coord = tmp % outShape[d];
                            tmp /= outShape[d];
                            offO += coord * oView.strides[d];
                        }
                        switch(op) {
                            case OP_Code::OP_ADD:
                                data[idx] += oView.data[offO];
                                break;
                            case OP_Code::OP_SUB:
                                data[idx] -= oView.data[offO];
                                break;
                            default:
                                throw std::invalid_argument("Unsupported in-place broadcast op");
                        }
                    }
                } else {
                    // CUDA
                    if (device.type == DeviceType::CUDA) {
                        cuda::launch_elementwise_contiguous<double>(data, other.data, data, shape, op);
                        return *this;
                    }
                    size_t N = sizeOfTensor();
                    switch(op) {
                        case OP_Code::OP_ADD:
                            for (size_t i=0;i<N;++i) data[i]+=other.data[i];
                            break;
                        case OP_Code::OP_SUB:
                            for (size_t i=0;i<N;++i) data[i]-=other.data[i];
                            break;
                        default:
                            throw std::invalid_argument("Unsupported in-place contiguous op");
                    }
                }
                return *this;
            }

            // in-place scalar-Tensor operator
            Tensor& inplace_scalar(double b, OP_Code op) {
                if (device.type == DeviceType::CUDA) {
                    cuda::launch_elementwise_contiguous_with_constant<double>(this->data, b, this->data, shape, op, LHS_RHS_Code::OP_RHS);
                    return *this;
                }
                size_t N = sizeOfTensor();
                switch(op) {
                    case OP_Code::OP_ADD:
                        for (size_t i=0;i<N;++i) data[i]+=b;
                        break;
                    case OP_Code::OP_SUB:
                        for (size_t i=0;i<N;++i) data[i]-=b;
                        break;
                    default:
                        throw std::invalid_argument("Unsupported in-place scalar op");
                }
                return *this;
            }



            // Unary Operations

            template<typename Func>
            static Tensor elementwise_unary_cpu(const Tensor& A, Func fn) {
                Tensor C(A.shape, A.device);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.data[i] = fn(A.data[i]);
                return C;
            }

            static Tensor elementwise_unary(const Tensor& A, OP_Code op) {
                if (A.device.type == DeviceType::CUDA) {
                    Tensor C(A.shape, A.device);
                    cuda::launch_elementwise_unary<double>(
                        A.data,
                        C.data,
                        C.getShape(),
                        op
                    );
                    return C;
                }
                // CPU path with appropriate lambda
                switch(op) {
                    case OP_Code::OP_EXP: return elementwise_unary_cpu(A, [](double a){ return std::exp(a); });
                    case OP_Code::OP_LOG: return elementwise_unary_cpu(A, [](double a){ return std::log(a); });
                    case OP_Code::OP_SQRT: return elementwise_unary_cpu(A, [](double a){ return std::sqrt(a); });
                    case OP_Code::OP_ABS: return elementwise_unary_cpu(A, [](double a){ return std::abs(a); });
                    case OP_Code::OP_SIN: return elementwise_unary_cpu(A, [](double a){ return std::sin(a); });
                    case OP_Code::OP_COS: return elementwise_unary_cpu(A, [](double a){ return std::cos(a); });
                    case OP_Code::OP_TAN: return elementwise_unary_cpu(A, [](double a){ return std::tan(a); });
                    case OP_Code::OP_SINH: return elementwise_unary_cpu(A, [](double a){ return std::sinh(a); });
                    case OP_Code::OP_COSH: return elementwise_unary_cpu(A, [](double a){ return std::cosh(a); });
                    case OP_Code::OP_TANH: return elementwise_unary_cpu(A, [](double a){ return std::tanh(a); });
                    default:
                        throw std::invalid_argument("OP Code does not exist - in elementwise_unary.");
                }
            }

            // ================================ OPERATORS - Using the above new element_wise functions =========================================

            // --------------------------------ADDITION----------------------------------------------------------------------------

            friend Tensor operator+(const Tensor &A, const Tensor &B) {
                return elementwise_binary(A, B, OP_Code::OP_ADD);
            }

            Tensor& operator+=(const Tensor &other) {
                return inplace_tensor_binary(other, OP_Code::OP_ADD);
            }

            // element wise add
            Tensor& operator+=(const double& b) {
                return inplace_scalar(b, OP_Code::OP_ADD);
            }

            // element wise add
            friend Tensor operator+(const Tensor &A, const double& b) {
                return elementwise_scalar(A, b, OP_Code::OP_ADD, LHS_RHS_Code::OP_RHS);
            }

            friend Tensor operator+(const double& b, const Tensor &A) {
                return A+b;
            } // same as above


            // --------------------------------SUBTRACTION-------------------------------------------------------------------------
            friend Tensor operator-(const Tensor &A, const Tensor &B) {
                return elementwise_binary(A, B, OP_Code::OP_SUB);
            }
            Tensor& operator-=(const Tensor &other) {
                return inplace_tensor_binary(other, OP_Code::OP_SUB);
            }
            // element wise subtract
            Tensor& operator-=(const double& b) {
                return inplace_scalar(b, OP_Code::OP_SUB);
            }
            // element wise subtract
            friend Tensor operator-(const Tensor &A, const double& b) {
                return elementwise_scalar(A, b, OP_Code::OP_SUB, LHS_RHS_Code::OP_RHS);
            }
            // element wise subtract - operation is switched (b - A)
            friend Tensor operator-(const double& b, const Tensor &A) {
                return elementwise_scalar(A, b, OP_Code::OP_SUB, LHS_RHS_Code::OP_LHS);
            }

            // --------------------------------MULTIPLICATION----------------------------------------------------------------------

            // Hadamard product
            friend Tensor linear_algebra::hadamard(const Tensor &a, const Tensor &other);

            // THIS is where we will be doing the multiplication when the dimensions exceed the normal 2 of a matrix
            friend Tensor linear_algebra::batchedMatMul(const Tensor& a, const Tensor& b);

            // TODO: write CUDA kernel
            friend Tensor linear_algebra::reduce(const Tensor& a, std::vector<int>& afterShape, reductions::ReductionOps op);

            // element wise multiply (now uses elementwise_scalar dispatch)
            friend Tensor operator*(const Tensor &A, const double& b) {
                return elementwise_scalar(A, b, OP_Code::OP_MUL, LHS_RHS_Code::OP_RHS);
            }

            // for when the operations are reversed
            friend Tensor operator*(const double& b, const Tensor &A) {
                return A*b; // order of multiplication doesn't matter here -> does it?
            }


            friend Tensor operator*(const Tensor &a, const Tensor &b) {
                utils::errorCheckSameDevice(a, b); // will throw an error if devices don't match

                // 5 cases that need to be checked for this
                //                  case 0: A and B are scalars in the form of Tensors [TAKEN INSIDE CASE 1 and 2]
                // case 1: A is scalar
                // case 2: B is scalar
                // case 3: A is a vector and B is a vector
                // case 4: A is matrix and B is a vector
                // case 5: A is a vector and B is a matrix
                // case 6: matrix multiplication - batched and unbatched (GEMM operations)

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
                        throw std::invalid_argument("Vector dimensions must match for dot product");
                    }

                    if (a.device == DeviceType::CUDA) {
                        // Treat as 1x1 matmul: (1,N) * (N,1) = (1,1), extract scalar
                        Tensor result({1}, a.device);
                        cuda::launch_gemm_contiguous<double>(
                            a.data, b.data, result.data,
                            1,    // batchsize
                            1,    // m (rows of a treated as row vector)
                            a_shape[0], // k (common dim),
                            1,    // n (cols of result)
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const Eigen::VectorXd> vec_a(a.data, a_shape[0]);
                    Eigen::Map<const Eigen::VectorXd> vec_b(b.data, b_shape[0]);

                    Tensor output({1});
                    output.data[0] = vec_a.dot(vec_b);
                    return output;
                }

                // CASE 4: matrix n by m multiplied with m by 1 vector
                if (a_shape.size() == 2 && b_shape.size() == 1) {
                    if (a_shape[1] != b_shape[0]) {
                        throw std::invalid_argument("Matrix columns must match vector size");
                    }

                    if (a.device == DeviceType::CUDA) {
                        // Treat vector as (m,1) matrix: (n,m) * (m,1) = (n,1)
                        Tensor result({a_shape[0]}, a.device);
                        cuda::launch_gemm_contiguous<double>(
                            a.data, b.data, result.data,
                            1,           // batchsize
                            a_shape[0],  // m
                            a_shape[1],  // k (common dim),
                            1,           // n (output cols)
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const MatrixRowMajor> mat_a(a.data, a_shape[0], a_shape[1]);
                    Eigen::Map<const Eigen::VectorXd> vec_b(b.data, b_shape[0]);

                    Tensor result({a_shape[0]});
                    Eigen::Map<Eigen::VectorXd> result_vec(result.data, a_shape[0]);
                    result_vec = mat_a * vec_b;

                    return result;
                }

                // CASE 5: vector m multiplied with m by n matrix
                if (a_shape.size() == 1 && b_shape.size() == 2) {
                    if (a_shape[0] != b_shape[0]) {
                        throw std::invalid_argument("Matrix columns must match vector size: : vector m multiplied with m by n matrix");
                    }

                    if (a.device == DeviceType::CUDA) {
                        // Treat vector as (1,m) matrix: (1,m) * (m,n) = (1,n)
                        Tensor result({1,b_shape[1]}, a.device);
                        cuda::launch_gemm_contiguous<double>(
                            a.data, b.data, result.data,
                            1,           // batchsize
                            1,           // m
                            b_shape[0],  // k (common dim),
                            b_shape[1],  // n (output cols)
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const Eigen::VectorXd> vec_a(a.data, a_shape[0]);
                    Eigen::Map<const MatrixRowMajor> mat_b(b.data, b_shape[0], b_shape[1]);


                    Tensor result({b_shape[1]});
                    Eigen::Map<Eigen::VectorXd> result_vec(result.data, b_shape[1]);
                    result_vec = mat_b * vec_a;

                    return result;
                }

                // CASE 6: unbatched matmul
                if (a_shape.size() == 2 && b_shape.size() == 2) {
                    if (a_shape[1] != b_shape[0]) {
                        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
                    }

                    if (a.device == DeviceType::CUDA) {
                        Tensor result({a_shape[0], b_shape[1]}, a.device);
                        cuda::launch_gemm_contiguous<double>(
                            a.data, b.data, result.data,
                            1,           // batchsize
                            a_shape[0],  // m
                            a_shape[1],  // k
                            b_shape[1],  // n
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const MatrixRowMajor> mat_a(a.data, a_shape[0], a_shape[1]);
                    Eigen::Map<const MatrixRowMajor> mat_b(b.data, b_shape[0], b_shape[1]);

                    Tensor result({a_shape[0], b_shape[1]});
                    Eigen::Map<MatrixRowMajor> result_mat(result.data, a_shape[0], b_shape[1]);
                    result_mat = (mat_a * mat_b);

                    return result;
                }

                // CASE 6: batched matmul (delegates to batchedMatMul which handles CUDA)
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
            friend Tensor operator/(const Tensor &A, const double& b) {
                return elementwise_scalar(A, b, OP_Code::OP_DIV, LHS_RHS_Code::OP_RHS);
            }

            friend Tensor operator/(const double& b, const Tensor &A) {
                return elementwise_scalar(A, b, OP_Code::OP_DIV, LHS_RHS_Code::OP_LHS);
            }

            friend Tensor operator/(const Tensor &a, const Tensor &b) {
                if (b.shape != a.shape){
                    throw std::runtime_error("Shapes should match for element-wise divide");
                }
                return elementwise_binary(a, b, OP_Code::OP_DIV);
            }

            // Another case exists but that is when matrix b is invertible and then it just becomes matrix mul
            friend Tensor linear_algebra::inverse(const Tensor& a); // TODO: add CUDA dispatch function to compute inverse


            // --------------------------------EQUALITY--------------------------------------------------------------------------

            // TODO: write a CUDA check kernel
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

            // TODO: write a CUDA check kernel
            friend bool operator!=(const Tensor &a, const Tensor &b){
                return !(a==b);
            }


            // -----------------------------------------------Operations helpful in mask generations------------------


            friend Tensor linear_algebra::compare(const Tensor& a, const Tensor& b,CompareOp op, double true_val, double false_val);

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


            // padding (in convolution_layers.h file)
            friend Tensor neural_network::padding(const simplenet::Tensor& input, int pad_amount, Padding_Op_Code padding_mode);

            // TODO: refactor for native CUDA support
            //----------------------------------------ACCUMULATORs (used in reduce)------------------------------------------------------
            Tensor accumulate(int dim, reductions::ReductionOps op, bool keepdims = false){
                if (dim<0 || dim>=shape.size()){
                    throw std::invalid_argument("DIM not in the correct range!");
                }

                // edge cases to consider - when we only have a vector then sum will give a scalar
                if (shape.size()==1 && dim ==0){
                    // no need to check keep dims as if keepdims is false then it will be a scalar anyways
                    Tensor new_t({1});
                    new_t.data[0] = (op == reductions::ReductionOps::PROD) ? 1.0
                                             : (op == reductions::ReductionOps::MAX)  ? -std::numeric_limits<double>::infinity()
                                             : (op == reductions::ReductionOps::MIN)  ?  std::numeric_limits<double>::infinity()
                                             : 0.0; // even for argmin/argmax initial value is 0
                    double value = new_t.data[0]; // for argmin/argmax
                    for (size_t i =0; i < sizeOfTensor(); i++){
                        switch (op) {
                            case reductions::ReductionOps::SUM: case reductions::ReductionOps::MEAN:
                                new_t.data[0] += data[i];
                                break;
                            case reductions::ReductionOps::PROD:
                                new_t.data[0] *= data[i];
                                break;
                            case reductions::ReductionOps::MAX:
                                new_t.data[0] = std::max(new_t.data[0] , data[i]);
                                break;
                            case reductions::ReductionOps::MIN:
                                new_t.data[0] = std::min(new_t.data[0] , data[i]);
                                break;
                            case reductions::ReductionOps::ARG_MAX:
                                if (data[i] > value) {
                                    new_t.data[0] = i;
                                    value = data[i];
                                }
                                break;
                            case reductions::ReductionOps::ARG_MIN:
                                if (data[i] < value) {
                                    new_t.data[0] = i;
                                    value = data[i];
                                }
                                break;
                            default:
                                throw std::runtime_error("Unsupported reduction op");
                        }
                    }
                    // for mean, divide by the number of elements (get scalar value)
                    if (op == reductions::ReductionOps::MEAN) {
                        new_t.data[0] /= sizeOfTensor();
                    }
                    new_t.to_(this->device);
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

                // gpu direct access not allowed, so we copy to CPU first
                Tensor copy_tensor = *this;
                copy_tensor.to_(Device(DeviceType::CPU, -1));

                ll dest_idx = 0;
                for (size_t v =0; v<sizeOfTensor(); v+=offset_old){
                    for (size_t s = 0; s<offset_new_shape;s++){
                        // accumulate initial value based on reduction op
                        double val = (op == reductions::ReductionOps::PROD) ? 1.0
                                                 : (op == reductions::ReductionOps::MAX || op == reductions::ReductionOps::ARG_MAX)  ? -std::numeric_limits<double>::infinity()
                                                 : (op == reductions::ReductionOps::MIN || op == reductions::ReductionOps::ARG_MIN)  ?  std::numeric_limits<double>::infinity()
                                                 : 0.0;
                        double arg_idx = 0.0;

                        for (int idx = 0; idx<oldDim; idx++){
                            // edited from this->data to copy_tensor.data
                            double elem = copy_tensor.data[v + idx * offset_new_shape + s];
                            switch (op) {
                                case reductions::ReductionOps::SUM: case reductions::ReductionOps::MEAN:
                                    val += elem;
                                    break;
                                case reductions::ReductionOps::PROD:
                                    val *= elem;
                                    break;
                                case reductions::ReductionOps::MAX:
                                    val = std::max(val, elem);
                                    break;
                                case reductions::ReductionOps::MIN:
                                    val = std::min(val, elem);
                                    break;
                                case reductions::ReductionOps::ARG_MAX:
                                    if (elem > val) {
                                        val = elem;
                                        arg_idx = idx;
                                    }
                                    break;
                                case reductions::ReductionOps::ARG_MIN:
                                    if (elem < val) {
                                        val = elem;
                                        arg_idx = idx;
                                    }
                                    break;
                                default:
                                    throw std::runtime_error("Unsupported reduction op");

                            }
                        }
                        if (op == reductions::ReductionOps::ARG_MAX || op == reductions::ReductionOps::ARG_MIN) {
                            flat_data[dest_idx] = arg_idx;  // store the index, not the value
                        } else {
                            if (op == reductions::ReductionOps::MEAN) val /= oldDim; // only for mean
                            flat_data[dest_idx]= val;
                        }
                        dest_idx++;
                    }
                }
                if (keepdims) {
                    new_t.to_(this->device);
                    return new_t;
                }
                new_t.flatten_inplace((dim<shape.size()-1)? dim : (dim-1),  (dim<shape.size()-1) ? dim+1 : -1, keepdims); // PROBLEM FOUND HERE in case when the dim passed in dim= shape.size()-1
                new_t.to_(this->device);
                return new_t;
            }

            // Tensor argmax(int dim, bool keepdims = false);
            // Tensor argmin(int dim, bool keepdims = false);


            //----------------------------------------Exponential------------------------------------------------------
            static Tensor exp(Tensor& t){
                // std::cout <<"EXPONENTIATED" <<std::endl;
                return elementwise_unary(t, OP_Code::OP_EXP);
            }

            //----------------------------------------Sin------------------------------------------------------
            static Tensor sin(Tensor& t){
                // std::cout <<"SIN" <<std::endl;
                return elementwise_unary(t, OP_Code::OP_SIN);
            }

            //----------------------------------------Cos------------------------------------------------------
            static Tensor cos(Tensor& t){
                // std::cout <<"COS" <<std::endl;
                return elementwise_unary(t, OP_Code::OP_COS);
            }

            //----------------------------------------Tan------------------------------------------------------
            static Tensor tan(Tensor& t){
                // std::cout <<"TAN" <<std::endl;
                return elementwise_unary(t, OP_Code::OP_TAN);
            }

            //----------------------------------------Hyperbolic------------------------------------------------------
            static Tensor sinh(Tensor& t){
                // std::cout <<"SINH" <<std::endl;
                return elementwise_unary(t, OP_Code::OP_SINH);
            }

            static Tensor cosh(Tensor& t){
                // std::cout <<"COSH" <<std::endl;
                return elementwise_unary(t, OP_Code::OP_COSH);
            }

            static Tensor tanh(Tensor& t){
                // std::cout <<"TANH" <<std::endl;
                return elementwise_unary(t, OP_Code::OP_TANH);
            }



            //----------------------------------------Max------------------------------------------------------

            // TODO: fix this - CUDA kernel as well
            static Tensor max(const Tensor& t,const  double val){
                // std::cout <<"MAX" <<std::endl;
                if (t.device == DeviceType::CUDA) {
                    // Create a scalar tensor on CUDA filled with val
                    Tensor scalar_t(t.getShape());
                    for (size_t i = 0; i < t.sizeOfTensor(); i++) scalar_t.data[i] = val;
                    scalar_t.to_(t.device);
                    return Tensor::max(t, scalar_t);
                }
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::max({t.data[i], val});
                }
                return a;
            }

            static Tensor max(const double val, const Tensor& t){
                return Tensor::max(t,val);
            }

            static Tensor max(const Tensor& t, const Tensor& s){
                // std::cout <<"MAX" <<std::endl;
                if (t.getShape() != s.getShape()){
                    throw std::invalid_argument("Shapes dont match for max operation");
                }
                if (t.device == DeviceType::CUDA) {
                    Tensor result(t.getShape(), t.device);
                    cuda::launch_elementwise_contiguous<double>(
                        t.data, s.data, result.data,
                        t.getShape(), OP_Code::OP_MAX
                    );
                    return result;
                }
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::max({t.data[i], s.data[i]});
                }
                return a;
            }

            //----------------------------------------Min------------------------------------------------------

            static Tensor min(const Tensor& t, double val){
                // std::cout <<"MIN" <<std::endl;
                if (t.device == DeviceType::CUDA) {
                    Tensor scalar_t(t.getShape());
                    for (size_t i = 0; i < t.sizeOfTensor(); i++) scalar_t.data[i] = val;
                    scalar_t.to_(t.device);
                    return Tensor::min(t, scalar_t);
                }
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::min({t.data[i], val});
                }
                return a;
            }

            static Tensor min(const double val, const Tensor& t){
                return Tensor::min(t,val);
            }

            static Tensor min(const Tensor& t, const Tensor& s){
                // std::cout <<"MIN" <<std::endl;
                if (t.getShape() != s.getShape()){
                    throw std::invalid_argument("Shapes dont match for min operation");
                }
                if (t.device == DeviceType::CUDA) {
                    Tensor result(t.getShape(), t.device);
                    cuda::launch_elementwise_contiguous<double>(
                        t.data, s.data, result.data,
                        t.getShape(), OP_Code::OP_MIN
                    );
                    return result;
                }
                Tensor  a = t; // copied
                for (size_t i =0; i<t.sizeOfTensor(); i++){
                    a.data[i] = std::min({t.data[i], s.data[i]});
                }
                return a;
            }

            //----------------------------------------Absolute value------------------------------------------------------
            // Note: const accepts both non-const and const tensors
            static Tensor abs(const Tensor &t ){
                return elementwise_unary(t, OP_Code::OP_ABS);
            }


            //----------------------------------------Square root------------------------------------------------------
            // Note: const accepts both non-const and const tensors

            static Tensor sqrt(const Tensor &t ){
                return elementwise_unary(t, OP_Code::OP_SQRT);
            }



            //----------------------------------------Mean ------------------------------------------------------
            static Tensor mean(Tensor &t ){
                Tensor result({1}, t.device);
                double sum = 0.0;
                if (t.device == DeviceType::CUDA) {
                    cuda::launch_sum_kernel(t.data, result.data, t.sizeOfTensor());
                    result.to_(Device::cpu()); // send to CPU to get the sum as GPU direct memory access is NOT SET UP
                    result.set(result.data[0] / static_cast<double>(t.sizeOfTensor()), {0}); // set the mean value
                    result.to_(t.device); // send back to GPU
                }else {
                    size_t sizeTensor = t.sizeOfTensor();
                    for (size_t i =0; i<sizeTensor; i++){
                       sum += t.data[i];

                    }
                    result.set(sum / static_cast<double>(t.sizeOfTensor()), {0}) ;
                }
                return result;
            }


            //---------------------------------------- Log ------------------------------------------------------

            static Tensor log(Tensor &t ){
                return elementwise_unary(t, OP_Code::OP_LOG);
            }


            // Tensor(bool owns_data) : data(nullptr), owns_data(owns_data) {};
            void fill(double v){
                // CUDA fill
                if (!this->device.is_cpu()) {
                    cuda::launch_fill<double>(
                        this->data,
                        v,
                        this->shape
                    );
                    return;
                }

                // CPU fill
                for (size_t i =0;i<sizeOfTensor(); i++){
                    this->data[i] = v;
                }
            }

            // TODO: refactor
            static bool has_nonzero_gradient(Tensor& t){

                if (t.device.is_cpu()) {
                    // we only have to check if there is at least one of the numbers that is non-zero
                    for (size_t i =0;i<t.sizeOfTensor(); i++){
                        if (std::abs(t.data[i]) > 1e-12) {
                            return true;
                        }
                    }
                    return false;
                }

                //CUDA support
                return cuda::launch_check_zero_kernel(t.data, t.sizeOfTensor());

            }

            Tensor flatten(int start_dim =0, int end_dim = -1, bool keepdims=false); // function declaration

            // flatten - inplace
            void flatten_inplace(int start_dim =0, int end_dim = -1, bool keepdims=false){
                this->shape = Tensor::flatten_(start_dim, end_dim, keepdims);
                computeStrides();
            }


            // TODO: refactor - CUDA
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




            Tensor transpose(){
                // 3 cases:
                // Case 1: 1 as the shape return the same thing - scalar - If transposing the same dimension or a single element tensor, return a copy
                if (sizeOfTensor() <= 1 ) {
                    return *this; // copy op
                }

                // Case 2: Vectors: ROW Tranpose or COLUMN Transpose
                if (this->shape.size() == 2 && (this->shape[0]==1 || this->shape[1]==1)) {
                    Tensor newTensor = *this; // copy
                    std::swap(newTensor.shape[0], newTensor.shape[1]);
                    newTensor.computeStrides();
                    return newTensor;
                }


                // Case 3: Transpose the last 2 dims
                std::vector<int> new_shape = this->shape;
                std::reverse(new_shape.begin()+(new_shape.size()-2), new_shape.end()); // reverse the shape
                Tensor n (new_shape, this->device); // new matrix with reversed shape (transposed)
                ll offset = new_shape[new_shape.size()-1]*new_shape[new_shape.size()-2];


                if (this->device.type == DeviceType::CUDA) {

                    ll batch_size = this->sizeOfTensor() / offset;
                    // batch size is the number of elements in each batch
                    cuda::launch_transpose_kernel(this->data, n.data, batch_size, this->shape[this->shape.size()-2], this->shape[this->shape.size()-1]);
                } else {
                    for (size_t s = 0; s<n.sizeOfTensor(); s+=offset){
                        for (int r = 0 ; r<new_shape[this->shape.size()-2]; r++){
                            for (int c = 0; c < new_shape[this->shape.size()-1]; c++){
                                n.data[s+r*new_shape[new_shape.size()-1]+c] = this->data[s+c*this->shape[this->shape.size()-1]+r]; // transpose last two dims
                            }
                        }
                    }
                }
                return n; // transposed
            }


            // Permute - Not the same as TRANSPOSE
            Tensor permute(std::vector<int> new_order){
                Tensor result= *this; // copy constructor
                result.inplace_permute(new_order);
                return result;
            }

            void inplace_permute(std::vector<int> new_order) {
                std::vector<int> temp ;
                for (int i = 0; i < new_order.size(); i++){
                        if (new_order[i] < 0 || new_order[i] >= this->shape.size()){
                            throw std::invalid_argument("Invalid permute order");
                        }
                }
                // permuting the shape
                for (int i = 0; i < new_order.size(); i++){
                    temp.push_back(this->shape[new_order[i]]);
                }
                this->shape = temp;
                this->computeStrides();
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

            // TODO: CUDA support
            static Tensor concat(std::initializer_list<Tensor> tensors, int dim =0 ){
                if (tensors.size() <= 1) {
                    throw std::invalid_argument("More than one tensor is required for concatenation.");
                }
                std::vector<int> concatShapePrev = tensors.begin()[0].getShape();
                int concatShapeSize = concatShapePrev.size();
                int concatDim = concatShapePrev[dim];
                Device device = tensors.begin()[0].device;


                concatShapePrev[dim]= 0 ; // for comparison
                for (size_t i = 1; i < tensors.size(); ++i) {
                    std::vector<int> copiedShape = tensors.begin()[i].getShape();
                    copiedShape[dim] = 0;
                    if (copiedShape != concatShapePrev){
                        throw std::invalid_argument("Tensors must have the same shape for concatenation except along the concatenation dimension");
                    }
                    if (copiedShape.size() != concatShapeSize) {
                        throw std::invalid_argument("Tensors must have the same shape for concatenation");
                    }

                    if (tensors.begin()[i].device != device) {
                        throw std::invalid_argument("Tensors must have the same device for concatenation");
                    }
                    concatDim += tensors.begin()[i].getShape()[dim];
                }

                std::vector<int> temp = tensors.begin()[0].getShape();
                temp[dim] = concatDim;
                Tensor result(temp, tensors.begin()[0].device);

                if (device.type == DeviceType::CUDA) {
                    result.to_(Device::cpu());
                }

                // outer_dim, dim, inner_dim is what we have
                // copy the dim*inner_dim elements for each outer_dim (like the chunks)
                ll outerDim = 1;
                for (int i = 0; i < dim; ++i) {
                    outerDim *= concatShapePrev[i];
                }

                ll innerDim = 1;
                for (int i = dim + 1; i < concatShapeSize; i++) {
                    innerDim *= concatShapePrev[i];
                }

                for (size_t o = 0; o < outerDim; ++o) {
                    int offset = 0; // offset will be used to move the pointer to execute the copies
                    for (size_t i = 0; i < tensors.size(); ++i) {
                        int src_cat_dim = tensors.begin()[i].getShape()[dim];
                        int copy_size = src_cat_dim * innerDim;
                        // we copy the data for each tensor into the result tensor
                        // outer*concatDim * innerDim moves the pointer to the correct position in the result tensor
                        // offset * innerDim is the offset that will be copied from the source tensor using (o*src_cat_dim*innerDim)
                        std::memcpy(result.data  + o*concatDim *innerDim + offset*innerDim, tensors.begin()[i].data + o*src_cat_dim*innerDim, copy_size * sizeof(double));
                        offset += copy_size;
                    }
                }

                if (device.type == DeviceType::CUDA) {
                    result.to_(device);
                }

                return result;
            }



            void stack(std::initializer_list<Tensor> tensors){
                if (tensors.size() == 0) {
                    throw std::invalid_argument("At least one tensor is required for stacking.");
                }
                // need to check that all tensors have the same shape - for stacking
                for (const Tensor& tensor : tensors){
                    if (tensor.shape != this->shape){
                        throw std::invalid_argument("Tensors must have the same shape for stacking");
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
            // static Tensor identity_matrix(std::vector<int>& sizePassed, int n)

        };
}
#endif
