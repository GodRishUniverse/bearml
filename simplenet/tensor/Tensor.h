#pragma once
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <limits>

#include <span>

#include "devices/device_type.h"
#include "devices/device_allocator.h"


#include "Eigen/Dense" // IMPORTING eigen for BLAS functions
#include "Eigen/src/Core/Matrix.h"

#include <boost/algorithm/string.hpp> // for string manipulation

#include "utils/shape_utils.h"
#include "utils/debug_utils.h"
#include "utils/linalg_utils.h"
#include "utils/device_utils.h"
#include "utils/slice.h"
#include "utils/ordering.h"

#include "reductions/reduction_ops.h"

// #include <cmath>
#include <iomanip>


// Operation-code enums (OP_Code, LHS_RHS_Code, Padding_Op_Code). These are
// plain C++ and used by CPU paths too, so they must NOT sit behind the CUDA
// guard. Previously they only reached here transitively via cuda_imports.h.
#include "operators/ops.h"
#include "operators/padding_ops.h"

#if defined(SIMPLENET_USE_CUDA)
    #include "cuda/includes/cuda_helper.h"
    #include "cuda/includes/kernel_links.h"
    #include "cuda/kernels/utils.cuh"
#else
    // CPU-only build: provide throwing stubs for the cuda::launch_* API so the
    // (runtime-dead) CUDA dispatch branches still compile.
    #include "cuda/includes/kernel_stubs.h"
#endif

using ll = long long; // can also use int_fast64_t

// templated Eigen helpers - element type follows the Tensor<T> element type
template<typename ET>
using MatrixRowMajorT = Eigen::Matrix<ET, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<typename ET>
using VectorXT = Eigen::Matrix<ET, Eigen::Dynamic, 1>;


// TODO :implementation needed - division (inversion - should work for constants and matrix inversion ), unflatten
// TODO: element-wise divide
// TODO: allow float values as well (32-bit precision) - template specialize equal to
// TODO: fix inplace operators for CUDA

#ifndef TENSOR_H
#define TENSOR_H
namespace simplenet{

    // forward declaration of the class template - needed by the free-function declarations below
    template <typename T> class Tensor;

    // forward declaration
    namespace linear_algebra {
       template <typename T> Tensor<T> batchedMatMul(const Tensor<T>& a, const Tensor<T>& b); // Forward declare the friend function
       template <typename T> Tensor<T> reduce(const Tensor<T>& a, std::vector<int>& afterShape, reductions::ReductionOps op); // forward declare for the friend reduce
       template <typename T> Tensor<T> hadamard(const Tensor<T> &a, const Tensor<T> &other);

       template <typename T> Tensor<T> compare(const Tensor<T>& a, const Tensor<T>& b,CompareOp op, T true_val, T false_val);

       // Tensor and Tensor
       template <typename T> Tensor<T> mask_of_greater_than_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_greater_than(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_less_than_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_less_than(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val);

       // Scalar and Tensor
       template <typename T> Tensor<T> mask_of_greater_than_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_greater_than(T first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_less_than_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_less_than(T first, const Tensor<T>& other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val);

       // Tensor and Scalar
       template <typename T> Tensor<T> mask_of_greater_than_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_greater_than(const Tensor<T>& first, T other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_less_than_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_less_than(const Tensor<T>& first, T other,  T first_val, T second_val);
       template <typename T> Tensor<T> mask_of_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val);

       template <typename T> Tensor<T> sign(const Tensor<T>& a);

       template <typename T> Tensor<T> inverse(const Tensor<T>& a);

       // TODO: implement im2col
       template <typename T> Tensor<T> im2col_2d(Tensor<T>& a, int kernel_size, int stride, int padding, int dilation);

    }

    namespace neural_network {
        // DT is the Tensor element type; constant_value follows that element type
        template <typename DT>
        Tensor<DT> padding(const simplenet::Tensor<DT>& input, int pad_amount, Padding_Op_Code padding_mode = Padding_Op_Code::PAD_CONSTANT, DT constant_value = DT(0));

    }

    template <typename T>
    class Tensor {

        // ==============================PRIVATE========================================
        private:
            // All Tensor<U> instantiations are mutual friends so dtype-converting
            // ops (e.g. change_dtype) can access another instantiation's private data.
            template <typename U> friend class Tensor;

            // The below line was added by an LLM to help me in the template instantiation
            // CPU compute (Eigen / std::math / host arithmetic) is only well-formed for these element
            // types. __nv_bfloat16 / __half are CUDA-only: their CPU branches are if-constexpr'd out
            // and throw at runtime. Used to keep all explicit instantiations compiling.
            static constexpr bool kHostCompute =
                std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int>;

            // inline static so each Tensor<T> instantiation gets its own definition
            inline static size_t print_precision = 14;
            static constexpr double MIN_DIFF = 1e-12;

            std::vector<int> shape;
            std::vector<int> strides; // will be used in permute and in GEMM
            std::vector<int> strides_col_major; // TODO: use this to make all ops transpose friendly and avoid copy operations
            T * data; // need to change to Tensor<T> where T can be custom data types like int8, float16, float32, float64 (double)
            bool owns_data;  // NEEDED To not cause the double destructor deletion of the broadcasting methods
            Device device;
            std::unique_ptr<DeviceAllocator> allocator_; // the device allocator we will be using for moving -> we only want a unique allocator
            size_t data_offset;
            bool is_sliced_view;

            // default constructor - added for edge cases - private ONLY -> cpu only allocation
            Tensor() : data(nullptr), owns_data(false), device(Device(DeviceType::CPU, -1)) , allocator_(nullptr), data_offset(0), is_sliced_view(false){};

            static Tensor makeBroadcastView(const Tensor &t, const std::vector<int>& newShape) {
                Tensor v;            // default-constructed
                v.device = t.device; // same device (broadcasting)
                v.data    = t.data;
                v.shape   = newShape;
                v.data_offset = t.data_offset;
                v.strides = utils::computeBroadcastStrides(t.shape, t.strides, newShape); // no need to change this
                v.strides_col_major = compute_col_major_strides(newShape); // cached canonical col-major strides for this shape
                v.owns_data = false; // we do not want the broadcasted tensors to own the data that it points - so we do not double delete
                v.allocator_ = nullptr; // we do not have an allocator for the views
                v.is_sliced_view = t.is_sliced_view;
                return v;
            }

            static Tensor makeSliceView(const Tensor& t,
                utils::SliceReturn sliceResult) {
                Tensor v;
                v.device     = t.device;
                v.data       = t.data;
                v.data_offset = sliceResult.offset;
                v.shape      = sliceResult.shape;
                v.strides    = sliceResult.strides;
                v.strides_col_major = compute_col_major_strides(sliceResult.shape); // cached canonical col-major strides for this shape
                v.owns_data  = false;       // parent owns the storage
                v.allocator_ = nullptr;
                v.is_sliced_view = true;
                return v;
            }


            // Shallow view of `t` with the same shape/strides/offset.
            // Used as a starting point for ops that produce a strided view by
            // permuting metadata (transpose, permute) without moving storage.
            static Tensor makeStrideView(const Tensor& t){
                Tensor v;
                v.device            = t.device;
                v.data              = t.data;
                v.data_offset       = t.data_offset;
                v.shape             = t.shape;
                v.strides           = t.strides;
                v.strides_col_major = t.strides_col_major;
                v.owns_data         = false;          // parent (or its owner) holds the storage
                v.allocator_        = nullptr;
                v.is_sliced_view    = t.is_sliced_view;
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

            static T getScalarValue(const Tensor& t){
                if (!isScalar(t)) {
                    throw std::runtime_error("Cannot get scalar value from non-scalar tensor");
                }

                T result;
                if (t.device.is_cpu()) {
                    result = t.data[0];
                } else {
                    // we copy to the host (cpu) to get the scalar value
                    t.allocator_->copy_to_host(&result, t.data, sizeof(T));
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
            Tensor(std::vector<int> sizePassed, const Device& device = Device::cpu()) :shape(sizePassed),  owns_data(true), device(device), data_offset(0), is_sliced_view(false){ // we own the data here
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
                size_t bytes = sizeTensor*sizeof(T);
                this->data = static_cast<T*>(allocator_->allocate(bytes));

                // initialize to zero
                if (this->device.is_cpu()) {
                    std::fill_n(this->data, sizeTensor, 0.0);
                } else {
                    // Zero initialize on GPU
                    CUDA_CHECK(cudaMemset(this->data, 0, bytes));
                }

            };


            // copy constructor
            Tensor(const Tensor& other) : owns_data(true), shape(other.shape), data_offset(0), is_sliced_view(false){ // we own the data when we copy;
                computeStrides();
                this->allocator_.reset(get_allocator(other.device));
                size_t full_size = sizeOfTensor();
                size_t bytes = full_size * sizeof(T);
                this->data =static_cast<T*>(allocator_->allocate(bytes));

                if (other.is_contiguous()) {
                    if (this->device == other.device) {
                        // cpu to cpu or gpu to gpu
                        allocator_->copy_device_to_device(this->data, other.data + other.data_offset, bytes);
                    } else {
                        if (this->device.is_cpu()) {
                            // from device to cpu
                            other.allocator_->copy_to_host(this->data, other.data + other.data_offset, bytes);
                        } else {
                            // from cpu to device
                            this->allocator_->copy_to_device(this->data, other.data + other.data_offset, bytes);
                        }
                    }
                    this->device = other.device;
                } else {
                    // other is a strided view (transpose / permute / slice / broadcast).
                    // gather it into our freshly allocated dst, which is row-major for `shape`.
                    // We stay on other's device — cross-device copies still require an explicit .to().
                    this->device = other.device;
                    const int nd = (int)other.shape.size();
                    if (other.device.type == DeviceType::CUDA) {
                        // device kernel wants shape/strides on the device; pack them into size_t and hand off
                        std::vector<size_t> h_shape(nd);
                        std::vector<size_t> h_strides(nd);
                        for (int d = 0; d < nd; ++d) {
                            h_shape[d]   = (size_t)other.shape[d];
                            h_strides[d] = (size_t)other.strides[d];
                        }
                        cuda::utils::launch_contiguous_gather<T>(
                            other.data, this->data, other.data_offset,
                            h_shape.data(), h_strides.data(), (size_t)nd, full_size);
                    } else {
                        // i walks the DESTINATION in row-major (dst is dense, so we write data[i] in flat order)
                        for (size_t i = 0; i < full_size; ++i) {
                            size_t tmp = i;                 // running quotient — peels one axis at a time
                            size_t src = other.data_offset; // start at the view's offset into the source's storage
                            // innermost dim varies fastest in row-major, so we modulo it out first and work outwards
                            for (int d = nd - 1; d >= 0; --d) {
                                size_t coord = tmp % (size_t)other.shape[d];   // coord along axis d for this dst index
                                tmp /= (size_t)other.shape[d];                 // strip that axis from tmp for the next iteration
                                src += coord * (size_t)other.strides[d];       // step in source storage by source's own stride along axis d
                            }
                            this->data[i] = other.data[src];
                        }
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
                    this->strides_col_major = other.strides_col_major;
                    this->owns_data = true;  // Copy always owns its data
                    this->data_offset = other.data_offset;
                    this->is_sliced_view = other.is_sliced_view;

                    this->allocator_.reset(get_allocator(device));
                    size_t bytes = sizeOfTensor() * sizeof(T);
                    this->data =static_cast<T*>(allocator_->allocate(bytes));
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
            Tensor(Tensor&& other) noexcept : owns_data(other.owns_data), shape(std::move(other.shape)), strides(std::move(other.strides)), strides_col_major(std::move(other.strides_col_major)), device(other.device), data(other.data), allocator_(std::move(other.allocator_)), data_offset(other.data_offset){
                // this->shape = other.shape;
                // this->data = other.data;
                // this->device = other.device;
                // this->strides = other.strides;
                other.data = nullptr;
                other.owns_data = false;
                other.is_sliced_view = false;
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
                    this->strides_col_major = std::move(other.strides_col_major);
                    this->allocator_ = std::move(other.allocator_);
                    this->owns_data = other.owns_data;
                    this->data_offset = other.data_offset;
                    other.data = nullptr;
                    other.owns_data = false;
                    other.is_sliced_view = false;
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
                size_t bytes = sizeOfTensor() * sizeof(T);
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

                size_t bytes = sizeOfTensor() * sizeof(T);
                auto new_allocator = std::unique_ptr<DeviceAllocator>(get_allocator(targetDevice));
                T* new_data = static_cast<T*>(new_allocator->allocate(bytes));

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


            // A Tensor<T> cannot change its own dtype in place: T is fixed by the class template and `data` is T*.
            // Assumes a contiguous tensor (only data_offset is respected, like to()).
            // TODO: handle strided / non-contiguous views -> depends if we want to support in-place dtype change
            template <typename T2>
            Tensor<T2> change_dtype() const {
                if (!this->is_contiguous()) {
                    throw std::runtime_error("change_dtype: cannot change dtype of non-contiguous/sliced view");
                }

                if constexpr (std::is_same_v<T, T2>) {
                    return *this; // same dtype -> just a copy
                } else {
                    Tensor<T2> new_tensor(this->shape, this->device);
                    const size_t n = sizeOfTensor();

                    if (this->device.is_cuda()){
                        cuda::utils::launch_dtype_change<T, T2>(this->data + data_offset, new_tensor.data, n);
                    } else {
                        for (size_t i = 0; i < n; ++i) {
                            new_tensor.data[i] = static_cast<T2>(this->data[data_offset + i]);
                        }
                    }

                    return new_tensor;
                }
            }

            // TODO: redefine as (row or col) once copy constructor, change_dtype, contiguous() route by layout
            // checks if the tensor is contiguous or not (row-major only for now)
            bool is_contiguous() const {
                  size_t s = 1;
                  for (int d = (int)shape.size() - 1; d >= 0; --d) {
                      if ((size_t)strides[d] != s) return false;
                      s *= shape[d];
                  }
                  return data_offset == 0;
            }

            // strides match canonical row-major (C order) for the current shape AND offset is zero
            bool is_row_major_contiguous() const {
                return is_contiguous();
            }

            // strides match canonical col-major (Fortran order) for the current shape AND offset is zero
            bool is_col_major_contiguous() const {
                if (strides.size() != shape.size()) return false;
                size_t v = 1;
                for (size_t d = 0; d < shape.size(); ++d) {
                    if ((size_t)strides[d] != v) return false;
                    v *= shape[d];
                }
                return data_offset == 0;
            }

            // Returns the layout descriptor for this tensor's current strides.
            // Note: for 1-D and scalar tensors row-major and col-major coincide;
            // we report ROW_MAJOR in that case (row is checked first).
            utils::Layout layout() const {
                if (is_row_major_contiguous()) return utils::Layout::ROW_MAJOR;
                if (is_col_major_contiguous()) return utils::Layout::COL_MAJOR;
                return utils::Layout::STRIDED;
            }

            // Accessor for the cached canonical col-major strides of this tensor's shape.
            std::vector<int> getStridesColMajor() const { return this->strides_col_major; }

            Device getDevice() const {
                return this->device;
            }


            static size_t getPrintPrecision() {
                return print_precision;
            }

            static void setPrintPrecision(size_t new_precision) {
                print_precision = new_precision;
            }


            T get(std::vector<int> index) const {
                if (!this->device.is_cpu()) {
                    throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use get()");
                }
                if (index.size() != shape.size()){
                    throw std::invalid_argument("Invalid index size: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" +utils::debugShapes(this->shape)+"\n");
                }

                if (utils::isIndexValid(index, this->shape) == false){
                    throw std::invalid_argument("Invalid index shape: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" + utils::debugShapes(this->shape)+"\n");
                }

                size_t off = 0;
                for (size_t d = 0; d < shape.size(); ++d)
                    off += index[d] * this->strides[d];
                return data[this->data_offset+ off];
            }

            T get(std::span<int> index) const {
                if (!this->device.is_cpu()) {
                    throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use get()");
                }
                if (index.size() != shape.size()){
                    throw std::invalid_argument("Invalid index size: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" +utils::debugShapes(this->shape)+"\n");
                }

                if (utils::isIndexValid(index, this->shape) == false){
                    throw std::invalid_argument("Invalid index shape: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" + utils::debugShapes(this->shape)+"\n");
                }

                size_t off = 0;
                for (size_t d = 0; d < shape.size(); ++d)
                    off += index[d] * this->strides[d];
                return data[this->data_offset+ off];
            }


            void set(T val, std::vector<int> index) const {

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

                data[this->data_offset+ off] = val;
            }

            // TODO: refactor
            // TODO: add test to check offset values
            void set_with_offset(ll offset, int row, int col, T val){
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

            // helper function
            size_t getDataOffset() const {
                return this->data_offset;
            }

            static std::vector<int> compute_row_major_strides(const std::vector<int>& sh) {
                std::vector<int> s(sh.size());
                size_t v = 1;
                for (int d = (int)sh.size() - 1; d >= 0; --d) {
                    s[d] = (int)v;
                    v *= sh[d];
                }
                return s;
            }

            static std::vector<int> compute_col_major_strides(const std::vector<int>& sh) {
                std::vector<int> s(sh.size());
                size_t v = 1;
                for (size_t d = 0; d < sh.size(); ++d) {
                    s[d] = (int)v;
                    v *= sh[d];
                }
                return s;
            }

            void computeStrides() {
                strides            = compute_row_major_strides(shape);
                strides_col_major  = compute_col_major_strides(shape);
            }

            // utility functions
            static Tensor ones(const std::vector<int>& shape, const Device& device = Device::cpu()) {
                Tensor t(shape, device);
                t.fill(1.0); // use fill for better performance
                return t;
            }

            static Tensor ones_like(const Tensor& t) {
                Tensor result(t.shape, t.device);
                result.fill(1.0); // use fill for better performance
                return result;
            }

            // baseline
            static Tensor zeros_like(const Tensor& t) {
                return Tensor(t.shape, t.device);
            }

            // helper function for recursive printing
            static void print_recursive(std::ostream& os, const Tensor& t, size_t dim, size_t offset,int indent) {
                  if (dim == t.shape.size() - 1) {
                      os << "[";
                      for (size_t i = 0; i < t.shape[dim]; ++i) {
                          os << std::setw(9) << std::setprecision(print_precision)
                             << t.data[offset + i * t.strides[dim]];
                          if (i + 1 < t.shape[dim]) os << ", ";
                      }
                      os << "]";
                      return;
                  }
                  os << "[";
                  for (size_t i = 0; i < t.shape[dim]; ++i) {
                      if (i > 0) {
                          os << ",\n";
                          // one blank line between 2-D "chunks" when there are >2 outer dims
                          if (dim < t.shape.size() - 2) os << "\n";
                          os << std::string(indent + 1, ' ');
                      }
                      print_recursive(os, t, dim + 1, offset + i * t.strides[dim], indent + 1);
                  }
                  os << "]";
              }


            friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {


                os << "Tensor with shape: [";
                for (size_t i = 0; i < tensor.shape.size(); ++i) {
                    os << tensor.shape[i];
                    if (i != tensor.shape.size() - 1) os << ", ";
                }
                os << "]\n";

                os << "Tensor on device: ";
                os << tensor.device.to_string() << "\n";

                os << "Tensor dtype: " << utils::print_type<T>() << "\n"; // cause we templated so T is not known at compile time

                os << "Tensor data:\n";
                if (tensor.device.is_cpu()) {
                    // No copy: walk the view in place.
                    if (tensor.shape.empty()) {
                        os << tensor.data[tensor.data_offset];
                    } else {
                        print_recursive(os, tensor, 0, tensor.data_offset, 0);
                    }
                } else {
                    // copy constructor used here
                    Tensor host = tensor.to(Device::cpu());
                    if (host.shape.empty()) {
                        os << host.data[host.data_offset];
                    } else {
                        print_recursive(os, host, 0, host.data_offset, 0);
                    }
                }
                os << "\n";

                os << "OWNERSHIP: " << ((tensor.owns_data) ? "TRUE" : "FALSE") << std::endl;
                os << "SLICED VIEW: " << ((tensor.getDataOffset() != 0) ? "TRUE" : "FALSE") << std::endl;

                return os;
            }

            // use the span overload for get() - which allows for any contiguous bloc
            T operator()(std::initializer_list<int> indices) {
                return this->get(indices);
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
                        cuda::launch_elementwise_contiguous<T>(A.data, B.data, C.data, C.getShape(), op);
                        return C;
                    }
                    auto outShape = utils::computeBroadcastShape(A.shape, B.shape);

                    Tensor aView = makeBroadcastView(A, outShape);
                    Tensor bView = makeBroadcastView(B, outShape);

                    // aView.to_(A.device);
                    // bView.to_(A.device);

                    Tensor C(outShape, A.device);

                    cuda::launch_elementwise_broadcast<T>(aView.data, bView.data, C.data,
                        aView.getStrides(), bView.getStrides(), C.getShape(), op);
                    return C;
                }
                // CPU path with appropriate lambda
                switch(op) {
                    case OP_Code::OP_ADD: return elementwise_binary_cpu(A, B, std::plus<T>{});
                    case OP_Code::OP_SUB: return elementwise_binary_cpu(A, B, std::minus<T>{});
                    case OP_Code::OP_MUL: return elementwise_binary_cpu(A, B, std::multiplies<T>{});
                    case OP_Code::OP_DIV: return elementwise_binary_cpu(A, B, std::divides<T>{});
                    case OP_Code::OP_MAX: return elementwise_binary_cpu(A, B, [](T a, T b){ return std::max(a,b); });
                    case OP_Code::OP_MIN: return elementwise_binary_cpu(A, B, [](T a, T b){ return std::min(a,b); });
                    default:
                        throw std::invalid_argument("OP Code does not exist - in elementwise_binary.");
                }
            }

            // Tensor-scalar operator: handles Tensor op scalar and scalar op Tensor
            static Tensor elementwise_scalar(const Tensor& A, T b, OP_Code op, LHS_RHS_Code side) {
                Tensor C(A.shape, A.device);
                if (A.device.type == DeviceType::CUDA) {
                    cuda::launch_elementwise_contiguous_with_constant<T>(A.data, b, C.data, A.shape, op, side);
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

                        cuda::launch_elementwise_broadcast<T>(data, other.data, data,
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
                        cuda::launch_elementwise_contiguous<T>(data, other.data, data, shape, op);
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
            Tensor& inplace_scalar(T b, OP_Code op) {
                if (device.type == DeviceType::CUDA) {
                    cuda::launch_elementwise_contiguous_with_constant<T>(this->data, b, this->data, shape, op, LHS_RHS_Code::OP_RHS);
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
                    cuda::launch_elementwise_unary<T>(
                        A.data,
                        C.data,
                        C.getShape(),
                        op
                    );
                    return C;
                }
                // CPU path with appropriate lambda
                switch(op) {
                    case OP_Code::OP_EXP: return elementwise_unary_cpu(A, [](T a){ return std::exp(a); });
                    case OP_Code::OP_LOG: return elementwise_unary_cpu(A, [](T a){ return std::log(a); });
                    case OP_Code::OP_SQRT: return elementwise_unary_cpu(A, [](T a){ return std::sqrt(a); });
                    case OP_Code::OP_ABS: return elementwise_unary_cpu(A, [](T a){ return std::abs(a); });
                    case OP_Code::OP_SIN: return elementwise_unary_cpu(A, [](T a){ return std::sin(a); });
                    case OP_Code::OP_COS: return elementwise_unary_cpu(A, [](T a){ return std::cos(a); });
                    case OP_Code::OP_TAN: return elementwise_unary_cpu(A, [](T a){ return std::tan(a); });
                    case OP_Code::OP_SINH: return elementwise_unary_cpu(A, [](T a){ return std::sinh(a); });
                    case OP_Code::OP_COSH: return elementwise_unary_cpu(A, [](T a){ return std::cosh(a); });
                    case OP_Code::OP_TANH: return elementwise_unary_cpu(A, [](T a){ return std::tanh(a); });
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
            Tensor& operator+=(const T& b) {
                return inplace_scalar(b, OP_Code::OP_ADD);
            }

            // element wise add
            friend Tensor operator+(const Tensor &A, const T& b) {
                return elementwise_scalar(A, b, OP_Code::OP_ADD, LHS_RHS_Code::OP_RHS);
            }

            friend Tensor operator+(const T& b, const Tensor &A) {
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
            Tensor& operator-=(const T& b) {
                return inplace_scalar(b, OP_Code::OP_SUB);
            }
            // element wise subtract
            friend Tensor operator-(const Tensor &A, const T& b) {
                return elementwise_scalar(A, b, OP_Code::OP_SUB, LHS_RHS_Code::OP_RHS);
            }
            // element wise subtract - operation is switched (b - A)
            friend Tensor operator-(const T& b, const Tensor &A) {
                return elementwise_scalar(A, b, OP_Code::OP_SUB, LHS_RHS_Code::OP_LHS);
            }

            // --------------------------------MULTIPLICATION----------------------------------------------------------------------

            // Hadamard product
            template<typename U> friend Tensor<U> linear_algebra::hadamard(const Tensor<U> &a, const Tensor<U> &other);

            // THIS is where we will be doing the multiplication when the dimensions exceed the normal 2 of a matrix
            template<typename U> friend Tensor<U> linear_algebra::batchedMatMul(const Tensor<U>& a, const Tensor<U>& b);

            // TODO: write CUDA kernel
            template<typename U> friend Tensor<U> linear_algebra::reduce(const Tensor<U>& a, std::vector<int>& afterShape, reductions::ReductionOps op);

            // element wise multiply (now uses elementwise_scalar dispatch)
            friend Tensor operator*(const Tensor &A, const T& b) {
                return elementwise_scalar(A, b, OP_Code::OP_MUL, LHS_RHS_Code::OP_RHS);
            }

            // for when the operations are reversed
            friend Tensor operator*(const T& b, const Tensor &A) {
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
                        cuda::launch_gemm_contiguous<T>(
                            a.data, b.data, result.data,
                            1,    // batchsize
                            1,    // m (rows of a treated as row vector)
                            a_shape[0], // k (common dim),
                            1,    // n (cols of result)
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const VectorXT<T>> vec_a(a.data, a_shape[0]);
                    Eigen::Map<const VectorXT<T>> vec_b(b.data, b_shape[0]);

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
                        cuda::launch_gemm_contiguous<T>(
                            a.data, b.data, result.data,
                            1,           // batchsize
                            a_shape[0],  // m
                            a_shape[1],  // k (common dim),
                            1,           // n (output cols)
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const MatrixRowMajorT<T>> mat_a(a.data, a_shape[0], a_shape[1]);
                    Eigen::Map<const VectorXT<T>> vec_b(b.data, b_shape[0]);

                    Tensor result({a_shape[0]});
                    Eigen::Map<VectorXT<T>> result_vec(result.data, a_shape[0]);
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
                        cuda::launch_gemm_contiguous<T>(
                            a.data, b.data, result.data,
                            1,           // batchsize
                            1,           // m
                            b_shape[0],  // k (common dim),
                            b_shape[1],  // n (output cols)
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const VectorXT<T>> vec_a(a.data, a_shape[0]);
                    Eigen::Map<const MatrixRowMajorT<T>> mat_b(b.data, b_shape[0], b_shape[1]);


                    Tensor result({b_shape[1]});
                    Eigen::Map<VectorXT<T>> result_vec(result.data, b_shape[1]);
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
                        cuda::launch_gemm_contiguous<T>(
                            a.data, b.data, result.data,
                            1,           // batchsize
                            a_shape[0],  // m
                            a_shape[1],  // k
                            b_shape[1],  // n
                            1.0, 0.0, nullptr
                        );
                        return result;
                    }

                    Eigen::Map<const MatrixRowMajorT<T>> mat_a(a.data, a_shape[0], a_shape[1]);
                    Eigen::Map<const MatrixRowMajorT<T>> mat_b(b.data, b_shape[0], b_shape[1]);

                    Tensor result({a_shape[0], b_shape[1]});
                    Eigen::Map<MatrixRowMajorT<T>> result_mat(result.data, a_shape[0], b_shape[1]);
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
            Tensor& operator*=(const T& B) {
                 *this = *this * B;
                 return *this;
            }

            // -------------------------------DIVISION--------------------------------------------------------------------------
            friend Tensor operator/(const Tensor &A, const T& b) {
                return elementwise_scalar(A, b, OP_Code::OP_DIV, LHS_RHS_Code::OP_RHS);
            }

            friend Tensor operator/(const T& b, const Tensor &A) {
                return elementwise_scalar(A, b, OP_Code::OP_DIV, LHS_RHS_Code::OP_LHS);
            }

            friend Tensor operator/(const Tensor &a, const Tensor &b) {
                if (b.shape != a.shape){
                    throw std::runtime_error("Shapes should match for element-wise divide");
                }
                return elementwise_binary(a, b, OP_Code::OP_DIV);
            }

            // Another case exists but that is when matrix b is invertible and then it just becomes matrix mul
            template<typename U> friend Tensor<U> linear_algebra::inverse(const Tensor<U>& a); // TODO: add CUDA dispatch function to compute inverse


            // --------------------------------EQUALITY--------------------------------------------------------------------------

            // will change as float precision is added
            friend bool operator==(const Tensor &a, const Tensor &b){
                if (a.getShape() == b.getShape() && a.getStrides() == b.getStrides()){
                    if (a.getDevice().type == DeviceType::CUDA && b.getDevice().type == DeviceType::CUDA){
                        return cuda::launch_check_equal_kernel<T>(a.data, b.data, a.sizeOfTensor()); // need to test this
                    }
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


            template<typename U> friend Tensor<U> linear_algebra::compare(const Tensor<U>& a, const Tensor<U>& b,CompareOp op, U true_val, U false_val);

            template<typename U> friend Tensor<U> linear_algebra::mask_of_greater_than_equal_to(const Tensor<U>& first, const Tensor<U>& other, U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_greater_than(const Tensor<U>& first, const Tensor<U>& other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_less_than_equal_to(const Tensor<U>& first, const Tensor<U>& other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_less_than(const Tensor<U>& first, const Tensor<U>& other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_equal_to(const Tensor<U>& first, const Tensor<U>& other,  U first_val, U second_val);

            // Double and Tensor
            template<typename U> friend Tensor<U> linear_algebra::mask_of_greater_than_equal_to(U first, const Tensor<U>& other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_greater_than(U first, const Tensor<U>& other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_less_than_equal_to(U first, const Tensor<U>& other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_less_than(U first, const Tensor<U>& other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_equal_to(U first, const Tensor<U>& other,  U first_val, U second_val);

            // Tensor and Double
            template<typename U> friend Tensor<U> linear_algebra::mask_of_greater_than_equal_to(const Tensor<U>& first, U other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_greater_than(const Tensor<U>& first, U other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_less_than_equal_to(const Tensor<U>& first, U other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_less_than(const Tensor<U>& first, U other,  U first_val, U second_val);
            template<typename U> friend Tensor<U> linear_algebra::mask_of_equal_to(const Tensor<U>& first, U other,  U first_val, U second_val);


            // sign matrix
            template<typename U> friend Tensor<U> linear_algebra::sign(const Tensor<U>& a);

            // TODO: implement im2col
            template<typename U> friend Tensor<U> linear_algebra::im2col_2d(Tensor<U>& a, int kernel_size, int stride, int padding, int dilation);



            // padding (in convolution_layers.h file)
            template<typename DU> friend Tensor<DU> neural_network::padding(const simplenet::Tensor<DU>& input, int pad_amount, Padding_Op_Code padding_mode, DU constant_value);

            // TODO: refactor for native CUDA support
            //----------------------------------------ACCUMULATORs (used in reduce)------------------------------------------------------
            Tensor accumulate(int dim, reductions::ReductionOps op, bool keepdims = false){
                if (dim<0 || dim>=shape.size()){
                    throw std::invalid_argument("DIM not in the correct range!");
                }

                // edge cases to consider - when we only have a vector then sum will give a scalar
                if (shape.size()==1 && dim ==0){
                    // no need to check keep dims as if keepdims is false then it will be a scalar anyways
                    Tensor new_t({1}, this->device);
                    new_t.data[0] = (op == reductions::ReductionOps::PROD) ? T(1)
                                             : (op == reductions::ReductionOps::MAX)  ? std::numeric_limits<T>::lowest()
                                             : (op == reductions::ReductionOps::MIN)  ?  std::numeric_limits<T>::max()
                                             : T(0); // even for argmin/argmax initial value is 0
                    // argmin/argmax need the running comparison value seeded to the
                    // opposite extreme (the index in new_t.data[0] stays 0 by default)
                    T value = (op == reductions::ReductionOps::ARG_MAX) ? std::numeric_limits<T>::lowest()
                            : (op == reductions::ReductionOps::ARG_MIN) ? std::numeric_limits<T>::max()
                            : new_t.data[0]; // for argmin/argmax
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
                int oldDim = newShape[dim]; // same as dim width
                newShape[dim] = 1; // we will change the shape afterwards

                ll offset_new_shape{1};
                for (int d = dim+1; d <newShape.size(); d++){
                    offset_new_shape*=newShape[d];
                }

                ll offset_old{offset_new_shape*oldDim};

                Tensor new_t(newShape, this->device);
                T* flat_data = new_t.data;

                if (this->device.type == DeviceType::CUDA) {
                    cuda::launch_accumulate_kernel<T>(this->data, flat_data, this->shape, newShape, sizeOfTensor(), offset_new_shape, offset_old, op,  keepdims);

                } else {

                    // gpu direct access not allowed, so we copy to CPU first
                    Tensor copy_tensor = *this;
                    copy_tensor.to_(Device(DeviceType::CPU, -1));

                    ll dest_idx = 0;
                    for (size_t v =0; v<sizeOfTensor(); v+=offset_old){
                        for (size_t s = 0; s<offset_new_shape;s++){
                            // accumulate initial value based on reduction op
                            T val = (op == reductions::ReductionOps::PROD) ? T(1)
                                                     : (op == reductions::ReductionOps::MAX || op == reductions::ReductionOps::ARG_MAX)  ? std::numeric_limits<T>::lowest()
                                                     : (op == reductions::ReductionOps::MIN || op == reductions::ReductionOps::ARG_MIN)  ?  std::numeric_limits<T>::max()
                                                     : T(0);
                            int64_t arg_idx = 0;

                            for (int idx = 0; idx<oldDim; idx++){
                                // edited from this->data to copy_tensor.data
                                T elem = copy_tensor.data[v + idx * offset_new_shape + s];
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
                }


                if (keepdims) {
                    // new_t.to_(this->device);
                    return new_t;
                }
                new_t.flatten_inplace((dim<shape.size()-1)? dim : (dim-1),  (dim<shape.size()-1) ? dim+1 : -1, keepdims); // PROBLEM FOUND HERE in case when the dim passed in dim= shape.size()-1
                // new_t.to_(this->device);
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
            static Tensor max(const Tensor& t,const  T val){
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

            static Tensor max(const T val, const Tensor& t){
                return Tensor::max(t,val);
            }

            static Tensor max(const Tensor& t, const Tensor& s){
                // std::cout <<"MAX" <<std::endl;
                if (t.getShape() != s.getShape()){
                    throw std::invalid_argument("Shapes dont match for max operation");
                }
                if (t.device == DeviceType::CUDA) {
                    Tensor result(t.getShape(), t.device);
                    cuda::launch_elementwise_contiguous<T>(
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

            static Tensor min(const Tensor& t, T val){
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

            static Tensor min(const T val, const Tensor& t){
                return Tensor::min(t,val);
            }

            static Tensor min(const Tensor& t, const Tensor& s){
                // std::cout <<"MIN" <<std::endl;
                if (t.getShape() != s.getShape()){
                    throw std::invalid_argument("Shapes dont match for min operation");
                }
                if (t.device == DeviceType::CUDA) {
                    Tensor result(t.getShape(), t.device);
                    cuda::launch_elementwise_contiguous<T>(
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
                T sum = T(0);
                if (t.device == DeviceType::CUDA) {
                    cuda::launch_sum_kernel(t.data, result.data, t.sizeOfTensor());
                    result.to_(Device::cpu()); // send to CPU to get the sum as GPU direct memory access is NOT SET UP
                    result.set(static_cast<T>(result.data[0] / static_cast<double>(t.sizeOfTensor())), {0}); // set the mean value
                    result.to_(t.device); // send back to GPU
                }else {
                    size_t sizeTensor = t.sizeOfTensor();
                    for (size_t i =0; i<sizeTensor; i++){
                       sum += t.data[i];

                    }
                    result.set(static_cast<T>(sum / static_cast<double>(t.sizeOfTensor())), {0}) ;
                }
                return result;
            }


            //---------------------------------------- Log ------------------------------------------------------

            static Tensor log(Tensor &t ){
                return elementwise_unary(t, OP_Code::OP_LOG);
            }


            // Tensor(bool owns_data) : data(nullptr), owns_data(owns_data) {};
            void fill(T v){
                // CUDA fill
                if (!this->device.is_cpu()) {
                    cuda::launch_fill<T>(
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

            // flatten - definition moved inline (was in Tensor.cpp; templates need
            // the definition visible at instantiation)
            Tensor flatten(int start_dim =0, int end_dim = -1, bool keepdims=false) {
                Tensor result = *this; // copy the tensor
                std::vector<int> newShape = this->flatten_(start_dim, end_dim, keepdims);
                result.setShape(newShape);
                result.computeStrides();
                return result;
            }

            // flatten - inplace
            void flatten_inplace(int start_dim =0, int end_dim = -1, bool keepdims=false){
                this->shape = Tensor::flatten_(start_dim, end_dim, keepdims);
                computeStrides();
            }


            // linspace function to edit the current tensor
            Tensor& linspace(T start, T end){
                size_t long_size = this->sizeOfTensor();
                double size = static_cast<double>(long_size);
                // accumulate in double for precision, store back as the element type T
                double cur = static_cast<double>(start);
                double step = (static_cast<double>(end)-cur+1)/(size);
                if (this->device.is_cpu()) {
                    for (size_t i =0; i < long_size ; i++){
                        this->data[i] = static_cast<T>(cur);
                        cur+=step;
                    }
                } else {
                    cuda::launch_linspace_kernel<T>(this->data, static_cast<T>(start), static_cast<T>(step), long_size);
                }
                return *this;
            }


            // Returns an O(1) non-owning view that transposes the last two dims.
            // No data is moved; call .contiguous() before feeding into a
            // row-major-only kernel (GEMM, element_wise_contiguous, etc.).
            Tensor transpose(){
                // scalar / single-element: nothing to do
                if (sizeOfTensor() <= 1) {
                    return makeStrideView(*this);
                }

                if (this->shape.size() < 2){
                    throw std::invalid_argument("transpose requires a tensor of rank >= 2");
                }

                Tensor v = makeStrideView(*this);
                const int n = (int)v.shape.size();
                std::swap(v.shape[n-2],             v.shape[n-1]);
                std::swap(v.strides[n-2],           v.strides[n-1]);
                std::swap(v.strides_col_major[n-2], v.strides_col_major[n-1]);
                return v;
            }


            // Permute - Not the same as TRANSPOSE
            // Returns a non-owning strided view; no data is moved.
            Tensor permute(std::vector<int> new_order){
                Tensor result = makeStrideView(*this);
                result.inplace_permute(new_order);
                return result;
            }

            void inplace_permute(std::vector<int> new_order) {
                if (new_order.size() != this->shape.size()){
                    throw std::invalid_argument("Permute order must have same length as tensor rank");
                }
                std::vector<bool> seen(this->shape.size(), false);
                for (int i = 0; i < (int)new_order.size(); i++){
                    if (new_order[i] < 0 || new_order[i] >= (int)this->shape.size()){
                        throw std::invalid_argument("Invalid permute order");
                    }
                    if (seen[new_order[i]]){
                        throw std::invalid_argument("Permute order must be a permutation (no duplicates)");
                    }
                    seen[new_order[i]] = true;
                }

                std::vector<int> new_shape(this->shape.size());
                std::vector<int> new_strides(this->shape.size());
                std::vector<int> new_strides_col_major(this->shape.size());
                for (int i = 0; i < (int)new_order.size(); i++){
                    new_shape[i]             = this->shape[new_order[i]];
                    new_strides[i]           = this->strides[new_order[i]];
                    new_strides_col_major[i] = this->strides_col_major[new_order[i]];
                }
                this->shape             = std::move(new_shape);
                this->strides           = std::move(new_strides);
                this->strides_col_major = std::move(new_strides_col_major);
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
            // Finalized (default dim = 0   )
            void unsqueeze(int dim = 0){
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


            static Tensor contiguous(const Tensor& t, const Device& device = Device::cpu()) {
                if (t.is_contiguous()) return t; // already row-major + offset 0, nothing to do

                // result is a fresh row-major tensor with the same shape — its storage will be written
                // in linear order (data[0], data[1], ...) by the loop below
                Tensor result(t.shape, t.device);
                const size_t n   = t.sizeOfTensor();
                const int    nd  = (int)t.shape.size();

                if (t.device.type == DeviceType::CUDA) {
                    // device kernel needs shape/strides on the device; copy them into size_t buffers and hand off
                    std::vector<size_t> h_shape(nd);
                    std::vector<size_t> h_strides(nd);
                    for (int d = 0; d < nd; ++d) {
                        h_shape[d]   = (size_t)t.shape[d];
                        h_strides[d] = (size_t)t.strides[d];
                    }
                    cuda::utils::launch_contiguous_gather<T>(
                        t.data, result.data, t.data_offset,
                        h_shape.data(), h_strides.data(), (size_t)nd, n);
                } else {
                    // i walks the DESTINATION row-major (so we write result.data[i] in the natural flat order)
                    for (size_t i = 0; i < n; ++i) {
                        size_t tmp = i;               // running quotient — peels one axis at a time
                        size_t src = t.data_offset;   // running flat index into source storage; non-zero when t is a slice view
                        // innermost dim varies fastest in row-major, so we modulo it out first and work outward
                        for (int d = nd - 1; d >= 0; --d) {
                            size_t coord = tmp % (size_t)t.shape[d];   // coord along axis d for this destination index
                            tmp /= (size_t)t.shape[d];                  // strip that axis from tmp so the next iteration sees the outer dims
                            src += coord * (size_t)t.strides[d];        // step in source storage by source's own stride along axis d
                                                                        // -> for transpose this is the swapped dim's stride, for broadcast it's 0, for slice it's the parent's stride
                        }
                        result.data[i] = t.data[src]; // dst is dense, so flat index == i; src lands wherever the source's strides put us
                    }
                }
                return result;
            }

            // non static variation
            Tensor contiguous(const Device& device = Device::cpu()) {
                if (is_contiguous()) return *this;
                return Tensor::contiguous(*this, device);
            }


            static Tensor slice(const Tensor& t, std::string parse) {
                // We will have to compute a new shape and a new stride as well
                // [start0:end0, start1:end1, ...]

                // use python-like slice syntax
                // parse is a string that represents the slice operation
                // e.g. "0:10, 5:15" means slice the first 10 elements of the first dimension and the next 10 elements of the second dimension
                // parse the slice operation
                std::vector<std::string> sliceOps;
                boost::split(sliceOps, parse, boost::is_any_of(","));
                // check if the number of slice operations matches the number of dimensions
                if (sliceOps.size() != t.getShape().size()) {
                    throw std::invalid_argument("Number of slice operations does not match number of dimensions.");
                }

                // parse the slice operations into start and end indices for each dimension
                size_t dim = 0;
                std::vector<utils::Slice> sliceIndices;
                for (const auto& op : sliceOps) {
                    std::string cleaned = op;
                    std::erase(cleaned, ' ');
                    size_t dim_size = t.getShape()[dim];

                    // check if the slice operation is valid
                    if (cleaned ==":") {
                        sliceIndices.emplace_back(0, dim_size, 1, utils::SliceMode::FULL);
                    }else if (cleaned.find(':') == std::string::npos) {
                        int idx = std::stoi(cleaned);
                        sliceIndices.emplace_back(idx, idx + 1, 1, utils::SliceMode::RANGE);
                    } else {
                        std::vector<std::string> indices;
                        boost::split(indices, cleaned, boost::is_any_of(":"));
                        if (indices.size() > 3) throw std::invalid_argument("Invalid slice: " + cleaned);
                        int start = indices[0].empty() ? 0       : std::stoi(indices[0]);
                        int end   = indices[1].empty() ? dim_size : std::stoi(indices[1]);
                        int step  = 1;
                        if (indices.size() > 2 && !indices[2].empty()) step = std::stoi(indices[2]);
                        if (start < 0 || start >= dim_size || end < start || end > dim_size || step <= 0)
                            throw std::invalid_argument("Invalid slice: " + cleaned);
                        sliceIndices.emplace_back(start, end, step, utils::SliceMode::RANGE);
                    }
                    ++dim;

                }

                auto sliced_result = utils::computing_slice_parameters(t.getShape(), t.getStrides(), sliceIndices);

                Tensor slicedView = Tensor::makeSliceView(t, sliced_result);

                return slicedView;
            }

            // non-static variant of slice
            Tensor slice(std::string parse) {
                return Tensor::slice(*this, parse);
            }

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

                if (device.type == DeviceType::CUDA) {
                    const size_t n = tensors.size();

                    std::vector<T*> h_data(n);
                    std::vector<int*> h_shape_ptrs(n); // each entry is a device pointer

                    for (size_t i = 0; i < n; ++i) {
                        const Tensor& t = tensors.begin()[i];
                        h_data[i] = t.data; // get the raw data pointer

                        int* d_shape = nullptr;
                        size_t shape_size = t.getShape().size();
                        const size_t bytes = shape_size * sizeof(int);
                        // allocate device memory for the shape and copy from host
                        CUDA_CHECK(cudaMalloc(&d_shape, bytes));
                        CUDA_CHECK(cudaMemcpy(d_shape, t.getShape().data(), bytes, cudaMemcpyHostToDevice));
                        h_shape_ptrs[i] = d_shape;
                    }

                    T** d_allInputs = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_allInputs, n * sizeof(T*)));
                    CUDA_CHECK(cudaMemcpy(d_allInputs, h_data.data(),
                                          n * sizeof(T*), cudaMemcpyHostToDevice));

                    // allocate the copied shapes pointers in a list of pointers on the device
                    int** d_shapes = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_shapes, n * sizeof(int*)));
                    CUDA_CHECK(cudaMemcpy(d_shapes, h_shape_ptrs.data(),
                                          n * sizeof(int*), cudaMemcpyHostToDevice));

                    cuda::launch_concat_kernel<T>(d_allInputs, d_shapes, n, result.data,
                                                       outerDim, innerDim, dim, concatDim);

                    for (int* d_shape : h_shape_ptrs) CUDA_CHECK(cudaFree(d_shape));
                    CUDA_CHECK(cudaFree(d_allInputs));
                    CUDA_CHECK(cudaFree(d_shapes));
                } else {
                    for (size_t o = 0; o < outerDim; ++o) {
                        int offset = 0; // offset will be used to move the pointer to execute the copies
                        for (size_t i = 0; i < tensors.size(); ++i) {
                            int src_cat_dim = tensors.begin()[i].getShape()[dim];
                            int copy_size = src_cat_dim * innerDim;
                            // we copy the data for each tensor into the result tensor
                            // outer*concatDim * innerDim moves the pointer to the correct position in the result tensor
                            // offset * innerDim is the offset that will be copied from the source tensor using (o*src_cat_dim*innerDim)
                            std::memcpy(result.data  + o*concatDim *innerDim + offset*innerDim, tensors.begin()[i].data + o*src_cat_dim*innerDim, copy_size * sizeof(T));
                            offset += src_cat_dim;
                        }
                    }
                }

                // if (device.type == DeviceType::CUDA) {
                //     result.to_(device);
                // }

                return result;
            }


            // TODO: CUDA support
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
                T * temp = new T[total];
                size_t stride = this->sizeOfTensor();

                this->shape.insert(this->shape.begin(), tensors.size()+1);
                computeStrides();


                std::copy(this->data, this->data + stride, temp + 0);
                ll start = stride;
                for (const Tensor& tensor : tensors){
                    T * tempData = tensor.data;

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

    // dtype aliases - the supported element types for Tensor<T>
    using TensorD  = Tensor<double>;          // 64-bit float (default / legacy precision)
    using Tensorf  = Tensor<float>;           // 32-bit float
    using TensorI  = Tensor<int>;             // 32-bit signed int


    // trait: is_tensor_v<U> is true iff U is some Tensor<E> specialization.
    // Used by autograd so Node<T> stays generic over Tensor<T> (any element
    // type) instead of hard-coding a concrete alias like TensorD.
    template<typename>   struct is_tensor                : std::false_type {};
    template<typename E> struct is_tensor<Tensor<E>>     : std::true_type  {};
    template<typename U> inline constexpr bool is_tensor_v = is_tensor<U>::value;
}

// linear_algebra friend templates - definitions included here so they are visible
// for instantiation, now that Tensor<T> is a complete type (breaks the include cycle).
#include "utils/linalg_utils.tpp"
#endif
