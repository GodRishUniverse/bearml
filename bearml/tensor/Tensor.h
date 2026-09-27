#pragma once
#include "tensor/Storage.h"
#include <cmath>
#include <cstddef>
#include <cstring>
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
#include <bit>

#include <span>

// C++23 fixed-width floating types (std::bfloat16_t). Only available on the host (CPU)
// (g++13+ have C++23 mode); NVCC compiles .cu at C++17 so this stays excluded there.
// __STDCPP_BFLOAT16_T__ is defined by the compiler only when std::bfloat16_t exists.
#if __has_include(<stdfloat>)
    #include <stdfloat>
#endif

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
#include "utils/dtype.h"

#include "Scalar.h"

#include "reductions/reduction_ops.h"

// #include <cmath>
#include <iomanip>


// Operation-code enums (OP_Code, LHS_RHS_Code, Padding_Op_Code). These are
// plain C++ and used by CPU paths too, so they must NOT sit behind the CUDA
// guard. Previously they only reached here transitively via cuda_imports.h.
#include "operators/ops.h"
#include "operators/padding_ops.h"

#if defined(BEARML_USE_CUDA)
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
namespace bearml{

    // forward declaration of the class template - needed by the free-function declarations below
    template <typename T> class Tensor;
    template <typename T> struct Node;               // autograd record, defined in autograd/autogradient.h
    namespace autogradient { struct TensorAccess; } // autograd's access to graph_node

    // bearml::cuda_type_trait<U>::type maps a host element type U to the CUDA
    // kernel type used to launch it (defaults to U itself; overridden for the
    // handful of types whose device representation differs from the host one).
    // Declared before class Tensor so inline members can use it — must be
    // visible before its first use, since it's a free namespace-scope template
    // rather than a member of Tensor.
    template <typename U> struct cuda_type_trait{ using type = U; };
    #if defined(BEARML_USE_CUDA) && defined(__STDCPP_BFLOAT16_T__)
        template<> struct cuda_type_trait<std::bfloat16_t> { using type = __nv_bfloat16; };
    #endif
    #if defined(BEARML_USE_CUDA)
        // TODO: F16 needs its own host type (std::float16_t) distinct from int16_t
        template<> struct cuda_type_trait<int16_t> { using type = __half; };
    #endif
    template <typename U> using cuda_type_trait_t = typename cuda_type_trait<U>::type;

    // Reinterprets a host element pointer as the CUDA kernel type launch_* functions expect.
    // No-op cast for types where host and device representations coincide (float, double, ints);
    // bit-reinterpret for the handful (bf16/half) where the host storage type and CUDA type differ.
    template <typename U>
    inline cuda_type_trait_t<U>* cuda_ptr(U* p) { return reinterpret_cast<cuda_type_trait_t<U>*>(p); }
    template <typename U>
    inline const cuda_type_trait_t<U>* cuda_ptr(const U* p) { return reinterpret_cast<const cuda_type_trait_t<U>*>(p); }

    // Same idea as cuda_ptr but for by-value scalar args (e.g. the constant in
    // elementwise-with-constant launches) — bit_cast since host and device types
    // are same-size, same-layout but distinct C++ types.
    template <typename U>
    inline cuda_type_trait_t<U> cuda_val(U v) { return std::bit_cast<cuda_type_trait_t<U>>(v); }

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
        Tensor<DT> padding(const bearml::Tensor<DT>& input, int pad_amount, Padding_Op_Code padding_mode = Padding_Op_Code::PAD_CONSTANT, DT constant_value = DT(0));

    }

    template <typename T>
    class Tensor {

        // ==============================PRIVATE========================================
        private:
            // All Tensor<U> instantiations are mutual friends so dtype-converting
            // ops (e.g. change_dtype) can access another instantiation's private data.
            template <typename U> friend class Tensor;

            // inline static so each Tensor<T> instantiation gets its own definition
            inline static size_t print_precision = 14;
            static constexpr double MIN_DIFF = 1e-12;

            Shape tensor_shape; // dims, strides and offset into storage
            std::shared_ptr<Storage> storage; // the storage for this tensor - will handle

            Device device;

            std::shared_ptr<bearml::Node<Tensor<T>>> graph_node; // integration with autograd
            friend struct autogradient::TensorAccess;
            template <typename U> friend struct Node;   // Node default-constructs its val/grad

            // elements in the storage (a view's shape can be smaller)
            size_t storage_elements() const {
                return storage ? storage->number_of_bytes() / sizeof(T) : 0;
            }

            // Debug-only bounds check for at()
            void check_in_storage(size_t flat) const {
#ifndef NDEBUG
                if (flat >= storage_elements()) {
                    throw std::out_of_range("Tensor element " + std::to_string(flat) +
                        " is outside its storage (" + std::to_string(storage_elements()) + " elements)");
                }
#endif
            }

            // fresh storage for the current shape and device
            void allocate_storage() {
                storage = std::make_shared<Storage>(sizeOfTensor() * sizeof(T), this->device);
            }

            void record_op(OP_Code op, std::initializer_list<const Tensor*> inputs,
                          OpAttributes attrs, Tensor& output);

            // in-place ops don't record: reject them on tracked intermediates and with a tracked operand
            void check_inplace(const Tensor* other = nullptr) const;


            // default constructor - added for edge cases - private ONLY -> cpu only allocation
            Tensor() :  device(Device(DeviceType::CPU, -1)){};

            static Tensor makeBroadcastView(const Tensor &t, const std::vector<int>& newShape);

            static Tensor makeSliceView(const Tensor& t,
                utils::SliceReturn sliceResult);


            // Shallow view of `t` with the same shape/strides/offset.
            // Used as a starting point for ops that produce a strided view by
            // permuting metadata (transpose, permute) without moving storage.
            static Tensor makeStrideView(const Tensor& t);


            static bool isScalar(const Tensor& t);

            static Scalar<T> getScalarValue(const Tensor& t);

            std::vector<int> flatten_(int start_dim =0, int end_dim = -1, bool keepdims=false); // has a Tensor return type specialization in the cpp file

            // utility for permute
            void setShape(std::vector<int> &newShape){
                this->tensor_shape.shape = newShape;
            }

            // ==============================PRIVATE========================================


            // untracked matrix/scalar product behind operator*
            static Tensor matmul(const Tensor &a, const Tensor &b);


            // untracked elementwise max behind max(t, s)
            static Tensor max_impl(const Tensor& t, const Tensor& s);


            // untracked elementwise min behind min(t, s)
            static Tensor min_impl(const Tensor& t, const Tensor& s);


            // untracked reduction behind accumulate()
            Tensor accumulate_impl(int dim, reductions::ReductionOps op, bool keepdims = false);


            // untracked transposed view behind transpose()
            Tensor transpose_view() const;

            // operator<< and operator== bodies, defined in Tensor.cpp
            static std::ostream& print_impl(std::ostream& os, const Tensor& tensor);
            static bool equal_impl(const Tensor &a, const Tensor &b);

        public:

            // flat index into storage (caller adds data_offset); bounds-checked in Debug
            T& at(size_t flat) {
                check_in_storage(flat);
                return storage->data<T>()[flat];
            }

            const T& at(size_t flat) const {
                check_in_storage(flat);
                return storage->data<T>()[flat];
            }

            // raw pointers for CUDA kernels, Eigen::Map and memcpy
            T* mutable_data() { return storage ? storage->data<T>() : nullptr; }
            const T* const_data() const { return storage ? storage->data<T>() : nullptr; }

            // ------------------------------ autograd (defined in autograd/autogradient.cpp) ------------------------------
            void set_requires_grad(bool on = true);
            bool requires_grad() const;
            const Tensor& grad() const;                  // throws if the tensor is untracked
            void zero_grad();

            // untracked and detached copy of this tensor
            Tensor detach() const { Tensor v = makeStrideView(*this); v.graph_node = nullptr; return v; }

            // shares storage and the graph node
            Tensor shared_view() const { Tensor v = makeStrideView(*this); v.graph_node = graph_node; return v; }


            // constructor when size and data are provided -> copies on same device as I want to ensure the programmer has explicit knowledge of where the tensor is and should use .to before doing "cross-devices" copies
            Tensor(std::vector<int> sizePassed, const Device& device = Device::cpu());


            // copy constructor
            // a copy stays on the source's device; use .to() to move it
            Tensor(const Tensor& other);

            // copy assignment operator
            Tensor& operator=(const Tensor& other);

            // move constructor - repoints the shared storage
            Tensor(Tensor&& other) noexcept;

            // move assignment operator
            Tensor& operator=(Tensor&& other) noexcept;

            // destructor - shared_ptr<Storage> frees the allocation
            ~Tensor() = default;

            // copy to
            Tensor to(const Device& targetDevice) const;


            // inplace to - new storage, so a view stops sharing with its parent
            void to_(const Device& targetDevice);


            // dtype can't change in place: T is fixed for this Tensor<T>
            // Assumes a contiguous tensor (only data_offset is respected).
            // TODO: handle strided / non-contiguous views -> depends if we want to support in-place dtype change
            template <typename T2>
            Tensor<T2> change_dtype() const {
                if (!this->is_contiguous()) {
                    throw std::runtime_error("change_dtype: cannot change dtype of non-contiguous/sliced view");
                }

                if constexpr (std::is_same_v<T, T2>) {
                    return *this; // same dtype -> just a copy
                } else {
                    Tensor<T2> new_tensor(this->tensor_shape.shape, this->device);
                    const size_t n = sizeOfTensor();

                    if (this->device.is_cuda()){
                        cuda::utils::launch_dtype_change<cuda_type_trait_t<T>, cuda_type_trait_t<T2>>(cuda_ptr(this->const_data() + tensor_shape.data_offset), cuda_ptr(new_tensor.mutable_data()), n);
                    } else {
                        for (size_t i = 0; i < n; ++i) {
                            new_tensor.at(i) = static_cast<T2>(this->at(tensor_shape.data_offset + i));
                        }
                    }

                    return new_tensor;
                }
            }

            // TODO: redefine as (row or col) once copy constructor, change_dtype, contiguous() route by layout
            // checks if the tensor is contiguous or not (row-major only for now)
            bool is_contiguous() const;

            // strides match canonical row-major (C order) for the current shape AND offset is zero
            bool is_row_major_contiguous() const {
                return is_contiguous();
            }

            // strides match canonical col-major (Fortran order) for the current shape AND offset is zero
            bool is_col_major_contiguous() const;

            // Returns the layout descriptor for this tensor's current strides.
            // Note: for 1-D and scalar tensors row-major and col-major coincide;
            // we report ROW_MAJOR in that case (row is checked first).
            utils::Layout layout() const;

            // Accessor for the cached canonical col-major strides of this tensor's shape.
            std::vector<int> getStridesColMajor() const { return this->tensor_shape.strides_col_major; }

            Device getDevice() const {
                return this->device;
            }


            static size_t getPrintPrecision() {
                return print_precision;
            }

            static void setPrintPrecision(size_t new_precision) {
                print_precision = new_precision;
            }


            Scalar<T> get(std::vector<int> index) const;

            Scalar<T> get(std::span<int> index) const;


            // only for contiguous tensors (this function will help with device-agnostic copying and memory management for bulk data like loading csv files, etc.)
            void set_using_bulk_copy(const std::vector<T>& values);

            void set(Scalar<T> val, std::vector<int> index);

            // TODO: refactor
            // TODO: add test to check offset values
            void set_with_offset(ll offset, int row, int col, Scalar<T> val);

            // helper function
            void printShape() const;

            // helper function
            size_t sizeOfTensor() const {
                size_t total = 1; // cause size may be huge
                for (const int & i : tensor_shape.shape){
                    total*= i;
                }
                return total;
            }

            // helper functions
            std::vector<int> getShape() const {
                return this->tensor_shape.shape;
            }

            // helper function
            std::vector<int> getStrides() const {
                return this->tensor_shape.strides;
            }

            // helper function
            size_t getDataOffset() const {
                return this->tensor_shape.data_offset;
            }

            // helper function
            static constexpr DType dtype() { return dtype_of<T>; }

            static std::vector<int> compute_row_major_strides(const std::vector<int>& sh);

            static std::vector<int> compute_col_major_strides(const std::vector<int>& sh);

            void computeStrides();

            // utility functions
            static Tensor ones(const std::vector<int>& shape, const Device& device = Device::cpu());

            static Tensor ones_like(const Tensor& t);

            // baseline
            static Tensor zeros_like(const Tensor& t) {
                return Tensor(t.tensor_shape.shape, t.device);
            }

            // build a tensor from a flat host buffer (e.g. parsed csv rows, decoded pixels)
            static Tensor from_host(const std::vector<int>& shape, const std::vector<T>& host,
                                    const Device& device = Device::cpu());

            // helper function for recursive printing
            static void print_recursive(std::ostream& os, const Tensor& t, size_t dim, size_t offset,int indent);


            friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) { return print_impl(os, tensor); }

            // use the span overload for get() - which allows for any contiguous bloc
            T operator()(std::initializer_list<int> indices) {
                return this->get(indices);
            }

            // TODO: REFACTOR ALL OPERATIONS for CUDA support as well
            // ================================ OPERATIONS ========================================================================

            // Reduced the repetitive code - CPU side element-wise binary — handles both contiguous and broadcast paths
            template<typename Func>
            static Tensor elementwise_binary_cpu(const Tensor& A, const Tensor& B, Func fn) {
                if (A.tensor_shape.shape == B.tensor_shape.shape) {
                    Tensor C(A.tensor_shape.shape, A.device);
                    for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                        C.at(i) = fn(A.at(i), B.at(i));
                    return C;
                }
                // broadcast path
                auto outShape = utils::computeBroadcastShape(A.tensor_shape.shape, B.tensor_shape.shape);
                Tensor aView = makeBroadcastView(A, outShape);
                Tensor bView = makeBroadcastView(B, outShape);
                Tensor C(outShape, A.device);
                size_t N = C.sizeOfTensor();
                for (ll idx = 0; idx < static_cast<ll>(N); ++idx) {
                    ll tmp = idx, offA = 0, offB = 0;
                    for (int d = static_cast<int>(outShape.size()) - 1; d >= 0; --d) {
                        int coord = tmp % outShape[d];
                        tmp /= outShape[d];
                        offA += coord * aView.tensor_shape.strides[d];
                        offB += coord * bView.tensor_shape.strides[d];
                    }
                    C.at(idx) = fn(A.at(offA), B.at(offB));
                }
                return C;
            }

            // Tensor-Tensor operator: uses device types to choose the path - uses OpCode for choosing the operator
            static Tensor elementwise_binary(const Tensor& A, const Tensor& B, OP_Code op);

            // Tensor-scalar operator: handles Tensor op scalar and scalar op Tensor
            static Tensor elementwise_scalar(const Tensor& A, T b, OP_Code op, LHS_RHS_Code side);

            // in-place Tensor-Tensor operator -> used only for + and -
            Tensor& inplace_tensor_binary(const Tensor& other, OP_Code op);

            // in-place scalar-Tensor operator
            Tensor& inplace_scalar(T b, OP_Code op);



            // Unary Operations

            template<typename Func>
            static Tensor elementwise_unary_cpu(const Tensor& A, Func fn) {
                Tensor C(A.tensor_shape.shape, A.device);
                for (size_t i = 0, N = A.sizeOfTensor(); i < N; ++i)
                    C.at(i) = fn(A.at(i));
                return C;
            }

            static Tensor elementwise_unary(const Tensor& A, OP_Code op);

            // ================================ OPERATORS - Using the above new element_wise functions =========================================

            // --------------------------------ADDITION----------------------------------------------------------------------------

            friend Tensor operator+(const Tensor &A, const Tensor &B) {
                Tensor out = elementwise_binary(A, B, OP_Code::OP_ADD);
                out.record_op(OP_Code::OP_ADD, {&A, &B}, {}, out);
                return out;
            }

            Tensor& operator+=(const Tensor &other);

            // element wise add
            Tensor& operator+=(Scalar<T> b);

            // element wise add
            friend Tensor operator+(const Tensor &A, Scalar<T> b) {
                Tensor out = elementwise_scalar(A, b.value(), OP_Code::OP_ADD, LHS_RHS_Code::OP_RHS);
                out.record_op(OP_Code::OP_ADD_SCALAR, {&A}, OpAttributes{.constant = static_cast<double>(b.value())}, out);
                return out;
            }

            friend Tensor operator+(Scalar<T> b, const Tensor &A) {
                return A+b;
            } // same as above


            // --------------------------------SUBTRACTION-------------------------------------------------------------------------
            friend Tensor operator-(const Tensor &A, const Tensor &B) {
                Tensor out = elementwise_binary(A, B, OP_Code::OP_SUB);
                out.record_op(OP_Code::OP_SUB, {&A, &B}, {}, out);
                return out;
            }
            Tensor& operator-=(const Tensor &other);
            // element wise subtract
            Tensor& operator-=(Scalar<T> b);
            // element wise subtract
            friend Tensor operator-(const Tensor &A, Scalar<T> b) {
                Tensor out = elementwise_scalar(A, b.value(), OP_Code::OP_SUB, LHS_RHS_Code::OP_RHS);
                out.record_op(OP_Code::OP_SUB_SCALAR, {&A}, OpAttributes{.constant = static_cast<double>(b.value())}, out);
                return out;
            }
            // element wise subtract - operation is switched (b - A)
            friend Tensor operator-(Scalar<T> b, const Tensor &A) {
                Tensor out = elementwise_scalar(A, b.value(), OP_Code::OP_SUB, LHS_RHS_Code::OP_LHS);
                out.record_op(OP_Code::OP_RSUB_SCALAR, {&A}, OpAttributes{.constant = static_cast<double>(b.value())}, out);
                return out;
            }

            // --------------------------------MULTIPLICATION----------------------------------------------------------------------

            // Hadamard product
            template<typename U> friend Tensor<U> linear_algebra::hadamard(const Tensor<U> &a, const Tensor<U> &other);

            // THIS is where we will be doing the multiplication when the dimensions exceed the normal 2 of a matrix
            template<typename U> friend Tensor<U> linear_algebra::batchedMatMul(const Tensor<U>& a, const Tensor<U>& b);

            // TODO: write CUDA kernel
            template<typename U> friend Tensor<U> linear_algebra::reduce(const Tensor<U>& a, std::vector<int>& afterShape, reductions::ReductionOps op);

            // element wise multiply (now uses elementwise_scalar dispatch)
            friend Tensor operator*(const Tensor &A, Scalar<T> b) {
                Tensor out = elementwise_scalar(A, b.value(), OP_Code::OP_MUL, LHS_RHS_Code::OP_RHS);
                out.record_op(OP_Code::OP_MUL_SCALAR, {&A}, OpAttributes{.constant = static_cast<double>(b.value())}, out);
                return out;
            }

            // for when the operations are reversed
            friend Tensor operator*(Scalar<T> b, const Tensor &A) {
                return A*b; // order of multiplication doesn't matter here -> does it?
            }


            friend Tensor operator*(const Tensor &a, const Tensor &b) {
                Tensor out = matmul(a, b);
                out.record_op(OP_Code::OP_MUL, {&a, &b}, {}, out);
                return out;
            }



            // In-place matrix multiplication - we call our function made above
            Tensor& operator*=(const Tensor& B);
            // calls the element wise mul
            Tensor& operator*=(Scalar<T> B);

            // -------------------------------DIVISION--------------------------------------------------------------------------
            friend Tensor operator/(const Tensor &A, Scalar<T> b) {
                Tensor out = elementwise_scalar(A, b.value(), OP_Code::OP_DIV, LHS_RHS_Code::OP_RHS);
                out.record_op(OP_Code::OP_DIV_SCALAR, {&A}, OpAttributes{.constant = static_cast<double>(b.value())}, out);
                return out;
            }

            friend Tensor operator/(Scalar<T> b, const Tensor &A) {
                Tensor out = elementwise_scalar(A, b.value(), OP_Code::OP_DIV, LHS_RHS_Code::OP_LHS);
                out.record_op(OP_Code::OP_RDIV_SCALAR, {&A}, OpAttributes{.constant = static_cast<double>(b.value())}, out);
                return out;
            }

            friend Tensor operator/(const Tensor &a, const Tensor &b) {
                if (b.tensor_shape.shape != a.tensor_shape.shape){
                    throw std::runtime_error("Shapes should match for element-wise divide");
                }
                Tensor out = elementwise_binary(a, b, OP_Code::OP_DIV);
                out.record_op(OP_Code::OP_DIV, {&a, &b}, {}, out);
                return out;
            }

            // Another case exists but that is when matrix b is invertible and then it just becomes matrix mul
            template<typename U> friend Tensor<U> linear_algebra::inverse(const Tensor<U>& a); // TODO: add CUDA dispatch function to compute inverse


            // --------------------------------EQUALITY--------------------------------------------------------------------------

            // will change as float precision is added
            friend bool operator==(const Tensor &a, const Tensor &b){ return equal_impl(a, b); }

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
            template<typename DU> friend Tensor<DU> neural_network::padding(const bearml::Tensor<DU>& input, int pad_amount, Padding_Op_Code padding_mode, DU constant_value);

            // TODO: refactor for native CUDA support
            //----------------------------------------ACCUMULATORs (used in reduce)------------------------------------------------------
            Tensor accumulate(int dim, reductions::ReductionOps op, bool keepdims = false);



            // Tensor argmax(int dim, bool keepdims = false);
            // Tensor argmin(int dim, bool keepdims = false);


            //----------------------------------------Exponential------------------------------------------------------
            static Tensor exp(Tensor& t);

            //----------------------------------------Sin------------------------------------------------------
            static Tensor sin(Tensor& t);

            //----------------------------------------Cos------------------------------------------------------
            static Tensor cos(Tensor& t);

            //----------------------------------------Tan------------------------------------------------------
            static Tensor tan(Tensor& t);

            //----------------------------------------Hyperbolic------------------------------------------------------
            static Tensor sinh(Tensor& t);

            static Tensor cosh(Tensor& t);

            static Tensor tanh(Tensor& t);



            //----------------------------------------Max------------------------------------------------------

            // TODO: fix this - CUDA kernel as well
            static Tensor max(const Tensor& t, Scalar<T> val);

            static Tensor max(Scalar<T> val, const Tensor& t){
                return Tensor::max(t,val);
            }

            static Tensor max(const Tensor& t, const Tensor& s);



            //----------------------------------------Min------------------------------------------------------

            static Tensor min(const Tensor& t, Scalar<T> val);

            static Tensor min(Scalar<T> val, const Tensor& t){
                return Tensor::min(t,val);
            }

            static Tensor min(const Tensor& t, const Tensor& s);



            //----------------------------------------Absolute value------------------------------------------------------
            // Note: const accepts both non-const and const tensors
            static Tensor abs(const Tensor &t );


            //----------------------------------------Square root------------------------------------------------------
            // Note: const accepts both non-const and const tensors

            static Tensor sqrt(const Tensor &t );



            //----------------------------------------Mean ------------------------------------------------------
            static Tensor mean(Tensor &t );


            //---------------------------------------- Log ------------------------------------------------------

            static Tensor log(Tensor &t );

            //---------------------------------------- Softmax ------------------------------------------------------
            static Tensor softmax(Tensor &t, int64_t dim =-1 );


            void fill(Scalar<T> v);

            // TODO: refactor
            static bool has_nonzero_gradient(Tensor& t);

            // flatten - definition moved inline (was in Tensor.cpp; templates need
            // the definition visible at instantiation)
            Tensor flatten(int start_dim =0, int end_dim = -1, bool keepdims=false);

            // flatten - inplace
            void flatten_inplace(int start_dim =0, int end_dim = -1, bool keepdims=false);


            // linspace function to edit the current tensor
            Tensor& linspace(Scalar<T> start, Scalar<T> end);


            // Returns an O(1) non-owning view that transposes the last two dims.
            // No data is moved; call .contiguous() before feeding into a
            // row-major-only kernel (GEMM, element_wise_contiguous, etc.).
            Tensor transpose();




            // Permute - Not the same as TRANSPOSE
            // Returns a non-owning strided view; no data is moved.
            Tensor permute(std::vector<int> new_order);

            void inplace_permute(std::vector<int> new_order);


            // reshape
            void reshape(std::vector<int> new_shape);


            // unsqueeze
            // Finalized (default dim = 0   )
            void unsqueeze(int dim = 0);

            // squeeze
            // default dim = 0
            void squeeze(int dim = 0 );


            static Tensor contiguous(const Tensor& t, const Device& device = Device::cpu());

            // non static variation
            Tensor contiguous(const Device& device = Device::cpu());


            static Tensor slice(const Tensor& t, std::string parse);

            // non-static variant of slice
            Tensor slice(std::string parse) {
                return Tensor::slice(*this, parse);
            }

            static Tensor concat(std::initializer_list<Tensor> tensors, int dim =0 );


            // TODO: CUDA support
            void stack(std::initializer_list<Tensor> tensors);

            // TODO
            // static Tensor identity_matrix(std::vector<int>& sizePassed, int n)

        };

    // dtype aliases - the supported element types for Tensor<T>
    using TensorD  = Tensor<double>;          // 64-bit float (default / legacy precision)
    using Tensorf  = Tensor<float>;           // 32-bit float
    #if defined(__STDCPP_BFLOAT16_T__)
        using TensorBF = Tensor<std::bfloat16_t>; // bfloat16
    #endif

    using TensorI  = Tensor<int>;             // 32-bit signed int

    // member bodies live in Tensor.cpp, instantiated there for these types
    extern template class Tensor<float>;
    extern template class Tensor<double>;
    extern template class Tensor<int>;
    #if defined(__STDCPP_BFLOAT16_T__)
        extern template class Tensor<std::bfloat16_t>;
    #endif

    // https://accu.org/journals/overload/9/43/frogley_442/ (C++ traits)
    // trait: is_tensor_v<U> is true iff U is some Tensor<E> specialization.
    // Used by autograd so Node<T> stays generic over Tensor<T> (any element
    // type) instead of hard-coding a concrete alias like TensorD.
    template<typename>   struct is_tensor                : std::false_type {};
    template<typename E> struct is_tensor<Tensor<E>>     : std::true_type  {};
    template<typename U> inline constexpr bool is_tensor_v = is_tensor<U>::value;

    // tensor_element_t<Tensor<E>> yields E, the scalar element type. Lets generic
    // code (e.g. autograd) cast scalars to a tensor's precision instead of assuming
    // double, so Tensor<float> instantiations don't hit double/float type clashes.
    template<typename>   struct tensor_element {};
    template<typename E> struct tensor_element<Tensor<E>> { using type = E; };
    template<typename U> using tensor_element_t = typename tensor_element<U>::type;

    // Convenience: true iff U is Tensor<E> with a floating-point element type.
    template<typename U>
    inline constexpr bool is_supported_float_tensor_v =
        is_tensor_v<U> && is_floating(dtype_of<tensor_element_t<U>>);

}

#endif
