#pragma once
#include <vector>
#include <stdexcept>
#include "compare_ops.h" // to get the compare ops enum
#include "../reductions/reduction_ops.h" // to get the reduction ops enum

namespace bearml {
    template<typename T> class Tensor; // forward declaration


    namespace linear_algebra {
        // TODO: add CPU side operator that allows a custom Comparison
        // TODO: add CUDA to all these to allow back propagation

        template<typename T> Tensor<T> compare(const Tensor<T>& a, const Tensor<T>& b,CompareOp op, T true_val, T false_val);

        template<typename T> Tensor<T> batchedMatMul(const Tensor<T>& a, const Tensor<T>& b);

        // Reduction operations (default is sum unless specified otherwise)
        template<typename T> Tensor<T> reduce(const Tensor<T>& a, std::vector<int>& afterShape, reductions::ReductionOps op= reductions::ReductionOps::SUM);

        template<typename T> Tensor<T> hadamard(const Tensor<T> &a, const Tensor<T> &other);

        template<typename T> Tensor<T> mask_of_greater_than_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val = T(1), T second_val = T(0));
        template<typename T> Tensor<T> mask_of_greater_than(const Tensor<T>& first, const Tensor<T>& other, T first_val = T(1), T second_val = T(0));
        template<typename T> Tensor<T> mask_of_less_than_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val = T(1), T second_val = T(0));
        template<typename T> Tensor<T> mask_of_less_than(const Tensor<T>& first, const Tensor<T>& other,  T first_val = T(1), T second_val = T(0));
        template<typename T> Tensor<T> mask_of_equal_to(const Tensor<T>& first, const Tensor<T>& other, T first_val = T(1), T second_val = T(0));



        // Scalar and Tensor
        template<typename T> Tensor<T> mask_of_greater_than_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_greater_than(T first, const Tensor<T>& other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_less_than_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_less_than(T first, const Tensor<T>& other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val);

        // Tensor and Scalar
        template<typename T> Tensor<T> mask_of_greater_than_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_greater_than(const Tensor<T>& first, T other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_less_than_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_less_than(const Tensor<T>& first, T other,  T first_val, T second_val);
        template<typename T> Tensor<T> mask_of_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val);

        // sign matrix
        template<typename T> Tensor<T> sign(const Tensor<T>& a);

        // inverse matrix
        template<typename T> Tensor<T> inverse(const Tensor<T>& a);


        // TODO: im2col
        template<typename T> Tensor<T> im2col_2d(Tensor<T>& a, int kernel_size, int stride, int padding, int dilation);

        // // Opposite of broadcasting - NOT A VIEW OPERATION
        // Tensor reduce(const Tensor &t, const std::vector<int>& targetShape);

        // // The idea for a reduction is summation and flattening so this just makes it explicit
        // Tensor flatten_and_sum_to_shape(const Tensor &t, const std::vector<int>& targetShape);
    }
}

// Template definitions live in the .tpp; it is included at the bottom of
// tensor/Tensor.h (after Tensor<T> is a complete type) to break the include cycle.
