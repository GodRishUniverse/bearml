#pragma once
#include <vector>
#include <stdexcept>
#include "compare_ops.h" // to get the compare ops enum
#include "../reductions/reduction_ops.h" // to get the reduction ops enum

namespace simplenet {
    class Tensor; // forward declaration


    namespace linear_algebra {
        // TODO: add CPU side operator that allows a custom Comparison
        // TODO: add CUDA to all these to allow back propagation

        Tensor compare(const Tensor& a, const Tensor& b,CompareOp op, double true_val, double false_val);

        Tensor batchedMatMul(const Tensor& a, const Tensor& b);

        // Reduction operations (default is sum unless specified otherwise)
        Tensor reduce(const Tensor& a, std::vector<int>& afterShape, reductions::ReductionOps op= reductions::ReductionOps::SUM);

        Tensor hadamard(const Tensor &a, const Tensor &other);

        Tensor mask_of_greater_than_equal_to(const Tensor& first, const Tensor& other,  double first_val = 1.0, double second_val = 0.0);
        Tensor mask_of_greater_than(const Tensor& first, const Tensor& other, double first_val = 1.0, double second_val = 0.0);
        Tensor mask_of_less_than_equal_to(const Tensor& first, const Tensor& other,  double first_val = 1.0, double second_val = 0.0);
        Tensor mask_of_less_than(const Tensor& first, const Tensor& other,  double first_val = 1.0, double second_val = 0.0);
        Tensor mask_of_equal_to(const Tensor& first, const Tensor& other, double first_val = 1.0, double second_val = 0.0);



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

        // sign matrix
        Tensor sign(const Tensor& a);

        // inverse matrix
        Tensor inverse(const Tensor& a);


        // TODO: im2col
        Tensor im2col(const Tensor& a, int kernel_size, int stride, int padding, int dilation);

        // // Opposite of broadcasting - NOT A VIEW OPERATION
        // Tensor reduce(const Tensor &t, const std::vector<int>& targetShape);

        // // The idea for a reduction is summation and flattening so this just makes it explicit
        // Tensor flatten_and_sum_to_shape(const Tensor &t, const std::vector<int>& targetShape);
    }
}
