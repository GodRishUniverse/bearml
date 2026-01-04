#pragma once
#include <vector>
#include <stdexcept>


namespace simplenet {
    class Tensor; // forward declaration
    namespace linear_algebra {

        Tensor batchedMatMul(const Tensor& a, const Tensor& b);

        Tensor reduce(const Tensor& a, std::vector<int>& afterShape);

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

        // // Opposite of broadcasting - NOT A VIEW OPERATION
        // Tensor reduce(const Tensor &t, const std::vector<int>& targetShape);

        // // The idea for a reduction is summation and flattening so this just makes it explicit
        // Tensor flatten_and_sum_to_shape(const Tensor &t, const std::vector<int>& targetShape);
    }
}
