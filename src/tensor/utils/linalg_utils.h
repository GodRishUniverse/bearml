#pragma once
#include <vector>
#include <stdexcept>


namespace simplenet {
    class Tensor; // forward declaration
    namespace linear_algebra {

        Tensor batchedMatMul(const Tensor& a, const Tensor& b);

         Tensor reduce(const Tensor& a, std::vector<int>& afterShape);

        // // Opposite of broadcasting - NOT A VIEW OPERATION
        // Tensor reduce(const Tensor &t, const std::vector<int>& targetShape);

        // // The idea for a reduction is summation and flattening so this just makes it explicit
        // Tensor flatten_and_sum_to_shape(const Tensor &t, const std::vector<int>& targetShape);
    }
}
