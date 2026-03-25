#pragma once

#include "../includes/cuda_helper.h"
#include "../includes/ops.h"
#include <vector>
#include <cstddef>

namespace simplenet {
    namespace cuda {

        template <typename T>
        void launch_elementwise_broadcast(
            const T* d_a,
            const T* d_b,
            T* d_out,
            const std::vector<int>& strides_a,
            const std::vector<int>& strides_b,
            const std::vector<int>& res_shape,
            OP_Code op_code,
            cudaStream_t stream = nullptr
        );

        template <typename T>
        void launch_elementwise_contiguous(
            const T* d_a,
            const T* d_b,
            T* d_out,
            const std::vector<int>& res_shape,
            OP_Code op_code,
            cudaStream_t stream = nullptr
        );


        // template <typename T>
        // void launch_elementwise_broadcast_with_constant(
        //     const T* d_a,
        //     const T b,
        //     T* d_out,
        //     const std::vector<int>& strides_a,
        //     const std::vector<int>& res_shape,
        //     OP_Code op_code,
        //     LHS_RHS_Code lhs_rhs_code,
        //     cudaStream_t stream = nullptr;
        // );


        template <typename T>
        void launch_elementwise_contiguous_with_constant(
            const T* d_a,
            const T b,
            T* d_out,
            const std::vector<int>& res_shape, // same as d_a shape cause contiguous
            OP_Code op_code,
            LHS_RHS_Code lhs_rhs_code,
            cudaStream_t stream = nullptr
        );

        template <typename T>
        void launch_elementwise_unary(
            const T* d_a,
            T* d_out,
            const std::vector<int>& res_shape,
            OP_Code op_code,
            cudaStream_t stream = nullptr
        );


        template <typename T>
        void launch_sign_contiguous(
            const T* d_a,
            T* d_out,
            const std::vector<int>& res_shape,
            cudaStream_t stream = nullptr
        );
    }
}
