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


        #ifndef INSTANTIATE_ELEMENT_WISE
        #define INSTANTIATE_ELEMENT_WISE(T) \
            template __global__ void simplenet::cuda::element_wise_broadcast<T>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const T*, const T*, T*, OP_Code); \
            template __global__ void simplenet::cuda::element_wise_contiguous<T>(const T* __restrict__ a,const T* __restrict__ b, T* res, size_t n, OP_Code op_code); \
            template __global__ void simplenet::cuda::element_wise_contiguous_with_constant<T>(const T*, const T*, T*, size_t, OP_Code, LHS_RHS_Code); \
            template void launch_elementwise_broadcast<T>(const T*, const T*, T*, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, OP_Code, cudaStream_t); \
            template void launch_elementwise_contiguous<T>(const T*, const T*, T*, const std::vector<int>&, OP_Code, cudaStream_t); \
            template void launch_elementwise_contiguous_with_constant<T>(const T*, const T, T*, const std::vector<int>&, OP_Code, LHS_RHS_Code, cudaStream_t); \
            template __global__ void simplenet::cuda::element_wise_unary<T>(const T*, T*, size_t, OP_Code); \
            template void launch_elementwise_unary<T>(const T*, T*, const std::vector<int>&, OP_Code, cudaStream_t);
        #endif

        template <typename T>
        void launch_sign_contiguous(
            const T* d_a,
            T* d_out,
            const std::vector<int>& res_shape,
            cudaStream_t stream = nullptr
        );
    }
}
