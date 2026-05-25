#pragma once

#include "../includes/cuda_helper.h"

namespace simplenet {
    namespace cuda {

        template<typename T>
        void launch_gemm_broadcasted(
            const T* d_a,
            const T* d_b,
            T* d_c,
            int m,
            int k,
            int n,
            T alpha,
            T beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );

        template<typename T>
        void launch_gemm_contiguous(
            T* d_a,
            T* d_b,
            T* d_c,
            int batchsize,
            int m,
            int n,
            int k,
            T alpha,
            T beta,
            int64_t row_aware_strde_a, int64_t col_aware_stride_a,
            int64_t row_aware_strde_b, int64_t col_aware_stride_b,
            cudaStream_t stream
        );


        #ifndef INSTANTIATE_GEMM
        #define INSTANTIATE_GEMM(T) \
            template __global__ void gemm_kernel_broadcast<T>(const int* __restrict__ batch_shape, int batch_shape_size, const int64_t* __restrict__ strides_a, const int64_t* __restrict__ strides_b, const int M, const int K, const int N, T alpha, T beta, int64_t total_batch_size, const T* __restrict__ a, const T* __restrict__ b, T* c); \
            template __global__ void gemm_kernel_contiguous<T>(int batchsize, const int M, const int K, const int N, T alpha, T beta, int64_t row_aware_strde_a, int64_t col_aware_stride_a, int64_t row_aware_strde_b, int64_t col_aware_stride_b, T* __restrict__ a, T* __restrict__ b, T* c); \
            template void launch_gemm_broadcasted<T>( const T* d_a, const T* d_b, T* d_c, int m, int k, int n, T alpha, T beta, std::vector<int>* batch_shape, int batch_shape_size, std::vector<int64_t>* strides_a, std::vector<int64_t>* strides_b, int64_t total_batch_size, cudaStream_t stream); \
            template void launch_gemm_contiguous<T>(T* d_a, T* d_b, T* d_c, int batchsize, int m, int k, int n, T alpha, T beta, int64_t row_aware_strde_a, int64_t col_aware_stride_a, int64_t row_aware_strde_b, int64_t col_aware_stride_b, cudaStream_t stream);
        #endif

    }
}
