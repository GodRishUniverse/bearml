#pragma once

#include "../includes/cuda_helper.h"
#include <cstdint>


namespace bearml {
    namespace cuda {

        template<typename T>
        void launch_sum_kernel(
            T *d_data,
            T *result,
            int64_t size,
            cudaStream_t stream = nullptr
        );

        template<typename T>
        void launch_sum_acc_kernel(T *d_data, T *d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_SUM
        #define INSTANTIATE_SUM(T) \
            template __global__ void sum_kernel<T>(const T *d_data, T *result, int64_t size); \
            template void launch_sum_kernel<T>(T *d_data, T *result, int64_t size, cudaStream_t stream); \
            template __global__ void sum_acc_kernel<T>(const T *d_data, T *d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size); \
            template void launch_sum_acc_kernel<T>(T *d_data, T *d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size, cudaStream_t stream);
        #endif
    }
}
