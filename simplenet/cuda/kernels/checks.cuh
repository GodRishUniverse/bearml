#pragma once

#include "../includes/cuda_helper.h"
#include <cstdint>

namespace simplenet {
    namespace cuda {

        template<typename T>
        bool launch_check_zero_kernel(T *d_data, int64_t size, cudaStream_t stream = nullptr);

        template<typename T>
        bool launch_check_equal_kernel(T *d_data, T *d_other, int64_t size, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_CHECKS
        #define INSTANTIATE_CHECKS(T) \
            template __global__ void check_nonzero_kernel<T>(const T *d_data, size_t size, int* result); \
            template bool launch_check_zero_kernel<T>(T *d_data, int64_t size, cudaStream_t stream); \
            template __global__ void check_equal_kernel<T>(const T* data, const T* other, size_t n, int* result); \
            template bool launch_check_equal_kernel<T>(T *d_data, T *d_other, int64_t size, cudaStream_t stream);
        #endif
    }
}
