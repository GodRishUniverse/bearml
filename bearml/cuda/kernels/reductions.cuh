#pragma once

#include "../includes/cuda_helper.h"

namespace bearml {
    namespace cuda {

        template<typename T>
        void launch_prod_acc_kernel(T *d_data, T *d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_PROD_ACC
        #define INSTANTIATE_PROD_ACC(T) \
            template __global__ void prod_acc_kernel<T>(const T* d_data,T* d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size); \
            template void launch_prod_acc_kernel<T>(T *d_data, T *d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size, cudaStream_t stream = nullptr);
        #endif

    }
}
