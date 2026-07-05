#pragma once

#include "../includes/cuda_helper.h"

namespace bearml {
    namespace cuda {

        // declaration
        template<typename T>
        void launch_padd_with_constant(T* d_in, T* d_out, int batch_size, int in_rows, int in_cols, int padding, T constant, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_PADDING
        #define INSTANTIATE_PADDING(T) \
            template __global__ void padd_with_constant_kernel(const T* __restrict__ d_in, T* __restrict__ d_out, int batch_size, int in_rows,  int in_cols,  int padding); \
            template void launch_padd_with_constant<T>(T* d_in, T* d_out, int batch_size, int in_rows, int in_cols, int padding, T constant, cudaStream_t stream);
        #endif


    }
}
