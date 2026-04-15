#pragma once

#include "../includes/cuda_helper.h"
#include <cstdint>


namespace simplenet {
    namespace cuda {

        template<typename T>
        void launch_transpose_kernel(T* input, T* output, long long int batch_size, int rows, int cols, cudaStream_t stream= nullptr);

        #ifndef INSTANTIATE_TRANSPOSE
        #define INSTANTIATE_TRANSPOSE(T) \
        template __global__ void transpose_kernel<T>(T * input, T* output,  long long int batch_size, int rows, int cols); \
        template void launch_transpose_kernel(T* input, T* output, long long int batch_size, int rows, int cols, cudaStream_t stream= nullptr);
        #endif
    }
}
