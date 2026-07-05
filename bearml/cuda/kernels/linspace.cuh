#pragma once

#include "../includes/cuda_helper.h"


namespace bearml {
    namespace cuda {

        template<typename T>
        void launch_linspace_kernel(T* d_out, T start, T step, size_t size, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_LINSPACE_KERNEL
        #define INSTANTIATE_LINSPACE_KERNEL(T) \
            template __global__ void linspace( T* out,T start, T step, size_t size); \
            template void launch_linspace_kernel<T>(T* d_out, T start, T step, size_t size, cudaStream_t stream);
        #endif

    }
}
