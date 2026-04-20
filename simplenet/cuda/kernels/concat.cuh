#pragma once

#include "../includes/cuda_helper.h"
#include <vector>

namespace simplenet {
    namespace cuda {
        template <typename T>
        void launch_concat_kernel(T** d_allInputs, int** d_shapes, size_t concat_tensors_size, T* d_result, size_t outerDim, size_t innerDim, size_t dim, size_t concatDim, cudaStream_t stream = nullptr);


        #ifndef INSTANTIATE_CONCAT_KERNEL
        #define INSTANTIATE_CONCAT_KERNEL(T) \
            template void launch_concat_kernel<T>(T** d_allInputs, int** d_shapes, size_t concat_tensors_size, T* d_result, size_t outerDim, size_t innerDim, size_t dim, size_t concatDim, cudaStream_t stream);
        #endif
    }
}
