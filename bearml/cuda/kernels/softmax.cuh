#pragma once

#include "../includes/cuda_helper.h"
#include <cstdint>
#include <vector>
#include "utils.cuh"


namespace bearml {
    namespace cuda {

        template<typename T>
        void launch_softmax_kernel(
            T *d_data,
            T *d_out,
            std::vector<int> shape,
            std::vector<int> stride,
            int64_t size,
            int dim,
            cudaStream_t stream = nullptr
        );


        #ifndef INSTANTIATE_SOFTMAX
        #define INSTANTIATE_SOFTMAX(T) \
            template __global__ void softmax_kernel<T>(const T* d_data, T* d_out,  int* shape, int shape_size, int* stride, int stride_size, int dim); \
            template void launch_softmax_kernel<T>(T *d_data, T *d_out,std::vector<int> shape, std::vector<int> stride, int64_t size, int dim, cudaStream_t stream);
        #endif
    }
}
