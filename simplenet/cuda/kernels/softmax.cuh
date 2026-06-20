#pragma once

#include "../includes/cuda_helper.h"
#include <cstdint>
#include <vector>


namespace simplenet {
    namespace cuda {

        template<typename T>
        void launch_softmax_kernel(
            T *d_data,
            T *d_out,
            std::vector<int> shape,
            int dim, // where we want softmax to be applied
            cudaStream_t stream = nullptr
        );


        #ifndef INSTANTIATE_SOFTMAX
        #define INSTANTIATE_SOFTMAX(T) \
            template void launch_softmax_kernel<T>(T *d_data, T *d_out, std::vector<int> shape, int dim, cudaStream_t stream);
        #endif
    }
}
