#pragma once

#include "../includes/cuda_helper.h"
#include "../../tensor/reductions/reduction_ops.h"
#include <vector>
#include "sum.cuh"
#include <cstdint>

namespace simplenet {
    namespace cuda {

        template <typename T>
        void launch_accumulate_kernel(T *d_data, T *d_out, std::vector<int> data_shape, std::vector<int> out_shape, int64_t size, int64_t offset_new_shape, int64_t offset_old, reductions::ReductionOps op,  bool keepdims, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_ACCUMULATE
        #define INSTANTIATE_ACCUMULATE(T) \
            template void launch_accumulate_kernel<T>(T *d_data, T *d_out, std::vector<int> data_shape, std::vector<int> out_shape, int64_t size, int64_t offset_new_shape, int64_t offset_old, reductions::ReductionOps op, bool keepdims, cudaStream_t stream);
        #endif
    }
}
