#pragma once

#include "../includes/cuda_helper.h"

namespace bearml {
    namespace cuda {

        // declaration
        // out_rows/out_cols and the r_start/r_end/c_start/c_end overlap window are passed
        // in rather than re-derived from `padding` here, since the caller (neural_network::padding)
        // already computes all of these once as part of output_shape -- re-deriving them a second
        // time would just be a second source of truth that could drift from what was actually allocated.
        template<typename T>
        void launch_padd_with_constant(T* d_in, T* d_out, int batch_size, int in_rows, int in_cols, int out_rows, int out_cols, int padding, int r_start, int r_end, int c_start, int c_end, T constant, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_PADDING
        #define INSTANTIATE_PADDING(T) \
            template __global__ void padd_with_constant_kernel(const T* __restrict__ d_in, T* __restrict__ d_out, int batch_size, int in_rows, int in_cols, int out_rows, int out_cols, int padding, int r_start, int r_end, int c_start, int c_end); \
            template void launch_padd_with_constant<T>(T* d_in, T* d_out, int batch_size, int in_rows, int in_cols, int out_rows, int out_cols, int padding, int r_start, int r_end, int c_start, int c_end, T constant, cudaStream_t stream);
        #endif


    }
}
