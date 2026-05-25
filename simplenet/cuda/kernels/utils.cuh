#pragma once
#include "../includes/cuda_helper.h"
#include <vector>
#include <cstdint>


namespace simplenet {
    namespace cuda {
        namespace utils {
            template <typename InputT, typename OutputT>
            void launch_dtype_change(InputT *d_data, OutputT *d_out, int64_t size, cudaStream_t stream = nullptr);

            #ifndef INSTANTIATE_DTYPE_CHANGE
            #define INSTANTIATE_DTYPE_CHANGE(InT, OutT) \
                template void launch_dtype_change<InT, OutT>(InT *d_data, OutT *d_out, int64_t size, cudaStream_t stream);
            #endif

            // Walks `src` by (offset, shape, strides) and writes a row-major-contiguous copy into `dst`.
            // h_shape/h_strides are passed from the host and copied to device by the launcher.
            template <typename T>
            void launch_contiguous_gather(
                const T* d_src, T* d_dst, size_t src_offset,
                const size_t* h_shape, const size_t* h_strides, size_t nd, size_t total,
                cudaStream_t stream = nullptr);

            #ifndef INSTANTIATE_CONTIGUOUS_GATHER
            #define INSTANTIATE_CONTIGUOUS_GATHER(Ty) \
                template void launch_contiguous_gather<Ty>(const Ty* d_src, Ty* d_dst, size_t src_offset, \
                    const size_t* h_shape, const size_t* h_strides, size_t nd, size_t total, cudaStream_t stream);
            #endif
        }
    }
}
