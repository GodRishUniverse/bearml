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
        }
    }
}
