#include "../includes/cuda_helper.h"

namespace simplenet {
    namespace cuda {

        // declaration
        template<typename T>
        void launch_padd_with_zeroes(T* d_in, T* d_out, int batch_size, int in_rows, int in_cols, int padding, cudaStream_t stream = nullptr);

        #ifndef INSTANTIATE_PADDING
        #define INSTANTIATE_PADDING(T) \
            template __global__ void padd_with_zeroes_kernel(const T* __restrict__ d_in, T* __restrict__ d_out, int batch_size, int in_rows,  int in_cols,  int padding); \
            template void launch_padd_with_zeroes<T>(T* d_in, T* d_out, int batch_size, int in_rows, int in_cols, int padding, cudaStream_t stream);
        #endif


    }
}
