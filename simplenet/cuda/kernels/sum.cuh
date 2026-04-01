#include "../includes/cuda_helper.h"
#include <cstdint>


namespace simplenet {
    namespace cuda {

        template<typename T>
        void launch_sum_kernel(
            T *d_data,
            double *result,
            int64_t size,
            cudaStream_t stream = nullptr
        );

        #ifndef INSTANTIATE_SUM
        #define INSTANTIATE_SUM(T) \
            template __global__ void sum_kernel<T>(const T *d_data, double *result, int64_t size); \
            template void launch_sum_kernel<T>(T *d_data, double *result, int64_t size, cudaStream_t stream);
        #endif
    }
}
