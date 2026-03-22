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
    }
}
