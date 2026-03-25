#include "../includes/cuda_helper.h"
#include <cstdint>

namespace simplenet {
    namespace cuda {

        template<typename T>
        bool launch_check_zero_kernel(T *d_data, int64_t size, cudaStream_t stream = nullptr);
    }
}
