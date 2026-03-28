#include "../includes/cuda_helper.h"
#include <cstdint>


namespace simplenet {
    namespace cuda {

        template<typename T>
        void launch_transpose_kernel(T* input, T* output, long long int batch_size, int rows, int cols, cudaStream_t stream= nullptr);

    }
}
