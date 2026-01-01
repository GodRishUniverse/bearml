#include "../../../includes/helper.h"

namespace simplenet {
    namespace cuda {

        template <typename T>
        __global__
        void fill_kernel(
            T* data,
            T value,
            size_t n
        ) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = value;
            }
        }

        //TODO: write host code here
        // void fill(double* data, double value, size_t n, int threads) {
        //     dim3 blocks = get_blocks(n, threads);
        //     fill_kernel<<<blocks, threads>>>(data, value, n);
        //     CUDA_CHECK(cudaGetLastError());
        // }
    }
}
