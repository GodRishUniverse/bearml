#include "softmax.cuh"
#include "sum.cuh"
#include <cstddef>
#include <cstdint>

namespace simplenet {
    namespace cuda {

        // todo: think about how dim is used in softmax (I know per row is naive and default but generalization is good)
        template<typename T>
        __global__ void softmax_kernel(const T* d_data, T* d_out, int* shape, int64_t size) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  idx < size; idx += blockDim.x * gridDim.x) {
                // one thread per element
            }
            __syncthreads(); // I was thinking of using a shared memory reduction to do a cumulative sum and then divide by the sum
        }


        // floats
        INSTANTIATE_SOFTMAX(__nv_bfloat16);
        INSTANTIATE_SOFTMAX(__half);
        INSTANTIATE_SOFTMAX(float);
        INSTANTIATE_SOFTMAX(double);

        // ints
        // INSTANTIATE_SOFTMAX(int8_t);
        // INSTANTIATE_SOFTMAX(int16_t);
        INSTANTIATE_SOFTMAX(int32_t);
        // INSTANTIATE_SOFTMAX(int64_t);

    }
}
