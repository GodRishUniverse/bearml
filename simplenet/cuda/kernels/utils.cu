#include "utils.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

namespace simplenet {
    namespace cuda {
        namespace utils {
            // Kernel to cast elements from one type to another
            template <typename InputT, typename OutputT>
            __global__ void castTypeKernel(const InputT* d_input, OutputT* d_output, int N) {
                for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
                    d_output[idx] = static_cast<OutputT>(d_input[idx]);
                }
            }

            template <typename InputT, typename OutputT>
            void launch_dtype_change(InputT *d_data, OutputT *d_out, int64_t size, cudaStream_t stream ) {
                bool own_stream = (stream == nullptr);
                if (own_stream) {
                    CUDA_CHECK(cudaStreamCreate(&stream));
                }

                dim3 block(THREAD_COUNT); // Threads per block
                // total size of the grid is outerDim
                dim3 grid = get_blocks(size, THREAD_COUNT); // Number of blocks

                castTypeKernel<InputT, OutputT><<<grid, block, 0, stream>>>(d_data, d_out, size);

                CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

                if (own_stream) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaStreamDestroy(stream));
                }
            }

            // floats
            INSTANTIATE_DTYPE_CHANGE(double, float)
            INSTANTIATE_DTYPE_CHANGE(double, __half)
            INSTANTIATE_DTYPE_CHANGE(double, __nv_bfloat16)

            INSTANTIATE_DTYPE_CHANGE(float, __half)
            INSTANTIATE_DTYPE_CHANGE(float, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(float, double)

            INSTANTIATE_DTYPE_CHANGE(__half, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(__half, double)
            INSTANTIATE_DTYPE_CHANGE(__half, float)

            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, __half)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, double)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, float)

            // ints
            INSTANTIATE_DTYPE_CHANGE(int32_t, int64_t)
            INSTANTIATE_DTYPE_CHANGE(int32_t, int16_t)
            INSTANTIATE_DTYPE_CHANGE(int32_t, int8_t)

            INSTANTIATE_DTYPE_CHANGE(int64_t, int32_t)
            INSTANTIATE_DTYPE_CHANGE(int64_t, int16_t)
            INSTANTIATE_DTYPE_CHANGE(int64_t, int8_t)

            INSTANTIATE_DTYPE_CHANGE(int16_t, int32_t)
            INSTANTIATE_DTYPE_CHANGE(int16_t, int64_t)
            INSTANTIATE_DTYPE_CHANGE(int16_t, int8_t)

            INSTANTIATE_DTYPE_CHANGE(int8_t, int32_t)
            INSTANTIATE_DTYPE_CHANGE(int8_t, int64_t)
            INSTANTIATE_DTYPE_CHANGE(int8_t, int16_t)

            // ints and floats
            INSTANTIATE_DTYPE_CHANGE(int32_t, float)
            INSTANTIATE_DTYPE_CHANGE(int64_t, float)
            INSTANTIATE_DTYPE_CHANGE(int16_t, float)
            INSTANTIATE_DTYPE_CHANGE(int8_t, float)

            INSTANTIATE_DTYPE_CHANGE(float, int32_t)
            INSTANTIATE_DTYPE_CHANGE(float, int64_t)
            INSTANTIATE_DTYPE_CHANGE(float, int16_t)
            INSTANTIATE_DTYPE_CHANGE(float, int8_t)

            // ints and doubles
            INSTANTIATE_DTYPE_CHANGE(int32_t, double)
            INSTANTIATE_DTYPE_CHANGE(int64_t, double)
            INSTANTIATE_DTYPE_CHANGE(int16_t, double)
            INSTANTIATE_DTYPE_CHANGE(int8_t, double)

            INSTANTIATE_DTYPE_CHANGE(double, int32_t)
            INSTANTIATE_DTYPE_CHANGE(double, int64_t)
            INSTANTIATE_DTYPE_CHANGE(double, int16_t)
            INSTANTIATE_DTYPE_CHANGE(double, int8_t)

            // ints and __half
            INSTANTIATE_DTYPE_CHANGE(int32_t, __half)
            INSTANTIATE_DTYPE_CHANGE(int64_t, __half)
            INSTANTIATE_DTYPE_CHANGE(int16_t, __half)
            INSTANTIATE_DTYPE_CHANGE(int8_t, __half)

            INSTANTIATE_DTYPE_CHANGE(__half, int32_t)
            INSTANTIATE_DTYPE_CHANGE(__half, int64_t)
            INSTANTIATE_DTYPE_CHANGE(__half, int16_t)
            INSTANTIATE_DTYPE_CHANGE(__half, int8_t)

            // ints and __nv_bfloat16
            INSTANTIATE_DTYPE_CHANGE(int32_t, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(int64_t, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(int16_t, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(int8_t, __nv_bfloat16)

            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int32_t)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int64_t)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int16_t)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int8_t)

        }
    }
}
