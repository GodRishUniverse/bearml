#include "sum.cuh"
#include <cstddef>
#include <cstdint>

namespace simplenet {
    namespace cuda {

        template<typename T>
        __global__ void sum_kernel(const T* d_data, double* result, int64_t size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                atomicAdd(result, static_cast<double>(d_data[idx]));
            }
        }

        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void sum_kernel<__nv_bfloat16>(const __nv_bfloat16 *d_data, double *result, int64_t size);

        // float16
        template __global__ void sum_kernel<__half>(const __half *d_data, double *result, int64_t size);

        // float32
        template __global__ void sum_kernel<float>(const float *d_data, double *result, int64_t size);

        // float64
        template __global__ void sum_kernel<double>(const double *d_data, double *result, int64_t size);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void sum_kernel<int8_t>(const int8_t *d_data, double *result, int64_t size);
        // int16
        template __global__ void sum_kernel<int16_t>(const int16_t *d_data, double *result, int64_t size);
        // int32
        template __global__ void sum_kernel<int32_t>(const int32_t *d_data, double *result, int64_t size);
        // int64
        template __global__ void sum_kernel<int64_t>(const int64_t *d_data, double *result, int64_t size);


        template<typename T>
        void launch_sum_kernel(T *d_data, double *result, int64_t size, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(size, THREAD_COUNT); // Number of blocks
            sum_kernel<T><<<grid, block, 0, stream>>>(d_data, result, size);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }

        // floats
        template void launch_sum_kernel<__nv_bfloat16>(__nv_bfloat16 *d_data, double *result, int64_t size, cudaStream_t stream);
        template void launch_sum_kernel<__half>(__half *d_data, double *result, int64_t size, cudaStream_t stream);
        template void launch_sum_kernel<float>(float *d_data, double *result, int64_t size, cudaStream_t stream);
        template void launch_sum_kernel<double>(double *d_data, double *result, int64_t size, cudaStream_t stream);
        // ints
        template void launch_sum_kernel<int8_t>(int8_t *d_data, double *result, int64_t size, cudaStream_t stream);
        template void launch_sum_kernel<int16_t>(int16_t *d_data, double *result, int64_t size, cudaStream_t stream);
        template void launch_sum_kernel<int32_t>(int32_t *d_data, double *result, int64_t size, cudaStream_t stream);
        template void launch_sum_kernel<int64_t>(int64_t *d_data, double *result, int64_t size, cudaStream_t stream);

    }
}
