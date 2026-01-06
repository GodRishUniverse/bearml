#include "../includes/helper.h"
#include <cstdint>

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


        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void simplenet::cuda::fill_kernel<__nv_bfloat16>(__nv_bfloat16 *data, __nv_bfloat16 value, size_t);

        // float16
        template __global__ void simplenet::cuda::fill_kernel<__half>(__half *data, __half value, size_t);

        // float32
        template __global__ void simplenet::cuda::fill_kernel<float>(float *data, float value, size_t);

        // float64
        template __global__ void simplenet::cuda::fill_kernel<double>(double *data, double value, size_t);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void simplenet::cuda::fill_kernel<int8_t>(int8_t *data, int8_t value, size_t);
        // int16
        template __global__ void simplenet::cuda::fill_kernel<int16_t>(int16_t *data, int16_t value, size_t);
        // int32
        template __global__ void simplenet::cuda::fill_kernel<int32_t>(int32_t *data, int32_t value, size_t);
        // int64
        template __global__ void simplenet::cuda::fill_kernel<int64_t>(int64_t *data, int64_t value, size_t);


        template <typename T>
        void launch_fill(T* data, T value,  std::vector<int>& res_shape, cudaStream_t stream = nullptr) {


            bool own_stream = (stream == nullptr);
            // if we do need to create a stream then we create it here
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            size_t n = 1;
            for (size_t d = 0; d < res_shape.size(); ++d) {
                n *= res_shape[d];
            }

            // Configuring kernel launch
            dim3 block(THREAD_COUNT);
            dim3 grid((n+THREAD_COUNT - 1) / THREAD_COUNT);

            fill_kernel<T><<<grid, block,0, stream>>>(data, value, n);
            CUDA_CHECK(cudaGetLastError());

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }
    }
}
