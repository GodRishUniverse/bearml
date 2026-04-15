#include "sum.cuh"
#include <cstddef>
#include <cstdint>

namespace simplenet {
    namespace cuda {

        template<typename T>
        __global__ void sum_kernel(const T* d_data, double* result, int64_t size) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  idx < size; idx += blockDim.x * gridDim.x) {
                atomicAdd(result, static_cast<double>(d_data[idx]));
            }
        }

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



        template<typename T>
        __global__ void sum_acc_kernel(const T* d_data,T* d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  idx < size; idx += blockDim.x * gridDim.x) {
                int64_t outer = idx / offset_old;
                int64_t s     = idx % offset_new_shape;
                int64_t out_idx = outer * offset_new_shape + s;
                atomicAdd(&d_out[out_idx], d_data[idx]);
            }
        }

        template<typename T>
        void launch_sum_acc_kernel(T *d_data, T *d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(size, THREAD_COUNT); // Number of blocks
            sum_acc_kernel<T><<<grid, block, 0, stream>>>(d_data, d_out, offset_new_shape, offset_old, size);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }

        // floats
        INSTANTIATE_SUM(__nv_bfloat16);
        INSTANTIATE_SUM(__half);
        INSTANTIATE_SUM(float);
        INSTANTIATE_SUM(double);

        // ints
        // INSTANTIATE_SUM(int8_t);
        // INSTANTIATE_SUM(int16_t);
        INSTANTIATE_SUM(int32_t);
        // INSTANTIATE_SUM(int64_t);

    }
}
