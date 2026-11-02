#include "fill.cuh"
#include <cassert>

// We only need cudaMalloc/cudaMemcpy for:
//   - Arrays/vectors (like your strides_a, strides_b)
//   - Large structs that you want to pass by pointer
//   - Data that the kernel will modify and you need back on host

namespace bearml {
    namespace cuda {

        template <typename T>
        __global__
        void fill_kernel(
            T* data,
            T value,
            size_t n
        ) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < n;
                 idx += blockDim.x * gridDim.x) {
                data[idx] = value;
            }
        }


        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void bearml::cuda::fill_kernel<__nv_bfloat16>(__nv_bfloat16 *data, __nv_bfloat16 value, size_t);

        // float16
        template __global__ void bearml::cuda::fill_kernel<__half>(__half *data, __half value, size_t);

        // float32
        template __global__ void bearml::cuda::fill_kernel<float>(float *data, float value, size_t);

        // float64
        template __global__ void bearml::cuda::fill_kernel<double>(double *data, double value, size_t);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void bearml::cuda::fill_kernel<int8_t>(int8_t *data, int8_t value, size_t);
        // int16
        template __global__ void bearml::cuda::fill_kernel<int16_t>(int16_t *data, int16_t value, size_t);
        // int32
        template __global__ void bearml::cuda::fill_kernel<int32_t>(int32_t *data, int32_t value, size_t);
        // int64
        template __global__ void bearml::cuda::fill_kernel<int64_t>(int64_t *data, int64_t value, size_t);




        template <typename T>
        __global__
        void identityMatrixGPU(
            T* data,
            size_t rows, // same as columns
            size_t batch
        ) {

            for (
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x,
                      idy = blockIdx.y * blockDim.y + threadIdx.y,
                      b = blockIdx.z * blockDim.z + threadIdx.z;

                      idx < rows && idy < rows && b < batch;

                      idx += blockDim.x * gridDim.x,
                      idy += blockDim.y * gridDim.y,
                      b += blockDim.z * gridDim.z
            ) {
                size_t offset = b*rows*rows+idy*rows + idx;
                data[offset] = (idx==idy) ? T(1.0) : T(0.0);
            }
        }



        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void bearml::cuda::identityMatrixGPU<__nv_bfloat16>(__nv_bfloat16 *data, size_t, size_t);

        // float16
        template __global__ void bearml::cuda::identityMatrixGPU<__half>(__half *data, size_t, size_t);

        // float32
        template __global__ void bearml::cuda::identityMatrixGPU<float>(float *data, size_t, size_t);

        // float64
        template __global__ void bearml::cuda::identityMatrixGPU<double>(double *data,  size_t, size_t);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void bearml::cuda::identityMatrixGPU<int8_t>(int8_t *data, size_t, size_t);
        // int16
        template __global__ void bearml::cuda::identityMatrixGPU<int16_t>(int16_t *data, size_t, size_t);
        // int32
        template __global__ void bearml::cuda::identityMatrixGPU<int32_t>(int32_t *data, size_t, size_t);
        // int64
        template __global__ void bearml::cuda::identityMatrixGPU<int64_t>(int64_t *data, size_t, size_t);



        template <typename T>
        void launch_iden(T* d_data, std::vector<int>& res_shape, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            size_t ndim = res_shape.size();

            assert(ndim >= 2);
            size_t rows = static_cast<size_t>(res_shape[ndim - 1]);

            assert(rows==res_shape[ndim-2]);
            // if we do need to create a stream then we create it here
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            size_t batch = 1;
            for (size_t i = 0; i < ndim - 2; ++i) {
                batch *= static_cast<size_t>(res_shape[i]);
            }

            // Configuring kernel launch
            dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
            dim3 grid((rows+BLOCK_SIZE-1) / BLOCK_SIZE, (rows+BLOCK_SIZE - 1) / BLOCK_SIZE, batch);

            identityMatrixGPU<T><<<grid, block,0, stream>>>(d_data, rows, batch);
            CUDA_CHECK(cudaGetLastError());



            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }

        // template specification
        // Ints
        template void launch_iden<int8_t>(int8_t*, std::vector<int>&, cudaStream_t);
        template void launch_iden<int16_t>(int16_t*, std::vector<int>&, cudaStream_t);
        template void launch_iden<int32_t>(int32_t*, std::vector<int>&, cudaStream_t);
        template void launch_iden<int64_t>(int64_t*, std::vector<int>&, cudaStream_t);

        // floats
        template void launch_iden<float>(float*, std::vector<int>&, cudaStream_t);
        template void launch_iden<double>(double*, std::vector<int>&, cudaStream_t);
        template void launch_iden<__half>(__half*, std::vector<int>&, cudaStream_t);
        template void launch_iden<__nv_bfloat16>(__nv_bfloat16*, std::vector<int>&, cudaStream_t);



        //  - right now the Launch code -> d_a, d_b and d_out already on device as the variable name implies
        template <typename T>
        void launch_fill(T* d_data, T value,  std::vector<int>& res_shape, cudaStream_t stream) {
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

            fill_kernel<T><<<grid, block,0, stream>>>(d_data, value, n);
            CUDA_CHECK(cudaGetLastError());

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }

        // template specification
        // Ints
        template void launch_fill<int8_t>(int8_t*, int8_t, std::vector<int>&, cudaStream_t);
        template void launch_fill<int16_t>(int16_t*, int16_t, std::vector<int>&, cudaStream_t);
        template void launch_fill<int32_t>(int32_t*, int32_t, std::vector<int>&, cudaStream_t);
        template void launch_fill<int64_t>(int64_t*, int64_t, std::vector<int>&, cudaStream_t);

        // floats
        template void launch_fill<float>(float*, float, std::vector<int>&, cudaStream_t);
        template void launch_fill<double>(double*, double, std::vector<int>&, cudaStream_t);
        template void launch_fill<__half>(__half*, __half, std::vector<int>&, cudaStream_t);
        template void launch_fill<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16, std::vector<int>&, cudaStream_t);
    }
}
