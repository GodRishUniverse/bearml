#include "checks.cuh"

namespace simplenet {
    namespace cuda {


        template <typename T>
        __global__ void check_nonzero_kernel(const T* data, size_t n, int* result) {
            // ensures we dont get outta bounds
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
                // exit early if result is already set
                if (*result != 0) {
                    return;
                }

                float val;
                if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
                    val = fabsf(float(data[idx]));
                } else {
                    val = fabsf(static_cast<float>(data[idx]));
                }
                if (val > 1e-12f) {
                    atomicExch(result, 1); // Set result to 1 if a nonzero element is found
                    // a CUDA C++ device function that atomically exchanges a value in global or shared memory with a new value, returning the old value
                }
            }
        }

        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void check_nonzero_kernel<__nv_bfloat16>(const __nv_bfloat16 *d_data, size_t size, int* result);

        // float16
        template __global__ void check_nonzero_kernel<__half>(const __half *d_data, size_t size, int* result);

        // float32
        template __global__ void check_nonzero_kernel<float>(const float *d_data, size_t size, int* result);

        // float64
        template __global__ void check_nonzero_kernel<double>(const double *d_data, size_t size, int* result);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void check_nonzero_kernel<int8_t>(const int8_t *d_data, size_t size, int* result);
        // int16
        template __global__ void check_nonzero_kernel<int16_t>(const int16_t *d_data, size_t size, int* result);
        // int32
        template __global__ void check_nonzero_kernel<int32_t>(const int32_t *d_data, size_t size, int* result);
        // int64
        template __global__ void check_nonzero_kernel<int64_t>(const int64_t *d_data, size_t size, int* result);


        //  launch code for checking non-zero elements
        template<typename T>
        bool launch_check_zero_kernel(T *d_data, int64_t size, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(size, THREAD_COUNT); // Number of blocks

            int h_result = 0;
            int* d_result;
            cudaMalloc(&d_result, sizeof(int));
            cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

            check_nonzero_kernel<T><<<grid, block, 0, stream>>>(d_data, size, d_result);

            cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_result);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

            return h_result != 0;

        }

        // floats - templates
        template bool launch_check_zero_kernel<__nv_bfloat16>(__nv_bfloat16 *d_data, int64_t size, cudaStream_t stream);
        template bool launch_check_zero_kernel<__half>(__half *d_data, int64_t size, cudaStream_t stream);
        template bool launch_check_zero_kernel<float>(float *d_data, int64_t size, cudaStream_t stream);
        template bool launch_check_zero_kernel<double>(double *d_data, int64_t size, cudaStream_t stream);

        // ints - templates
        template bool launch_check_zero_kernel<int8_t>(int8_t *d_data, int64_t size, cudaStream_t stream);
        template bool launch_check_zero_kernel<int16_t>(int16_t *d_data, int64_t size, cudaStream_t stream);
        template bool launch_check_zero_kernel<int32_t>(int32_t *d_data, int64_t size, cudaStream_t stream);
        template bool launch_check_zero_kernel<int64_t>(int64_t *d_data, int64_t size, cudaStream_t stream);


    }
}
