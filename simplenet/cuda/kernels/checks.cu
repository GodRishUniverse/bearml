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





        template <typename T>
        __global__ void check_equal_kernel(const T* data, const T* other, size_t n, int* result) {
            // ensures we dont get outta bounds
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
                // exit early if result is already set
                if (*result != 0) {
                    return;
                }

                float val;
                if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
                    val = fabsf(float(data[idx]) - float(other[idx]));
                } else {
                    val = fabsf(static_cast<float>(data[idx]) - static_cast<float>(other[idx]));
                }
                if (val > 1e-12f) {
                    atomicExch(result, 1); // Set result to 1 if a nonzero element is found
                    // a CUDA C++ device function that atomically exchanges a value in global or shared memory with a new value, returning the old value
                }
            }
        }


        template<typename T>
        bool launch_check_equal_kernel(T *d_data, T *d_other, int64_t size, cudaStream_t stream) {
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

            check_equal_kernel<T><<<grid, block, 0, stream>>>(d_data, d_other, size, d_result);

            cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_result);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
            return h_result == 0; // return true if all elements are equal
        }

        // floats - templates
        INSTANTIATE_CHECKS(__nv_bfloat16)
        INSTANTIATE_CHECKS(__half)
        INSTANTIATE_CHECKS(float)
        INSTANTIATE_CHECKS(double)

        // ints - templates
        INSTANTIATE_CHECKS(int8_t)
        INSTANTIATE_CHECKS(int16_t)
        INSTANTIATE_CHECKS(int32_t)
        INSTANTIATE_CHECKS(int64_t)


    }
}
