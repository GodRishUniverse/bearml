#include "linspace.cuh"

namespace simplenet {
    namespace cuda {

        template<typename T>
        __global__ void linspace( T* out,T start, T step, size_t size){
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx+=blockDim.x * gridDim.x ){
                out[idx] = start + T(idx) * step;
            }
        }

        template<typename T>
        void launch_linspace_kernel(T* d_out, T start, T step, size_t size, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(size, THREAD_COUNT); // Number of blocks
            linspace<T><<<grid, block, 0, stream>>>(d_out, start, step, size);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }

        // floats
        INSTANTIATE_LINSPACE_KERNEL(__nv_bfloat16)
        INSTANTIATE_LINSPACE_KERNEL(__half)
        INSTANTIATE_LINSPACE_KERNEL(float)
        INSTANTIATE_LINSPACE_KERNEL(double)

        // ints
        INSTANTIATE_LINSPACE_KERNEL(int8_t)
        INSTANTIATE_LINSPACE_KERNEL(int16_t)
        INSTANTIATE_LINSPACE_KERNEL(int32_t)
        INSTANTIATE_LINSPACE_KERNEL(int64_t)
    }
}
