#include "concat.cuh"
#include <numeric>


namespace simplenet {
    namespace cuda {

        // TODO: figure out indexing out here
        template <typename T>
        __global__ void concat_kernel(T** allInputs, int** shapes, size_t elements_count, T* result,  size_t outerDim, size_t innerDim, size_t dim, size_t concatDim ) {
            for (size_t o = blockIdx.x * blockDim.x + threadIdx.x; o < outerDim; o += blockDim.x * gridDim.x) {
                int offset = 0; // offset will be used to move the pointer to execute the copies
                for (size_t i = 0; i < elements_count; ++i) {
                    int src_cat_dim = shapes[i][dim];
                    int copy_size = src_cat_dim * innerDim;
                    // we copy the data for each tensor into the result tensor
                    // outer*concatDim * innerDim moves the pointer to the correct position in the result tensor
                    // offset * innerDim is the offset that will be copied from the source tensor using (o*src_cat_dim*innerDim)
                    memcpy(result + o*concatDim *innerDim + offset*innerDim, allInputs[i] + o*src_cat_dim*innerDim, copy_size * sizeof(T));
                    offset += src_cat_dim;
                }
            }
        }


        template <typename T>
        void launch_concat_kernel(T** d_allInputs, int** d_shapes, size_t concat_tensors_size,  T* d_result, size_t outerDim, size_t innerDim, size_t dim, size_t concatDim, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 block(THREAD_COUNT); // Threads per block
            // total size of the grid is outerDim
            dim3 grid = get_blocks(outerDim, THREAD_COUNT); // Number of blocks

            concat_kernel<T><<<grid, block, 0, stream>>>(d_allInputs, d_shapes, concat_tensors_size, d_result, outerDim, innerDim, dim, concatDim);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }

        // floats
        INSTANTIATE_CONCAT_KERNEL(float)
        INSTANTIATE_CONCAT_KERNEL(double)
        INSTANTIATE_CONCAT_KERNEL(__nv_bfloat16)
        INSTANTIATE_CONCAT_KERNEL(__half)

        // ints
        INSTANTIATE_CONCAT_KERNEL(int32_t)
        INSTANTIATE_CONCAT_KERNEL(int64_t)
        INSTANTIATE_CONCAT_KERNEL(int16_t)
        // INSTANTIATE_CONCAT_KERNEL(int8_t)
    }
}
