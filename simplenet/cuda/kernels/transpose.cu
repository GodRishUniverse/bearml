#include "transpose.cuh"
#include <cstdint>

namespace simplenet {
    namespace cuda {

        // TODO: figure out if batch_size >>> blockIdx.z
        template<typename T>
        __global__
        void transpose_kernel(T* input, T* output,  long long int batch_size, int rows, int cols) {
            // When multiple threads  in a warp
            // simultaneously request memory within the same bank in shared memory
            // but across distinct addresses, we say there is a bank conflict.
            // https://modal.com/gpu-glossary/perf/bank-conflict
            __shared__ T tile[TILE_SIZE][TILE_SIZE+1]; // create a shared tile - +1 to avoid bank conflicts

            // get the row and column indices in the tile
            int batch = blockIdx.z; // get the batch index

            int row = blockIdx.y * TILE_SIZE + threadIdx.y; // get the row index
            int col = blockIdx.x * TILE_SIZE + threadIdx.x; // get the column index

            // we first write to shared memory
            if (row < rows && col < cols && batch < batch_size) {
                // row major order
                tile[threadIdx.y][threadIdx.x] = input[batch * rows * cols + row * cols + col];
            }
            __syncthreads(); // synchronize threads before reading from shared memory so that there are no race conditions

            // then we write to global memory
            int transposed_col = blockIdx.y * TILE_SIZE + threadIdx.x;
            int transposed_row = blockIdx.x * TILE_SIZE + threadIdx.y;

            // write to global memory if the transposed indices are within bounds
            if (transposed_row < rows && transposed_col < cols && batch < batch_size) {
                output[batch * rows * cols + transposed_row * cols + transposed_col] = tile[threadIdx.x][threadIdx.y];
            }
        }

        template<typename T>
        void launch_transpose_kernel(T* input, T* output,  long long int batch_size, int rows, int cols, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 blockDim(TILE_SIZE, TILE_SIZE);
            // create a grid with cols and rows tiled by TILE_SIZE (that is why we divide by TILE_SIZE)
            dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE, batch_size);
            transpose_kernel<T><<<gridDim, blockDim>>>(input, output, batch_size, rows, cols);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }

        // floats
        INSTANTIATE_TRANSPOSE(__nv_bfloat16);
        INSTANTIATE_TRANSPOSE(__half);
        INSTANTIATE_TRANSPOSE(float);
        INSTANTIATE_TRANSPOSE(double);

        // ints
        INSTANTIATE_TRANSPOSE(int8_t);
        INSTANTIATE_TRANSPOSE(int16_t);
        INSTANTIATE_TRANSPOSE(int32_t);
        INSTANTIATE_TRANSPOSE(int64_t);

    }
}
