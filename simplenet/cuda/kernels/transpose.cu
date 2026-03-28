#include "transpose.cuh"

namespace simplenet {
    namespace cuda {

        template<typename T>
        __global__
        void transpose_kernel(T* input, T* output, int batch_size, int rows, int cols) {
            __shared__ T tile[TILE_SIZE][TILE_SIZE+1];

            // get the row and column indices in the tile
            int row = blockIdx.y * TILE_SIZE + threadIdx.y;
            int col = blockIdx.x * TILE_SIZE + threadIdx.x;

            // we first write to shared memory
            if (row < rows && col < cols) {
                tile[threadIdx.y][threadIdx.x] = input[col * rows + row];
            }
            __syncthreads();

            // then we write to global memory
            row = blockIdx.y * TILE_SIZE + threadIdx.y;
            col = blockIdx.x * TILE_SIZE + threadIdx.x;
            if (row < rows && col < cols) {
                output[row * cols + col] = tile[threadIdx.x][threadIdx.y];
            }
        }

        template<typename T>
        void launch_transpose_kernel() {

        }
    }
}
