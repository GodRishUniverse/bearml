#include "cuda_kernels.h"
#include <cuda_runtime.h>

//TODO: write broadcasting done on the gpu matmul code

//TODO: write host code here


namespace simplenet {
    namespace cuda {
        // helper
        inline dim3 get_blocks(size_t n, int threads= 256) {
            return dim3((n + threads - 1) / threads);
        }


        __global__ void fill_kernel(double* data, double value, size_t n) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = value;
            }
        }

        // void fill(double* data, double value, size_t n, int threads) {
        //     dim3 blocks = get_blocks(n, threads);
        //     fill_kernel<<<blocks, threads>>>(data, value, n);
        //     CUDA_CHECK(cudaGetLastError());
        // }

        __global__ void fill_kernel(float* data, float value, size_t n) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = value;
            }
        }

        // void fill(float* data, float value, size_t n, int threads) {
        //     dim3 blocks = get_blocks(n, threads);
        //     fill_kernel<<<blocks, threads>>>(data, value, n);
        //     CUDA_CHECK(cudaGetLastError());
        // }

    }
}
