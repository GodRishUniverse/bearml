// https://siboehm.com/articles/22/CUDA-MMM <- link to go over
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernels.h"

//TODO: write broadcasting done on the gpu matmul code

//TODO: write host code here


namespace simplenet {
    namespace cuda {
        // naive kernel right now
        template <typename T>
        __global__
        void gemm_kernel(
            int batchsize,
            int i,
            int j,
            int k,
            T* __restrict__ a,
            T* __restrict__ b,
            T* c
        ){
            int column = blockIdx.x*blockDim.x + threadIdx.x;
            int row = blockIdx.y*blockDim.y +threadIdx.y;
            int batchId = blockIdx.z *blockDim.z + threadIdx.z;

            if (batchId < batchsize &&  row<i && column<k){
                T sum {}; // brace initialization
                for (int mid = 0; mid<j; mid++){
                    sum+= a[batchId*i*j+row*j+mid]*b[batchId*j*k+mid*k+column];
                }
                c[batchId*i*k + row*k + column] = sum;
            }
        }

        // void gemm(int batchsize, int i, int j, int k, double* a, double* b, double* c, size_t n, int threads) {
        //     dim3 blocks = simplenet::cuda::get_blocks(n, threads);
        //     gemm_kernel<<<blocks, threads>>>(batchsize, i, j, k, a, b, c);
        //     CUDA_CHECK(cudaGetLastError());
        // }


    }
}
