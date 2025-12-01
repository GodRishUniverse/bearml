// https://siboehm.com/articles/22/CUDA-MMM <- link to go over
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernels.h"

//TODO: write broadcasting done on the gpu matmul code

//TODO: write host code here


namespace simplenet {
    namespace cuda {
        // naive kernel right now
        __global__
        void gemm_kernel(
            int batchsize,
            int i,
            int j,
            int k,
            float* __restrict__ a,
            float* __restrict__ b,
            float* c
        ){
            int column = blockIdx.x*blockDim.x + threadIdx.x;
            int row = blockIdx.y*blockDim.y +threadIdx.y;
            int batchId = blockIdx.z *blockDim.z + threadIdx.z;

            if (batchId < batchsize &&  row<i && column<k){
                float sum = 0.0f;
                for (int mid = 0; mid<j; mid++){
                    sum+= a[batchId*i*j+row*j+mid]*b[batchId*j*k+mid*k+column];
                }
                c[batchId*i*k + row*k + column] = sum;
            }
        }

        __global__
        void gemm_kernel(
            int batchsize,
            int i,
            int j,
            int k,
            double* __restrict__ a,
            double* __restrict__ b,
            double* c
        ){
            int column = blockIdx.x*blockDim.x + threadIdx.x;
            int row = blockIdx.y*blockDim.y +threadIdx.y;
            int batchId = blockIdx.z *blockDim.z + threadIdx.z;

            if (batchId < batchsize &&  row<i && column<k){
                double sum = 0.0;
                for (int mid = 0; mid<j; mid++){
                    sum+= a[batchId*i*j+row*j+mid]*b[batchId*j*k+mid*k+column];
                }
                c[batchId*i*k + row*k + column] = sum;
            }
        }

        // void gemm(int batchsize, int i, int j, int k, float* a, float* b, float* c, size_t n, int threads) {
        //     dim3 blocks = simplenet::cuda::get_blocks(n, threads);
        //     gemm_kernel<<<blocks, threads>>>(batchsize, i, j, k, a, b, c);
        //     CUDA_CHECK(cudaGetLastError());
        // }

        // void gemm(int batchsize, int i, int j, int k, double* a, double* b, double* c, size_t n, int threads) {
        //     dim3 blocks = simplenet::cuda::get_blocks(n, threads);
        //     gemm_kernel<<<blocks, threads>>>(batchsize, i, j, k, a, b, c);
        //     CUDA_CHECK(cudaGetLastError());
        // }


    }
}
