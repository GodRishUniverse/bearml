// https://siboehm.com/articles/22/CUDA-MMM <- link to go over
#include "../includes/helper.h"

namespace simplenet {
    namespace cuda {
        //TODO: write broadcasting done on the gpu matmul code

        template <typename T>
        __global__
        void gemm_kernel_broadcast(

        ){
            //
        }


        // naive kernel right now
        template <typename T>
        __global__
        void gemm_kernel_contiguous(
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

        //TODO: write host code here



    }
}
