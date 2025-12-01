#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernels.h"

// __restrict__ keyword usage: https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
namespace simplenet {
    namespace cuda {
        // restrict is an optimization measure that basically tells the compiler that these are separate data
        // this includes broadcasting op
        // TODO: test this
        __global__
        void add_kernel(
            const int* __restrict__ strides_a,
            const int* __restrict__ strides_b,
            const int* __restrict__ res_shape,
            int res_flat_shape,
            int outShapeSize,
            const float* __restrict__ a,
            const float* __restrict__ b,
            float* res
        ){
            int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

            if (thread_idx< res_flat_shape){
                size_t tmp = thread_idx;
                size_t offA = 0;
                size_t offB = 0;

                for (size_t d = outShapeSize - 1; d >= 0; --d) {
                    int coord  = tmp % res_shape[d];
                    tmp /= res_shape[d];
                    offA += coord * strides_a[d];
                    offB += coord * strides_b[d];
                }
               res[thread_idx] = a[offA]+b[offB];
            }
        }

        // double variant
        __global__
        void add_kernel(
            const int* __restrict__ strides_a,
            const int* __restrict__ strides_b,
            const int* __restrict__ res_shape,
            int res_flat_shape,
            int outShapeSize,
            const double* __restrict__ a,
            const double* __restrict__ b,
            double* res
        ){
            int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

            if (thread_idx< res_flat_shape){
                size_t tmp = thread_idx;
                size_t offA = 0;
                size_t offB = 0;

                for (size_t d = outShapeSize - 1; d >= 0; --d) {
                    int coord  = tmp % res_shape[d];
                    tmp /= res_shape[d];
                    offA += coord * strides_a[d];
                    offB += coord * strides_b[d];
                }
               res[thread_idx] = a[offA]+b[offB];
            }
        }

    }
}
