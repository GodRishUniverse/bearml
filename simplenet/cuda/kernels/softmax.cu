#include "softmax.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace simplenet {
    namespace cuda {

        // softmax max trick is that you subtract the max value from each element before taking the exp (as it gives the same result as not doing that and makes sure large values don't overflow)
        template<typename T>
        __global__ void softmax_kernel(const T* d_data, T* d_out,  int* shape, int shape_size, int* stride, int stride_size, int dim, int64_t size) {
            int shape_dim = shape[dim];
            int stride_dim = stride[dim];

            int row_base  = 0 ;
            int group = blockIdx.x ;// This row is assigned to this block
            // TODO: rowbase calculation


            //

            // shared memory for cumulative sum  = number of rows basically
            // example is like 4*5 will need 1 float per row (4 floats)
            // so shape[dim] * sizeof(T) is the size of the shared memory per block
            extern __shared__ T smemory[]; // needs to be declared outside the kernel in the <<<...>>> call with the shared memory size (has to be shape[dim] * sizeof(T))

            // local max for each thread block (partial max)
            T local_max = -std::numeric_limits<T>::infinity(); // in register
            for (int i = threadIdx.x; i < shape_dim; i += blockDim.x) {
                local_max = max(local_max, d_data[row_base+ i * stride_dim]);
            }
            smemory[threadIdx.x] = local_max; // store local max in shared memory (avoids data race to use their own block of memory)
            __syncthreads(); // synchronize to ensure all threads have written their local max to shared memory
            // Carry out a tree reduction to find the global max (LLMs helped me figure this out as atomicMax would have serialized this to O(N) rather than O(log N))
            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (threadIdx.x < s) smemory[threadIdx.x] = max(smemory[threadIdx.x], smemory[threadIdx.x+s]);
                __syncthreads();
            }
            T row_max = smemory[0];
            __syncthreads();

            // similar to the tree reduction above, but for the local row sum

            T local_row_sum = 0; // local sum for each thread block (partial sum)
            for (int i = threadIdx.x; i < shape_dim; i += blockDim.x) {
                local_row_sum += exp(d_data[row_base+   i * stride_dim]-row_max);;
            }
            smemory[threadIdx.x] = local_row_sum; // store local sum in shared memory (avoids data race to use their own block of memory)
            __syncthreads(); // synchronize to ensure all threads have written their local sum to shared memory

            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (threadIdx.x < s)
                    smemory[threadIdx.x] = smemory[threadIdx.x] + smemory[threadIdx.x+s];
                __syncthreads();
            }
            T row_sum = smemory[0];
            __syncthreads();

            // apply the divide after the cumulative sum is computed
            for (size_t idx = threadIdx.x;  idx < shape_dim; idx += blockDim.x) {
                // one thread per element
                d_out[row_base+idx] = exp(d_data[row_base+idx]-row_max) / row_sum;
            }
        }


        // floats
        INSTANTIATE_SOFTMAX(__nv_bfloat16);
        INSTANTIATE_SOFTMAX(__half);
        INSTANTIATE_SOFTMAX(float);
        INSTANTIATE_SOFTMAX(double);

        // ints
        // INSTANTIATE_SOFTMAX(int8_t);
        // INSTANTIATE_SOFTMAX(int16_t);
        INSTANTIATE_SOFTMAX(int32_t);
        // INSTANTIATE_SOFTMAX(int64_t);

    }
}
